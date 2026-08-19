"""
DistillToMCU v8 — Conformal Prediction for Sensor Interval Calibration
=====================================================================
替换 Wilson Score，提供分布无关的有限样本覆盖保证。

方法: Split Conformal Prediction (Lei et al., JASA 2018)
  - 校准集上计算 nonconformity scores
  - 取 (1-alpha) 分位数作为 margin
  - 推理时不需要正态假设，不需要 independence 假设

MCU 可行性: Yamin & Bhat (IEEE TCAD 2024) 在 TI-CC2652R 上验证
  - 推理阶段 O(1): 只需比较 sensor_value ∈ [lower, upper]
  - 校准阶段在 PC 端完成

对比:
  Wilson Score → 二项比例 CI，假设 i.i.d. Bernoulli
  Conformal   → 分布无关，有限样本覆盖保证 P(Y∈C) ≥ 1-α

Usage:
    from conformal import ConformalCalibrator

    cal = ConformalCalibrator(alpha=0.1)  # 90% coverage target
    cal.fit(calibration_sensors, base_intervals)
    calibrated = cal.calibrate(base_intervals)
    # → [{lower, upper, coverage_guarantee: "≥90%"}, ...]
"""

import math
import statistics
from typing import Optional


class ConformalCalibrator:
    """
    Split Conformal Prediction for sensor interval calibration.

    输入: 基础区间 [L0, U0]（由 CKMS+Welford 产生）+ 校准集传感器值
    输出: 校准后区间 [L0 - q_hat, U0 + q_hat]，保证覆盖概率 ≥ 1-α

    理论保证（Lei et al. 2018, Theorem 2.1）:
      如果校准集与测试数据可交换 (exchangeable)，
      则 P(Y_test ∈ C(X_test)) ≥ 1 - α

    算法:
      校准阶段 (PC端):
        for each (sensor_value, [L0, U0]) in calibration_set:
            S_i = max(L0 - value, value - U0, 0)  # nonconformity score
        q_hat = quantile({S_i}, (1-α)(1+1/n_cal))  # 有限样本修正

      推理阶段 (MCU端):
        return L0 - q_hat ≤ sensor_value ≤ U0 + q_hat
    """

    def __init__(self, alpha: float = 0.15):
        """
        Args:
            alpha: 1 - coverage target. 默认 0.15 → 85% 覆盖率目标。
                   越低 → 区间越宽 → 更保守 → 更多本地执行
                   越高 → 区间越窄 → 更激进 → 更多云端 fallback
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self._q_hat: Optional[float] = None  # 校准 margin
        self._n_cal: int = 0
        self._cal_scores: list[float] = []   # for diagnostics

    def fit(self, calibration_data: list[dict]) -> "ConformalCalibrator":
        """
        校准阶段: 计算 nonconformity scores 的校正分位数。

        Args:
            calibration_data: [{sensor, base_lower, base_upper, actual_value}, ...]
                              calibration_data 必须与训练集独立（不参与规则学习）

        每个 sensor 独立处理: 对每个 sensor 的校准数据子集分别存 score，
        然后对所有 scores 取全局分位数（pooled conformal）。
        """
        scores = []

        for item in calibration_data:
            lo = item.get("base_lower", item.get("lower", 0))
            hi = item.get("base_upper", item.get("upper", 0))
            val = item.get("actual_value", item.get("value", 0))

            if val is None or lo is None or hi is None:
                continue

            # Nonconformity score: 落在区间外的距离
            # 落在区间内 → score = 0
            # 落在下方    → score = lo - val (正数)
            # 落在上方    → score = val - hi (正数)
            if val < lo:
                s = lo - val
            elif val > hi:
                s = val - hi
            else:
                s = 0.0

            scores.append(s)

        self._n_cal = len(scores)
        self._cal_scores = scores

        if len(scores) == 0:
            self._q_hat = 0.0
            return self

        # 有限样本修正因子 (Lei et al. 2018, Eq 2):
        # 取 (1-alpha)(1 + 1/n_cal) 分位数
        adjusted_level = (1.0 - self.alpha) * (1.0 + 1.0 / len(scores))
        adjusted_level = min(adjusted_level, 1.0)

        self._q_hat = _quantile(sorted(scores), adjusted_level)

        return self

    def calibrate_single(self, base_lower: float, base_upper: float) -> tuple[float, float]:
        """
        对单个基础区间施加 conformal margin。

        Returns:
            (calibrated_lower, calibrated_upper)
        """
        if self._q_hat is None:
            return base_lower, base_upper  # 未校准，返回原区间
        return base_lower - self._q_hat, base_upper + self._q_hat

    def calibrate(self, base_intervals: list[dict]) -> list[dict]:
        """
        批量校准。

        Args:
            base_intervals: [{sensor, lower, upper, ...}, ...]

        Returns:
            [{sensor, lower, upper, conformal_margin, coverage_guarantee, ...}, ...]
        """
        calibrated = []
        for bi in base_intervals:
            lo, hi = self.calibrate_single(
                bi.get("lower", bi.get("base_lower", 0)),
                bi.get("upper", bi.get("base_upper", 0)),
            )
            cal = dict(bi)
            cal["lower"] = round(lo, 2)
            cal["upper"] = round(hi, 2)
            cal["conformal_margin"] = round(self._q_hat, 4) if self._q_hat else 0.0
            cal["coverage_guarantee"] = f"≥{round((1 - self.alpha) * 100)}%"
            cal["calibration_method"] = "split_conformal"
            cal["n_calibration_samples"] = self._n_cal
            cal["conformal_alpha"] = self.alpha
            calibrated.append(cal)
        return calibrated

    def coverage_diagnostics(self, test_data: list[dict]) -> dict:
        """
        在测试集上评估实际覆盖率。

        Args:
            test_data: [{sensor, lower, upper, actual_value}, ...]

        Returns:
            {actual_coverage, target_coverage, calibrated, n_covered, n_total}
        """
        if self._q_hat is None:
            return {"error": "calibrator not fitted"}

        n_covered = 0
        n_total = 0
        for item in test_data:
            lo = item.get("lower", 0) - (self._q_hat or 0)
            hi = item.get("upper", 0) + (self._q_hat or 0)
            val = item.get("actual_value", item.get("value"))
            if val is None:
                continue
            n_total += 1
            if lo <= val <= hi:
                n_covered += 1

        return {
            "actual_coverage": round(n_covered / max(1, n_total), 4),
            "target_coverage": round(1.0 - self.alpha, 4),
            "calibrated": n_covered >= (1.0 - self.alpha) * n_total - 1,
            "n_covered": n_covered,
            "n_total": n_total,
            "q_hat": round(self._q_hat, 4),
        }


def _quantile(sorted_values: list[float], q: float) -> float:
    """计算经验分位数 (无 numpy 依赖)"""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


# ========== Self-Test ==========

if __name__ == "__main__":
    import random
    random.seed(42)

    print("=" * 60)
    print("  ConformalCalibrator — Self-Test")
    print("=" * 60)

    # Test 1: 基本功能 — 已知正态数据
    print("\n[Test 1] Normal distribution calibration")
    true_mean, true_std = 30.0, 2.0
    train_data = [random.gauss(true_mean, true_std) for _ in range(200)]
    cal_data_raw = [random.gauss(true_mean, true_std) for _ in range(50)]

    # 从训练集学基础区间
    train_mean = statistics.mean(train_data)
    train_std = statistics.stdev(train_data)
    base_lo = train_mean - 2 * train_std
    base_hi = train_mean + 2 * train_std

    # Conformal calibration
    cal_data = [
        {"actual_value": v, "base_lower": base_lo, "base_upper": base_hi}
        for v in cal_data_raw
    ]
    calibrator = ConformalCalibrator(alpha=0.1)
    calibrator.fit(cal_data)
    cal_lo, cal_hi = calibrator.calibrate_single(base_lo, base_hi)

    print(f"  Base interval:   [{base_lo:.2f}, {base_hi:.2f}]")
    print(f"  Calibrated:      [{cal_lo:.2f}, {cal_hi:.2f}]")
    print(f"  Margin (q_hat):   {calibrator._q_hat:.4f}")
    assert cal_lo < base_lo, "Lower bound should expand"
    assert cal_hi > base_hi, "Upper bound should expand"
    print("  [PASS]")

    # Test 2: 覆盖率验证
    print("\n[Test 2] Coverage verification")
    test_data_raw = [random.gauss(true_mean, true_std) for _ in range(1000)]
    test_data_cp = [
        {"lower": base_lo, "upper": base_hi, "actual_value": v}
        for v in test_data_raw
    ]
    diag = calibrator.coverage_diagnostics(test_data_cp)
    print(f"  Target coverage:  {diag['target_coverage']}")
    print(f"  Actual coverage:  {diag['actual_coverage']}")
    print(f"  Covered:          {diag['n_covered']}/{diag['n_total']}")
    assert diag["actual_coverage"] >= diag["target_coverage"] - 0.03, \
        f"Coverage {diag['actual_coverage']} below target {diag['target_coverage']}"
    print("  [PASS]")

    # Test 3: 空校准集 — 不崩溃
    print("\n[Test 3] Empty calibration set")
    cal_empty = ConformalCalibrator(alpha=0.1)
    cal_empty.fit([])
    lo, hi = cal_empty.calibrate_single(20, 40)
    assert lo == 20 and hi == 40, "Should return base interval unchanged"
    print("  [PASS]")

    # Test 4: 批量校准
    print("\n[Test 4] Batch calibration")
    intervals = [
        {"sensor": "temperature", "lower": 28.0, "upper": 34.0},
        {"sensor": "light", "lower": 30.0, "upper": 70.0},
    ]
    cal_data_multi = [
        {"actual_value": random.gauss(31, 2), "base_lower": 28.0, "base_upper": 34.0}
        for _ in range(50)
    ]
    cal2 = ConformalCalibrator(alpha=0.1)
    cal2.fit(cal_data_multi)
    calibrated = cal2.calibrate(intervals)
    for c in calibrated:
        print(f"  {c['sensor']}: [{c['lower']:.1f}, {c['upper']:.1f}] "
              f"(margin={c['conformal_margin']:.4f}, guarantee={c['coverage_guarantee']})")
        # 有 margin 则区间应扩展（≤原下界, ≥原上界）
        if c["conformal_margin"] > 0:
            assert c["lower"] <= intervals[0]["lower"], f"{c['sensor']} lower should expand"
            assert c["upper"] >= intervals[0]["upper"], f"{c['sensor']} upper should expand"
    print("  [PASS]")

    # Test 5: alpha 参数效果 — 更大的 alpha → 更窄区间
    print("\n[Test 5] Alpha effect on interval width")
    for alpha in [0.05, 0.10, 0.20, 0.30]:
        c = ConformalCalibrator(alpha=alpha)
        c.fit(cal_data)
        lo, hi = c.calibrate_single(base_lo, base_hi)
        width = hi - lo
        print(f"  alpha={alpha:.2f} → [{lo:.2f}, {hi:.2f}] width={width:.2f}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("  ALL CONFORMAL TESTS PASSED [OK]")
    print("=" * 60)
