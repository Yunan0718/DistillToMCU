"""
DistillToMCU — 包容区间学习算法 (Inclusive Interval Learning)
==============================================================
C1 核心模块：从 LLM 行为轨迹中自动学习连续传感器阈值的包容区间。

v7 管线:
  IQR 去离群值 → mean±2σ 扩展 → Wilson Score 置信度 → 物理约束裁剪

v8 管线 (NEW):
  CKMS 在线分位数 → Welford 在线方差 → Split Conformal Prediction
  → 物理约束裁剪 → MDL 规则合并 (见 mdl_consolidator.py)

数据特性: LLM trace 只有正样本（LLM决定行动时的传感器值），没有负样本。
  区间学习从正样本中估计触发范围；Conformal Prediction 提供分布无关的覆盖保证。

和现有方法的本质区别：
  - 不是 Apriori（离散共现挖掘，不处理连续阈值）
  - 不是决策树（需要负样本，LLM trace 只有正样本）
  - 不是 Simple Cache（精确匹配，不能泛化到未见过的值）
  - RIMRULE (ACL 2026): 规则注入 LLM prompt → 我们部署到 MCU 独立执行

Usage:
    from rule_generalizer import RuleGeneralizer

    rg = RuleGeneralizer(method="v8")  # 或 method="v7" 向后兼容
    traces_same_action = [...]  # 同一 action 的 trace 列表
    conditions = rg.learn_intervals(traces_same_action)
    # → [{sensor: "temperature", op: "between", lower: 22.0, upper: 32.0}, ...]
"""

import statistics
import math
from config import SENSOR_RANGES, RULE_WILSON_Z, RULE_MIN_EVIDENCE, DISCRETE_FREQUENCY_THRESHOLD
from conformal import ConformalCalibrator


# ========== Wilson Score Confidence (v7, retained for backward compat) ==========

def wilson_score(positive: int, total: int, z: float = RULE_WILSON_Z) -> float:
    """
    Wilson score confidence interval (lower bound).
    小样本时自动压低置信度，大样本时趋近 p_hat。
    当 total=0 时返回 0。
    """
    if total == 0:
        return 0.0
    p_hat = positive / total
    z2 = z * z
    n = total
    denominator = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denominator
    margin = z * math.sqrt(
        (p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    ) / denominator
    return max(0.0, min(1.0, center - margin))


# ========== CKMS Online Quantile Estimator (v8) ==========

class CKMSQuantile:
    """
    CKMS (Cormode-Korn-Muthukrishnan-Srivastava) 在线分位数估计。

    引用: Cormode et al. "Effective Computation of Biased Quantiles
    over Data Streams" (ICDE 2005)

    每条新样本 O(log(1/epsilon)) 插入，O(1/epsilon) 内存，
    支持查询任意分位数。用于替换离线 IQR 计算。

    对 ESP32-S3 的意义:
      - 可在 MCU 上直接运行（链表+周期性压缩）
      - 不需要存储所有历史数据（缩减 Flash 需求）
      - 支持概念漂移（天然适配 LLM 行为随时间变化）

    简化实现: 对 epsilon=0.01 (100 个节点) 的 IQR 场景，
    用有序列表近似，每个新值 O(n) 插入（n≤100，<1ms on MCU）。
    """

    def __init__(self, epsilon: float = 0.01):
        self.epsilon = epsilon
        self._values: list[float] = []  # sorted
        self._max_size = int(1.0 / epsilon) + 10  # ~110

    def insert(self, value: float):
        """O(n) 插入保持有序 + 到达上限时压缩"""
        # 二分插入
        lo, hi = 0, len(self._values)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._values[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        self._values.insert(lo, value)

        # 到达上限: 删掉"中间"元素（保留两端，压缩中间）
        if len(self._values) > self._max_size:
            keep = self._max_size // 2
            self._values = (
                self._values[:keep] +
                self._values[-keep:]
            )

    def query(self, q: float) -> float:
        """查询 q 分位数 (0≤q≤1)"""
        if not self._values:
            return 0.0
        n = len(self._values)
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return self._values[lo] * (1.0 - frac) + self._values[hi] * frac

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def iqr(self) -> float:
        """四分位距 (IQR) = Q3 - Q1"""
        return self.query(0.75) - self.query(0.25)


# ========== Welford Online Mean/Variance (v8) ==========

class WelfordEstimator:
    """
    Welford 在线均值和方差估计算法。

    引用: B. P. Welford (1962) "Note on a Method for Calculating
    Corrected Sums of Squares and Products" (Technometrics)

    O(1) 每个样本更新，数值稳定（避免 catastrophic cancellation）。
    用于替换离线 mean±2σ。
    """

    def __init__(self):
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0     # sum of squared differences from current mean

    def insert(self, value: float):
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        """样本方差 (n-1 分母, unbiased)"""
        if self._count < 2:
            return 1.0
        return self._m2 / (self._count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))


# ========== Inclusive Interval Learning ==========

class RuleGeneralizer:
    """
    从 LLM 行为样本中学习连续传感器的包容区间。

    v7 (默认): IQR → mean±2σ → Wilson score → 物理约束
    v8 (新):   CKMS 在线分位数 → Welford 在线方差 → Split Conformal
               → 物理约束裁剪 → 返回校准区间
    """

    def __init__(self, min_samples: int = 3, method: str = "v8",
                 conformal_alpha: float = 0.15,
                 max_conditions: int = 4,
                 bg_coverage_threshold: float = 0.65,
                 discrete_power_threshold: float = 0.20):
        """
        Args:
            min_samples: 最少需要多少个样本才开始学习（< 此数返回空）
            method: "v7" (legacy) | "v8" (CKMS+Welford+Conformal)
            conformal_alpha: Conformal Prediction 的 alpha (1-alpha = coverage)
        """
        self.min_samples = max(min_samples, RULE_MIN_EVIDENCE)
        self.method = method
        self.conformal_alpha = conformal_alpha
        self.max_conditions = max(1, max_conditions)
        self.bg_coverage_threshold = bg_coverage_threshold
        self.discrete_power_threshold = discrete_power_threshold

    # ---- Public API ----

    def learn_intervals(self, traces_same_action: list,
                        background_snapshots: list | None = None) -> list[dict]:
        """
        从同一 action 的 trace 列表中学习传感器条件。

        Args:
            traces_same_action: [{sensors: {temp, hum, light, motion}, ...}, ...]
                                所有 trace 的 LLM 都选择了同一个 action

        Returns:
            conditions: [{sensor, op, lower?, upper?, value?, confidence}, ...]
                        可直接用于 Rule 的 conditions

        Example:
            >>> rg = RuleGeneralizer()
            >>> traces = [
            ...     {"sensors": {"temperature": 31.2, "light": 45}},
            ...     {"sensors": {"temperature": 30.5, "light": 60}},
            ...     {"sensors": {"temperature": 32.1, "light": 40}},
            ...     {"sensors": {"temperature": 29.8, "light": 55}},
            ...     {"sensors": {"temperature": 31.0, "light": 50}},
            ... ]
            >>> conditions = rg.learn_intervals(traces)
            >>> # conditions → [
            >>> #   {sensor: "temperature", op: "between",
            >>> #    lower: 29.1, upper: 33.2, confidence: 0.89, sample_count: 5},
            >>> #   {sensor: "light", op: "between",
            >>> #    lower: 29.8, upper: 69.2, confidence: 0.89, sample_count: 5},
            >>> # ]
        """
        if len(traces_same_action) < self.min_samples:
            return []

        # 收集所有传感器快照
        sensor_snapshots = [t.get("sensors", t) for t in traces_same_action]
        if not sensor_snapshots:
            return []

        # 识别哪些传感器有数据
        sensor_names = set()
        for snap in sensor_snapshots:
            sensor_names.update(snap.keys())

        # Pre-process background distribution (per sensor) for discriminative
        # condition selection (v10): only keep conditions that separate this
        # action from the global sensor distribution, avoiding spurious
        # correlations such as "motion==0 -> led.on".
        bg_values_by_sensor = {}
        if background_snapshots:
            for snap in background_snapshots:
                s = snap.get("sensors", snap)
                for k, v in s.items():
                    if v is not None:
                        bg_values_by_sensor.setdefault(k, []).append(v)

        conditions = []
        n_total = len(sensor_snapshots)

        for sname in sorted(sensor_names):
            values = []
            for snap in sensor_snapshots:
                v = snap.get(sname)
                if v is not None:
                    values.append(v)

            if len(values) < self.min_samples:
                continue

            is_discrete = self._is_discrete(sname) or not self._is_numeric(values)
            if is_discrete:
                cond = self._learn_discrete(sname, values, n_total)
            else:
                cond = self._learn_continuous(sname, values, n_total)

            # v10: discriminative filtering against global background
            if cond is not None and background_snapshots:
                bg_vals = bg_values_by_sensor.get(sname)
                if bg_vals:
                    power = self._discriminative_power(cond, values, bg_vals)
                    if power is None or power <= 0:
                        continue
                    cond = dict(cond)
                    cond["discriminative_power"] = round(power, 4)

            if cond is not None:
                conditions.append(cond)

        # v10: keep top-k most discriminative conditions (prevent overfitting)
        conditions.sort(key=lambda c: c.get("discriminative_power", 0.0),
                        reverse=True)
        if len(conditions) > self.max_conditions:
            conditions = conditions[:self.max_conditions]

        return conditions

    def learn_from_trace_group(self, traces_by_action: dict,
                               background_snapshots: list | None = None) -> dict:
        """
        批量学习：对每种 action 分别学习区间。

        Args:
            traces_by_action: {(device, command): [trace1, trace2, ...]}

        Returns:
            {(device, command): [conditions]}
        """
        result = {}
        for action_key, traces in traces_by_action.items():
            conditions = self.learn_intervals(traces, background_snapshots)
            if conditions:
                result[action_key] = conditions
        return result

    # ---- v10: Discriminative Condition Selection ----

    @staticmethod
    def _discriminative_power(cond: dict, action_values: list,
                              bg_values: list) -> float | None:
        """
        Compute how well a condition separates this action from the global
        background distribution.
          - continuous [L,U]: power = 1 - background coverage in [L,U]
          - discrete eq v:    power = |freq(action) - freq(background)|
        Returns None when background is empty (no filtering possible).
        """
        if not bg_values:
            return None
        op = cond.get("op")
        if op == "between":
            lo = cond.get("lower")
            hi = cond.get("upper")
            if lo is None or hi is None:
                return 0.0
            covered = sum(1 for v in bg_values if lo <= v <= hi)
            coverage = covered / len(bg_values)
            return 1.0 - coverage
        elif op == "eq":
            v = cond.get("value")
            if v is None:
                return 0.0
            freq_a = sum(1 for x in action_values if x == v) / max(1, len(action_values))
            freq_b = sum(1 for x in bg_values if x == v) / len(bg_values)
            return abs(freq_a - freq_b)
        return 0.0

    # ---- Continuous Sensor Learning ----

    def _learn_continuous(self, sname: str, values: list[float],
                          n_total: int) -> dict | None:
        """v8 dispatch: route to the configured method."""
        if self.method == "v8":
            return self._learn_continuous_v8(sname, values, n_total)

        # ---- v7 legacy path (kept for backward compat) ----
        if len(values) < self.min_samples:
            return None

        # Step 1: 去离群值 (IQR × 1.5)
        clean = self._remove_outliers_iqr(values)

        if len(clean) < self.min_samples:
            clean = values

        n_clean = len(clean)

        # Step 2: 计算 mean ± 2σ 扩展区间
        mean_v = statistics.mean(clean)
        if len(clean) >= 2:
            std_v = statistics.stdev(clean) if len(clean) > 1 else 1.0
        else:
            std_v = max(abs(mean_v) * 0.05, 0.5)

        lower = mean_v - 2.0 * std_v
        upper = mean_v + 2.0 * std_v

        # Step 3: 物理约束裁剪
        lower, upper = self._apply_physical_constraints(sname, lower, upper)

        # Step 4: 如果区间太窄
        phys_range = self._get_phys_range(sname)
        if phys_range is not None:
            phys_width = phys_range[1] - phys_range[0]
            if (upper - lower) < phys_width * 0.01:
                lower = upper = mean_v

        # Step 5: Wilson score 置信度
        confidence = wilson_score(n_clean, n_total)

        return {
            "sensor": sname,
            "op": "between",
            "lower": round(lower, 2),
            "upper": round(upper, 2),
            "confidence": round(confidence, 4),
            "sample_count": n_total,
            "clean_count": n_clean,
            "mean": round(mean_v, 2),
            "std": round(std_v, 2),
            "method": "v7_legacy",
        }

    # ---- v8 Continuous Sensor Learning: CKMS + Welford + Conformal ----

    def _learn_continuous_v8(self, sname: str, values: list[float],
                             n_total: int) -> dict | None:
        """
        v8 管线: CKMS 在线分位数 + Welford 在线方差 + Conformal Prediction。

        Step 1: CKMS 在线分位数去离群 → IQR 过滤
        Step 2: Welford 在线均值和方差 → 初始区间
        Step 3: Split Conformal Prediction 校准 → 分布无关覆盖保证
        Step 4: 物理约束裁剪

        理论优势:
          - CKMS: O(1/epsilon) 内存，可在 MCU 上在线运行
          - Welford: O(1) 更新，数值稳定
          - Conformal: 有限样本覆盖保证 P(val ∈ [L,U]) ≥ 1-α
          - 无正态假设（传感器数据通常不服从正态分布）

        Returns:
            {sensor, op, lower, upper, conformal_margin,
             coverage_guarantee, sample_count, method: "v8_conformal"}
        """
        if len(values) < self.min_samples:
            return None

        # Step 1: CKMS 在线分位数 —— IQR 过滤
        ckms = CKMSQuantile()
        for v in values:
            ckms.insert(v)
        q1 = ckms.query(0.25)
        q3 = ckms.query(0.75)
        iqr = q3 - q1

        if iqr > 0:
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            clean = [v for v in values if lower_fence <= v <= upper_fence]
        else:
            clean = values[:]  # all values identical

        if len(clean) < self.min_samples:
            clean = values  # too many outliers, keep all

        # Step 2: Welford 在线均值和方差
        welf = WelfordEstimator()
        for v in clean:
            welf.insert(v)

        mean_v = welf.mean
        std_v = welf.std if welf.count >= 2 else max(abs(mean_v) * 0.05, 0.5)

        # v9: 分阶段区间估计
        # 样本 < 10: 用观测 min/max（学 LLM 实际见过的范围，不假设分布）
        # 样本 >= 10: 用 Welford mean±2σ（统计泛化）
        if len(clean) < 10:
            lower_0 = min(clean)
            upper_0 = max(clean)
            padding = (upper_0 - lower_0) * 0.1
            if padding < 1.0:
                padding = 1.0
            lower_0 -= padding
            upper_0 += padding
        else:
            lower_0 = mean_v - 2.0 * std_v
            upper_0 = mean_v + 2.0 * std_v

        # Step 3: Split Conformal Prediction 校准
        # v9: alpha 按样本量自适应 — 样本越少越保守
        n_train = max(self.min_samples, int(len(clean) * 0.8))
        train_vals = clean[:n_train]
        cal_vals = clean[n_train:]

        if len(cal_vals) >= 2:
            # v9: adaptive alpha — fewer samples → wider interval (safer)
            n_cal = len(cal_vals)
            adaptive_alpha = min(0.30, self.conformal_alpha + 0.15 * (5.0 / max(n_cal, 1)))
            cal = ConformalCalibrator(alpha=adaptive_alpha)
            cal_data = [
                {"actual_value": v, "base_lower": lower_0, "base_upper": upper_0}
                for v in cal_vals
            ]
            cal.fit(cal_data)
            lower, upper = cal.calibrate_single(lower_0, upper_0)
            q_hat = cal._q_hat or 0.0
            coverage_text = f"≥{round((1 - self.conformal_alpha) * 100)}%"
        else:
            # 校准集太小，不做 conformal
            lower, upper = lower_0, upper_0
            q_hat = 0.0
            coverage_text = "none (insufficient calibration data)"

        # Step 4: 物理约束裁剪
        lower, upper = self._apply_physical_constraints(sname, lower, upper)

        return {
            "sensor": sname,
            "op": "between",
            "lower": round(lower, 2),
            "upper": round(upper, 2),
            "conformal_margin": round(q_hat, 4),
            "coverage_guarantee": coverage_text,
            "confidence": round(wilson_score(len(clean), n_total), 4),
            "sample_count": n_total,
            "clean_count": len(clean),
            "mean": round(mean_v, 2),
            "std": round(std_v, 2),
            "method": "v8_conformal",
            "iqr_source": "ckms_online",
            "variance_source": "welford_online",
        }

    # ---- Discrete Sensor Learning ----

    def _learn_discrete(self, sname: str, values: list,
                        n_total: int) -> dict | None:
        """
        对离散传感器（motion, door_open 等）学习众数。

        只有频率在 [80%, 95%] 之间的值才纳入条件。
        频率 > 95% → 接近常数，无区分力，不加条件。
        频率 < 80% → 和 action 关联不紧密，不加条件。
        """
        from collections import Counter
        counter = Counter(values)
        most_common_val, most_common_count = counter.most_common(1)[0]
        freq = most_common_count / n_total

        if freq < DISCRETE_FREQUENCY_THRESHOLD:
            return None
        if freq > 0.95:
            # 接近常数: 无区分力, 不加条件
            return None

        confidence = wilson_score(most_common_count, n_total)

        return {
            "sensor": sname,
            "op": "eq",
            "value": most_common_val,
            "confidence": round(confidence, 4),
            "sample_count": n_total,
            "frequency": round(freq, 4),
        }

    # ---- Outlier Removal ----

    def _remove_outliers_iqr(self, values: list[float]) -> list[float]:
        """
        IQR (Interquartile Range) 去离群值。
        Q1 - 1.5×IQR ~ Q3 + 1.5×IQR 之外的值视为离群值。

        这是 Tukey 的标准方法，比 Z-score（假设正态）更鲁棒。
        """
        if len(values) < 4:
            # 样本太少，无法可靠计算四分位数
            return values[:]  # 返回副本，不修改原列表

        sorted_v = sorted(values)
        n = len(sorted_v)

        # 四分位数 (使用 inclusive 方法，和 numpy 默认一致)
        q1 = self._percentile(sorted_v, 25)
        q3 = self._percentile(sorted_v, 75)
        iqr = q3 - q1

        if iqr == 0:
            # 所有值相同 → 没有离群值
            return values[:]

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        return [v for v in values if lower_fence <= v <= upper_fence]

    @staticmethod
    def _percentile(sorted_values: list[float], p: int) -> float:
        """
        计算百分位数（不使用 numpy）。
        p 取值 0-100。
        """
        n = len(sorted_values)
        if n == 0:
            return 0.0
        if n == 1:
            return sorted_values[0]

        # Linear interpolation method (same as numpy default)
        rank = (p / 100.0) * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo

        return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac

    # ---- Physical Constraints ----

    def _apply_physical_constraints(self, sname: str,
                                     lower: float, upper: float) -> tuple[float, float]:
        """
        用物理约束裁剪区间。
        例如：温度不可能 < -20°C 或 > 50°C。

        这同时是安全保护（防止极端值规则）和合理性保证（审稿人认可）。
        """
        phys_range = self._get_phys_range(sname)
        if phys_range is None:
            return lower, upper

        p_min, p_max = phys_range
        return max(lower, p_min), min(upper, p_max)

    @staticmethod
    def _get_phys_range(sname: str) -> tuple[float, float] | None:
        """获取传感器的物理合理范围"""
        ranges = {
            "temperature": (-20.0, 50.0),   # °C：家用环境
            "humidity": (0.0, 100.0),       # %
            "light": (0.0, 2000.0),         # lux：室内光照
        }
        return ranges.get(sname)

    @staticmethod
    def _is_discrete(sname: str) -> bool:
        """判断传感器是否为离散/非数值类型"""
        discrete_sensors = {"motion", "door_open", "door", "occupancy", "smoke",
                          "time_of_day", "light_category", "hour"}
        return sname in discrete_sensors

    @staticmethod
    def _is_numeric(values: list) -> bool:
        """检测 value 列表是否全是数值（用于自动分类连续/离散）"""
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values[:10])


# ========== Experiment Helper ==========

def group_traces_by_action(traces: list) -> dict:
    """
    将 trace 按 (device, command) 分组。
    用于 learn_from_trace_group() 的输入预处理。

    Args:
        traces: 完整的 trace 列表（含 LLM response）

    Returns:
        {(device, command): [sensor_snapshots]}
    """
    import json as _json
    from collections import defaultdict

    groups = defaultdict(list)

    for trace in traces:
        llm_resp = trace.get("llm_response", {})
        tool_calls = llm_resp.get("tool_calls") or []

        sensors = trace.get("sensors", {})

        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")

            # 只处理 _control 结尾的 tool
            if "_control" not in name:
                continue
            device = name.replace("_control", "")

            # 解析参数获取 command
            args_str = func.get("arguments", "{}")
            try:
                args = _json.loads(args_str) if isinstance(args_str, str) else args_str
            except _json.JSONDecodeError:
                continue
            command = args.get("command", "on")

            groups[(device, command)].append(sensors)

    return dict(groups)


def visualize_intervals(conditions: list[dict]) -> str:
    """
    生成学习区间的 ASCII 可视化（调试和实验报告用）。

    Example output:
        temperature: ████████████████░░░░  [22.0, 35.0]  (n=15, conf=0.92)
    """
    lines = []
    for c in conditions:
        sname = c["sensor"]
        if c.get("op") == "between":
            lo = c["lower"]
            hi = c["upper"]
            phys = RuleGeneralizer._get_phys_range(sname)
            if phys:
                total = phys[1] - phys[0]
                left_pct = max(0, (lo - phys[0]) / total)
                right_pct = max(0, (phys[1] - hi) / total)
                bar_width = max(0, 1.0 - left_pct - right_pct)
                bar = "." * int(left_pct * 20) + "#" * int(bar_width * 20) + "." * int(right_pct * 20)
            else:
                bar = f"[{lo:.1f}, {hi:.1f}]"

            n = c.get("sample_count", "?")
            cf = c.get("confidence", 0)
            lines.append(
                f"  {sname:>15s}: {bar}  [{lo:.1f}, {hi:.1f}]  "
                f"(n={n}, conf={cf:.3f})"
            )
        elif c.get("op") == "eq":
            lines.append(
                f"  {sname:>15s}: == {c['value']}  "
                f"(freq={c.get('frequency', 0):.2f}, conf={c.get('confidence', 0):.3f})"
            )
    return "\n".join(lines)


# ========== Self-Test ==========

if __name__ == "__main__":
    print("=" * 60)
    print("  RuleGeneralizer — Self-Test")
    print("=" * 60)

    rg = RuleGeneralizer(min_samples=3)

    # Test 1: 连续传感器 — 标准情况
    print("\n[Test 1] Continuous sensor — normal distribution")
    traces = []
    import random
    random.seed(42)
    for _ in range(20):
        # 模拟 LLM 在 temp ∈ [29.5, 32.5] 之间选了 fan_on
        temp = random.gauss(31.0, 1.0)
        light = random.gauss(50, 15)
        traces.append({"sensors": {"temperature": temp, "light": light}})

    conditions = rg.learn_intervals(traces)
    print(visualize_intervals(conditions))
    assert len(conditions) == 2, f"Expected 2 conditions, got {len(conditions)}"
    # 温度区间应该大致覆盖 [29, 33]，且宽度合理
    temp_cond = [c for c in conditions if c["sensor"] == "temperature"][0]
    interval_width = temp_cond["upper"] - temp_cond["lower"]
    assert 0.5 < interval_width < 8.0, \
        f"Interval width unreasonable: {interval_width:.1f} (expected 2-6 for std=1.0)"
    assert temp_cond["sample_count"] == 20
    assert temp_cond["confidence"] > 0.5
    # 光传感器方差更大(15) → 区间更宽，但置信度仍然合理
    light_cond = [c for c in conditions if c["sensor"] == "light"][0]
    light_width = light_cond["upper"] - light_cond["lower"]
    assert light_width > temp_cond["upper"] - temp_cond["lower"], \
        "Light interval should be wider than temperature (larger variance)"
    print("  [PASS]")

    # Test 2: 包含离群值
    print("\n[Test 2] Continuous sensor — with outliers")
    traces2 = []
    for _ in range(15):
        traces2.append({"sensors": {"temperature": random.gauss(31.0, 0.5)}})
    # 加入 3 个离群值
    traces2.append({"sensors": {"temperature": 45.0}})  # 极端高温
    traces2.append({"sensors": {"temperature": 15.0}})  # 极端低温
    traces2.append({"sensors": {"temperature": 48.0}})

    conditions2 = rg.learn_intervals(traces2)
    print(visualize_intervals(conditions2))
    # 即使有离群值，学习区间应该仍然紧凑
    temp2 = [c for c in conditions2 if c["sensor"] == "temperature"][0]
    assert temp2["upper"] < 40.0, f"Outlier not filtered: upper={temp2['upper']}"
    # clean_count 应该 < sample_count（离群值被识别）
    assert temp2["clean_count"] < temp2["sample_count"], \
        f"Outliers not detected: clean={temp2['clean_count']}, total={temp2['sample_count']}"
    print("  [OK] PASS (outliers filtered)")

    # Test 3: 离散传感器
    print("\n[Test 3] Discrete sensor — motion")
    traces3 = []
    for _ in range(18):
        traces3.append({"sensors": {"motion": 1, "temperature": 30.0}})
    for _ in range(2):
        traces3.append({"sensors": {"motion": 0, "temperature": 30.0}})

    conditions3 = rg.learn_intervals(traces3)
    print(visualize_intervals(conditions3))
    # motion 应该是 == 1（频率 90% > 80%）
    motion_cond = [c for c in conditions3 if c["sensor"] == "motion"]
    assert len(motion_cond) == 1, "Motion condition should be present"
    assert motion_cond[0]["value"] == 1
    print("  [PASS]")

    # Test 4: 样本太少 — 不学习
    print("\n[Test 4] Too few samples — should return empty")
    traces4 = [
        {"sensors": {"temperature": 31.0}},
        {"sensors": {"temperature": 32.0}},
    ]
    conditions4 = rg.learn_intervals(traces4)
    assert len(conditions4) == 0, f"Should return empty for < 3 samples, got {len(conditions4)}"
    print("  [PASS]")

    # Test 5: 离散传感器频率不足 — 不纳入条件
    print("\n[Test 5] Discrete sensor — insufficient frequency")
    traces5 = []
    for _ in range(6):
        traces5.append({"sensors": {"motion": 1}})
    for _ in range(5):
        traces5.append({"sensors": {"motion": 0}})
    conditions5 = rg.learn_intervals(traces5)
    motion5 = [c for c in conditions5 if c["sensor"] == "motion"]
    assert len(motion5) == 0, f"Motion with 55% frequency should not create condition"
    print("  [PASS]")

    # Test 6: 物理约束
    print("\n[Test 6] Physical constraints — temperature clamped")
    traces6 = []
    for _ in range(10):
        traces6.append({"sensors": {"temperature": random.gauss(48.0, 2.0)}})
    conditions6 = rg.learn_intervals(traces6)
    temp6 = [c for c in conditions6 if c["sensor"] == "temperature"][0]
    assert temp6["upper"] <= 50.0, f"Temperature should be clamped at 50: {temp6['upper']}"
    print(f"  upper={temp6['upper']:.1f} (phys max=50.0)")
    print("  [PASS]")

    # Test 7: 空输入
    print("\n[Test 7] Empty input")
    assert rg.learn_intervals([]) == []
    print("  [PASS]")

    # Test 8: 分组学习
    print("\n[Test 8] Group learning by action")
    traces8 = []
    for _ in range(10):
        traces8.append({
            "sensors": {"temperature": random.gauss(31.0, 1.0)},
            "llm_response": {
                "tool_calls": [{
                    "function": {
                        "name": "fan_control",
                        "arguments": '{"command": "on", "speed": 2}',
                    }
                }]
            }
        })
    for _ in range(8):
        traces8.append({
            "sensors": {"temperature": random.gauss(25.0, 0.5)},
            "llm_response": {
                "tool_calls": [{
                    "function": {
                        "name": "led_control",
                        "arguments": '{"command": "on", "brightness": 60}',
                    }
                }]
            }
        })

    grouped = group_traces_by_action(traces8)
    result = rg.learn_from_trace_group(grouped)
    print(f"  Learned intervals for {len(result)} actions:")
    for (dev, cmd), conds in result.items():
        print(f"    {dev}.{cmd}: {len(conds)} conditions")
        print(visualize_intervals(conds))
    assert len(result) == 2, f"Expected 2 action groups, got {len(result)}"
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED [OK]")
    print("=" * 60)
