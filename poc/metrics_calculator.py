"""
DistillToMCU Phase 0b — 指标自动计算器
==========================================
从实验输出 (traces.jsonl + metrics.jsonl + rules_snapshot.json) 自动计算全部 18 个指标。

⚠️ 数据来源诚实性（v6 修复）：
  - Phase 0 所有实验均为 PC 仿真，本地延迟 = 仿真随机值，不是硬件实测。
  - PMS 指标来自独立 bandit 仿真，CPU 周期数是估算，未在 MCU 上测量。
  - 内存/功耗均为设计估算，Phase 1 硬件实测后才能作为论文数据。
  - 每条指标输出带 data_provenance 字段，禁止冒充硬件测量。
"""

import json
import math
import statistics
import random
from collections import defaultdict


class MetricsCalculator:
    """从实验数据中计算全部 18 个指标"""

    def __init__(self, traces: list, daily_metrics: list, rules: list,
                 seed: int = 42):
        """
        Args:
            traces: JSONL trace 列表（完整交互记录）
            daily_metrics: 每日汇总指标
            rules: 规则快照
        """
        self.traces = traces
        self.daily = daily_metrics
        self.rules = rules
        random.seed(seed)

    def compute_all(self) -> dict:
        """计算全部 18 个指标，返回结构化报告"""
        return {
            "data_provenance": {
                "note": "Phase 0 PC simulation; hardware measurement pending Phase 1",
                "local_latency_source": "simulated random 3-15ms (not hardware)",
                "bandit_source": "synthetic bandit simulation (bandit_selector.py)",
                "power_memory_source": "design estimates, not measured",
            },
            "效能": self._compute_efficiency(),
            "规则质量": self._compute_rule_quality(),
            "ML性能": self._compute_ml_performance(),
            "系统-存储": self._compute_storage(),
            "系统-内存": self._compute_memory(),
            "系统-功耗": self._compute_power(),
        }

    # ---- 效能 (5) ----

    def _compute_efficiency(self) -> dict:
        local_traces = [t for t in self.traces
                        if t.get("execution", {}).get("mode") == "local"]
        cloud_traces = [t for t in self.traces
                        if t.get("execution", {}).get("mode") == "cloud"]
        total = len(self.traces)
        n_local = len(local_traces)

        # Latency
        local_lats = sorted([
            (t.get("llm_response", {}) or {}).get("latency_ms", 0)
            for t in local_traces
        ])
        cloud_lats = sorted([
            (t.get("llm_response", {}) or {}).get("latency_ms", 0)
            for t in cloud_traces
        ])

        def percentile(data, p):
            if not data:
                return 0
            k = (len(data) - 1) * p / 100
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            return data[int(f)] * (c - k) + data[int(c)] * (k - f)

        # Offline Availability = 本地可执行的规则数 / 总规则数
        actionable_rules = [r for r in self.rules
                            if r.get("state") in ("active", "verified")]
        offline_avail = len(actionable_rules) / max(1, len(self.rules)) * 100

        return {
            "autonomy_rate": round(n_local / max(1, total) * 100, 1),
            "cloud_call_reduction": round((1 - len(cloud_traces) / max(1, total)) * 100, 1),
            "latency_local_p50_ms": round(percentile(local_lats, 50), 1),
            "latency_local_p95_ms": round(percentile(local_lats, 95), 1),
            "latency_local_p99_ms": round(percentile(local_lats, 99), 1),
            "latency_cloud_p50_ms": round(percentile(cloud_lats, 50), 1),
            "latency_cloud_p95_ms": round(percentile(cloud_lats, 95), 1),
            "offline_availability_pct": round(offline_avail, 1),
        }

    # ---- 规则质量 (4) ----

    def _compute_rule_quality(self) -> dict:
        # Precision = 本地执行中用户接受的比例
        # ⚠️ Phase 0 所有 feedback 均为自动 "accepted"（仿真无真实用户），
        #    因此 precision/override/false_auto 当前是占位值，不能作为论文主表数据。
        #    Phase 1 使用板载 accept/correct 按钮后才有效。
        local_traces = [t for t in self.traces
                        if t.get("execution", {}).get("mode") == "local"]
        accepted = [t for t in local_traces
                    if (t.get("feedback", {}) or {}).get("type") == "accepted"]
        precision = len(accepted) / max(1, len(local_traces))

        # Recall = 可本地化的规则覆盖的任务比例（近似：有 tool_call 的 cloud trace 比例）
        cloud_with_action = [t for t in self.traces
                             if t.get("execution", {}).get("mode") == "cloud"
                             and (t.get("llm_response", {}) or {}).get("tool_calls")]
        total_with_action = len(local_traces) + len(cloud_with_action)
        recall = len(local_traces) / max(1, total_with_action)

        # Override Rate = corrected / (accepted + corrected)
        corrected = [t for t in local_traces
                     if (t.get("feedback", {}) or {}).get("type") == "corrected"]
        override = len(corrected) / max(1, len(local_traces))

        # False Automation Rate = 错误自动执行率（模拟：local 但无 tool_call）
        false_auto = len([t for t in local_traces
                          if not (t.get("llm_response", {}) or {}).get("tool_calls")])
        false_auto_rate = false_auto / max(1, len(local_traces))

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "override_rate": round(override, 4),
            "false_automation_rate": round(false_auto_rate, 4),
            "feedback_provenance": "simulated: all feedback auto-accepted, NOT user feedback",
            "reportable_in_paper": False,
        }

    # ---- ML 性能 (2) ----

    def _compute_ml_performance(self) -> dict:
        """从 bandit_selector.py 实时计算 PMS 指标（v7：不硬编码）。

        ⚠️ 这是合成 bandit 仿真，不是 MCU 实测；CPU 周期数为设计估算。
        """
        from bandit_selector import (
            PMSSelector, ExactTSSelector, GreedySelector,
            BanditEnvironment, run_bandit_comparison,
        )
        rng = __import__('random').Random(42)

        n_arms = 8
        n_rounds = 3000
        true_probs = [rng.uniform(0.3, 0.95) for _ in range(n_arms)]
        env = BanditEnvironment(true_probs, seed=42)

        selectors = {
            "PMS": PMSSelector(seed=42),
            "ExactTS": ExactTSSelector(seed=42),
            "Greedy": GreedySelector(),
        }
        comp = run_bandit_comparison(
            env, selectors, n_rounds=n_rounds, matching_size=3, seed=42)

        pms_regret = comp["PMS"]["final_regret"]
        ex_regret = comp["ExactTS"]["final_regret"]
        pms_rate = comp["PMS"]["final_optimal_rate"]
        ex_rate = comp["ExactTS"]["final_optimal_rate"]
        greedy_rate = comp["Greedy"]["final_optimal_rate"]

        return {
            "pms_regret": round(pms_regret, 1),
            "exact_ts_regret": round(ex_regret, 1),
            "regret_gap": round(pms_regret - ex_regret, 2),
            "regret_gap_pct": round((pms_regret - ex_regret) / max(0.01, ex_regret) * 100, 2),
            "pms_optimal_arm_rate_pct": round(pms_rate, 1),
            "exact_ts_optimal_arm_rate_pct": round(ex_rate, 1),
            "greedy_optimal_arm_rate_pct": round(greedy_rate, 1),
            "pms_memory_per_rule": "2 B (uint8)",
            "exact_ts_memory_per_rule": "8 B (float32)",
            "memory_savings_pct": 75.0,
            "mcu_feasible": True,
            "data_source": "synthetic bandit simulation, NOT hardware",
            "cpu_per_select_provenance": "design estimate (<10 cycles), NOT measured on MCU",
            "reportable_in_paper": "as ablation 4 (simulation) only; MCU claim pending C implementation + measurement",
        }

    # ---- 系统-存储 (3) ----

    def _compute_storage(self) -> dict:
        # Rule Store Size: 估算每条规则的 JSON 大小
        rule_sizes = [len(json.dumps(r, ensure_ascii=False)) for r in self.rules]
        total_size = sum(rule_sizes)

        # Flash Erase Cycles 估算
        # 假设 SPIFFS 4KB sector，每次写触发 erase
        trace_count = len(self.traces)
        writes_per_day = trace_count / max(1, len(self.daily))
        # RingBuffer: 每 15 条 trace flush 一次 → 减少 15x
        naive_erases = trace_count  # 每条 trace 一次
        ring_erases = trace_count / 15  # 每 15 条一次
        total_days = len(self.daily)
        flash_erase_per_day_naive = writes_per_day
        flash_erase_per_day_ring = writes_per_day / 15

        # ⚠️ v6 修复：匹配延迟不能冒充硬件实测。
        # Phase 0 无硬件数据，这里取仿真 trace 的 local latency 均值并明确标注来源。
        local_lats = [
            (t.get("llm_response", {}) or {}).get("latency_ms", 0)
            for t in self.traces if t.get("execution", {}).get("mode") == "local"
        ]
        sim_match_latency = round(
            statistics.mean(local_lats), 1) if local_lats else 0.0

        return {
            "rule_store_size_bytes": total_size,
            "avg_rule_size_bytes": round(statistics.mean(rule_sizes), 1) if rule_sizes else 0,
            "flash_erase_cycles_naive_est": int(naive_erases),
            "flash_erase_cycles_ringbuffer_est": int(ring_erases),
            "flash_lifetime_days_naive": int(100000 / max(1, flash_erase_per_day_naive)),
            "flash_lifetime_days_ringbuffer": int(100000 / max(1, flash_erase_per_day_ring)),
            "match_latency_ms": sim_match_latency,
            "match_latency_provenance": "simulated local latency (Phase 0), hardware measurement pending",
        }

    # ---- 系统-内存 (3) ----

    def _compute_memory(self) -> dict:
        n_rules = len(self.rules)
        # 设计估算（Phase 1 实测后替换）
        psram_peak = 32 * 1024 + 16 * 1024 + n_rules * 512  # 粗略估计

        # Internal SRAM
        internal_sram = 24 * 1024 + 8 * 1024 + 4 * 1024  # task stacks

        # Task Stack Watermark (FreeRTOS)
        task_stacks = {
            "agent_task": 24 * 1024,
            "outbound_task": 8 * 1024,
            "cli_task": 4 * 1024,
            "confirm_task": 2 * 1024,
        }

        return {
            "psram_peak_bytes": psram_peak,
            "internal_sram_bytes": internal_sram,
            "task_stack_agent_bytes": task_stacks["agent_task"],
            "task_stack_total_bytes": sum(task_stacks.values()),
            "rule_storage_memory_bytes": n_rules * 2,  # PMS uint8 × 2/rule
            "rule_storage_vs_float_bytes": n_rules * 8,  # float32 × 2/rule
            "memory_savings_pct": round((1 - 2/8) * 100, 1),  # 75% reduction
            "provenance": "design estimates based on config constants, NOT measured",
        }

    # ---- 系统-功耗 (2) ----

    def _compute_power(self) -> dict:
        # ESP32-S3 典型功耗（估算，Phase 1 用 USB 功率计实测后才能入论文）
        # Active: WiFi + CPU 240MHz ≈ 300mW
        # Idle: Light sleep ≈ 5mW
        power_active_mw = 300
        power_idle_mw = 5

        # Energy per interaction
        local_latency_s = 0.009  # 9ms
        cloud_latency_s = 3.8    # 3.8s

        local_energy_j = power_active_mw / 1000 * local_latency_s
        cloud_energy_j = power_active_mw / 1000 * cloud_latency_s

        n_local = len([t for t in self.traces
                       if t.get("execution", {}).get("mode") == "local"])
        n_cloud = len([t for t in self.traces
                       if t.get("execution", {}).get("mode") == "cloud"])

        total_energy_j = n_local * local_energy_j + n_cloud * cloud_energy_j

        return {
            "power_active_mw": power_active_mw,
            "power_idle_mw": power_idle_mw,
            "energy_per_local_interaction_mj": round(local_energy_j * 1000, 2),
            "energy_per_cloud_interaction_mj": round(cloud_energy_j * 1000, 2),
            "total_energy_consumed_j": round(total_energy_j, 2),
            "energy_savings_vs_pure_cloud_pct": round(
                (1 - total_energy_j / ((n_local + n_cloud) * cloud_energy_j)) * 100, 1
            ),
            "provenance": "typical-value estimates, NOT measured",
        }


# ============================================================
# 统计检验
# ============================================================

def bootstrap_confidence_interval(data: list, n_bootstrap: int = 10000,
                                  alpha: float = 0.05, seed: int = 42) -> tuple:
    """
    Bootstrap 95% 置信区间。
    用于 Autonomy Rate 和 Cloud Call Reduction。
    """
    rng = random.Random(seed)
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = [rng.choice(data) for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(n_bootstrap * alpha / 2)]
    hi = means[int(n_bootstrap * (1 - alpha / 2))]
    return lo, hi


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    # Load real data
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    traces_path = os.path.join(output_dir, "traces.jsonl")
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    rules_path = os.path.join(output_dir, "rules_snapshot.json")

    if all(os.path.exists(p) for p in [traces_path, metrics_path, rules_path]):
        print("Loading real 30-day experiment data...")
        with open(traces_path, "r", encoding="utf-8") as f:
            traces = [json.loads(l) for l in f if l.strip()]
        with open(metrics_path, "r", encoding="utf-8") as f:
            daily = [json.loads(l) for l in f if l.strip()]
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = json.load(f)

        calc = MetricsCalculator(traces, daily, rules)
        report = calc.compute_all()

        print("\n" + "=" * 60)
        print("  DistillToMCU — Complete Metrics Report (30 days)")
        print("=" * 60)

        for category, metrics in report.items():
            print(f"\n  [{category}]")
            for key, val in metrics.items():
                if isinstance(val, float):
                    print(f"    {key:<35s} {val:>12.4f}")
                else:
                    print(f"    {key:<35s} {str(val):>12s}")

        # Bootstrap CI for AR
        local_rate = [
            1 if t.get("execution", {}).get("mode") == "local" else 0
            for t in traces
        ]
        lo, hi = bootstrap_confidence_interval(local_rate, seed=42)
        print(f"\n  [Statistical Tests]")
        print(f"    Bootstrap 95% CI for AR:    [{lo:.4f}, {hi:.4f}]")
        print(f"    Bootstrap 95% CI for AR%:   [{lo*100:.1f}%, {hi*100:.1f}%]")

    else:
        print("No data files found. Using demo data...")
        # Demo
        traces = [
            {"execution": {"mode": "local"}, "llm_response": {"latency_ms": 9}},
            {"execution": {"mode": "cloud"}, "llm_response": {"latency_ms": 3800, "tool_calls": [{}]}},
            {"execution": {"mode": "local"}, "llm_response": {"latency_ms": 8}},
        ] * 100
        daily = [{"local_calls": 6, "cloud_calls": 4, "new_rules_today": 0}] * 30
        rules = [{"id": f"r_{i}", "state": "active", "conditions": []} for i in range(10)]

        calc = MetricsCalculator(traces, daily, rules)
        report = calc.compute_all()
        for cat, m in report.items():
            print(f"\n[{cat}]")
            for k, v in m.items():
                print(f"  {k}: {v}")
