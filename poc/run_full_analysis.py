"""
DistillToMCU — 全量正式分析运行器（论文主表数据源）
====================================================
补 Phase 0b 缺失的正式基线对比：
  1. 8 个基线 × 6 组实验（合成×3 + STRANDS + UCI V2 + UCI V3）
  2. Friedman + Nemenyi 统计检验（statistics_tests.py）
  3. Bootstrap 95% CI（Autonomy Rate / Cloud Call Reduction）

纯回放已有 traces，不调用任何 LLM API，零成本。
输出：
  poc/output/baseline_results.json
  poc/output/statistics_results.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from baselines import (
    PureCloudBaseline, ExactCacheBaseline, SemanticCacheBaseline,
    SensorVectorCacheBaseline,
    UserDefinedRulesBaseline, LLMOneShotBaseline, DecisionTreeBaseline,
    OnlineDailyRefitDecisionTreeBaseline,
    ESPHomeStateMachineBaseline, ESPClawStyleBaseline,
    run_baseline_comparison, save_baseline_results,
    USER_RULES_BY_LABEL,
)
from statistics_tests import friedman_test, nemenyi_posthoc, bootstrap_ci


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# v7: UCI V2 已归档至 legacy_20260802，不再纳入正式分析
SEEDS = [42, 123, 999, 777]


def build_experiments():
    """v10.7: 36 run4b dirs (9 datasets x 4 seeds)."""
    exps = []
    for key in ["seed42", "seed123", "seed999", "seed777"]:
        for s in SEEDS:
            exps.append((f"run4b_{key}_seed{s}", f"synthetic_{key}_s{s}"))
    for s in SEEDS:
        exps.append((f"run4b_strands_seed{s}", f"strands_s{s}"))
    for s in SEEDS:
        exps.append((f"run4b_uci_seed{s}", f"uci_v3_s{s}"))
    for key in ["sml2010", "steel", "airquality"]:
        for s in SEEDS:
            exps.append((f"run4b_{key}_seed{s}", f"{key}_s{s}"))
    return exps


EXPERIMENTS = build_experiments()

# B4 one-shot rules cached per dataset key (generated once, shared by seeds)
ONESHOT_RULE_CACHE = {}


def dataset_key_of(exp_dir):
    """'run4b_uci_seed42' -> 'uci'; 'run4b_seed123_seed42' -> 'seed123'."""
    base = exp_dir.replace("run4b_", "")
    for k in ["seed42", "seed123", "seed999", "seed777", "strands", "uci",
              "sml2010", "steel", "airquality"]:
        if base.startswith(k):
            return k
    return base


def load_traces(exp_dir):
    path = os.path.join(OUTPUT_DIR, exp_dir, "traces.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_day_bounds(exp_dir):
    """metrics.jsonl 每日累计 total → 用于把 trace 索引映射回 day。"""
    path = os.path.join(OUTPUT_DIR, exp_dir, "metrics.jsonl")
    bounds = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m = json.loads(line)
                    t = m.get("total")
                    if isinstance(t, (int, float)) and t > 0:
                        bounds.append(int(t))
    except FileNotFoundError:
        return None
    return bounds


def make_baselines(seed=42, dataset_key=None):
    """论文主表基线：6个MCU可运行或可对标的方法

    v10: B4 LLM One-shot 优先使用真实 LLM 生成规则（需要 DEEPSEEK_API_KEY），
    无 key 时退回手写规则并在 display_name 中诚实标注（handcrafted fallback）。
    """
    import os as _os
    api_key = _os.environ.get("DEEPSEEK_API_KEY", "")
    if api_key and not api_key.startswith("sk-placeholder"):
        from llm_client import call_llm_with_backend as _call

        class _LLMBackend:
            @staticmethod
            def call_llm_with_backend(messages, backend="deepseek-v4-flash",
                                      temperature=0.0, max_tokens=None):
                return _call(messages, backend=backend, temperature=temperature,
                             max_tokens=max_tokens)

            @staticmethod
            def get_available_llms():
                import os as _os
                out = []
                if len(_os.environ.get("DEEPSEEK_API_KEY", "")) > 10:
                    out.append("deepseek-v4-flash")
                return out

        llm_backend = _LLMBackend()
    else:
        llm_backend = None
    baselines = {
        "B1 Pure Cloud": PureCloudBaseline(seed=seed),
        "B2 Exact Cache": ExactCacheBaseline(seed=seed),
        "B3 User-defined Rules": UserDefinedRulesBaseline(
            seed=seed,
            rules=USER_RULES_BY_LABEL.get(dataset_key)),
        "B5 Decision Tree": DecisionTreeBaseline(seed=seed),
        "B5b Decision Tree (online refit)": OnlineDailyRefitDecisionTreeBaseline(seed=seed),
        "B6 ESP-Claw-style": ESPClawStyleBaseline(seed=seed),
    }
    # B4: generate once per dataset, share across its 4 seed runs
    b4 = LLMOneShotBaseline(llm_client=llm_backend, seed=seed)
    if dataset_key and dataset_key in ONESHOT_RULE_CACHE:
        b4.rules = list(ONESHOT_RULE_CACHE[dataset_key])
        b4.generated_by_llm = True
        b4._rules_generated = True
    else:
        b4._generate_rules()
        if dataset_key and b4.rules:
            ONESHOT_RULE_CACHE[dataset_key] = list(b4.rules)
    baselines["B4 LLM One-shot"] = b4
    return baselines


def make_semantic_sweep(seed=42):
    """扫 MCU 可行的传感器向量缓存多阈值——展示 AR/AGREE 随阈值的变化趋势。

    v10.6：MiniLM 已可离线加载 → 同时给出真实语义缓存（384 维）与
    MCU 可行的传感器向量缓存（4 维）两组曲线。
    """
    return {
        "Semantic Cache θ=0.90 (MiniLM)": SemanticCacheBaseline(threshold=0.90, seed=seed),
        "Semantic Cache θ=0.92 (MiniLM)": SemanticCacheBaseline(threshold=0.92, seed=seed),
        "Semantic Cache θ=0.95 (MiniLM)": SemanticCacheBaseline(threshold=0.95, seed=seed),
        "Semantic Cache θ=0.98 (MiniLM)": SemanticCacheBaseline(threshold=0.98, seed=seed),
        "Sensor-Vector Cache θ=0.90": SensorVectorCacheBaseline(threshold=0.90, seed=seed),
        "Sensor-Vector Cache θ=0.92": SensorVectorCacheBaseline(threshold=0.92, seed=seed),
        "Sensor-Vector Cache θ=0.95": SensorVectorCacheBaseline(threshold=0.95, seed=seed),
        "Sensor-Vector Cache θ=0.98": SensorVectorCacheBaseline(threshold=0.98, seed=seed),
        "B1 Pure Cloud": PureCloudBaseline(seed=seed),
    }


def compute_ours_decision_agreement(traces, exp_dir):
    """v7: 计算 Ours 的决策一致性（对标基线的 decision_agreement_pct）。

    对每条 cloud trace，用规则快照回放：检查规则引擎匹配结果是否与 LLM 决策一致。
    比较 (device, command) —— 规则匹配且 action 相同 = 一致。
    """
    import json as _json
    from rule_engine import RuleEngine

    rules_path = os.path.join(OUTPUT_DIR, exp_dir, "rules_snapshot.json")
    engine = RuleEngine()
    engine.load_snapshot(rules_path)

    cloud_traces = [t for t in traces if t.get("execution", {}).get("mode") == "cloud"]
    if not cloud_traces:
        return 0.0, 0

    agreed = 0
    total = 0
    for t in cloud_traces:
        tc = (t.get("llm_response", {}) or {}).get("tool_calls") or []
        llm_action = None
        if tc:
            func = tc[0].get("function", {})
            name = func.get("name", "").replace("_control", "")
            try:
                args = _json.loads(func.get("arguments", "{}"))
            except _json.JSONDecodeError:
                args = {}
            llm_action = (name, args.get("command", "on"))

        total += 1
        matches = engine.match(t.get("sensors", {}))
        best = engine.resolve_conflict(matches)
        if best:
            rule_dev = best.action.get("device", "")
            rule_cmd = best.action.get("command", "")
            if llm_action and (rule_dev, rule_cmd) == llm_action:
                agreed += 1
            # If LLM had no action but rule fired → disagreement (not counted)
        else:
            # No rule matched — agree only if LLM also took no action
            if llm_action is None:
                agreed += 1

    return round(agreed / max(1, total) * 100, 1), total


def compute_ours_eval_window(traces, exp_dir, day_bounds, train_ratio=0.7):
    """Ours 在 held-out 评估窗口（后 30% 天）上的 AR 与 AGREE。

    - AR：直接取 Ours 在线运行在评估窗口内的 local/total（每个决策只用过去规则，
      无泄漏）。
    - AGREE(warm)：只用前 70% 天 trace 离线重蒸馏出的规则回放评估窗口的云端
      决策——与批量 DT 完全相同的训练/评估纪律，零泄漏。
    - AGREE(final)：最终规则快照回放（补充：系统运行到第 30 天的完整能力）。
    """
    from baselines import trace_day_labels, extract_cloud_action
    from rule_engine import RuleEngine
    from distiller import Distiller

    day_labels = trace_day_labels(len(traces), day_bounds)
    n_days = max(day_labels)
    split_day = int(n_days * train_ratio) + 1
    eval_idx = [i for i, d in enumerate(day_labels) if d >= split_day]
    warm_idx = [i for i, d in enumerate(day_labels) if d < split_day]

    local = sum(1 for i in eval_idx
                if traces[i].get("execution", {}).get("mode") == "local")
    ar_eval = round(local / max(1, len(eval_idx)) * 100, 1)

    def agree_using_engine(engine):
        agreed = total = 0
        for i in eval_idx:
            t = traces[i]
            if t.get("execution", {}).get("mode") != "cloud":
                continue
            llm_act = extract_cloud_action(t)
            # 与基线口径一致：只统计"教师确实采取了动作"的云端决策
            if llm_act is None:
                continue
            matches = engine.match(t.get("sensors", {}))
            best = engine.resolve_conflict(matches)
            total += 1
            if best is not None and \
                    best.action.get("device") == llm_act["device"] and \
                    best.action.get("command") == llm_act["command"]:
                agreed += 1
        return round(agreed / max(1, total) * 100, 1), total

    # 最终规则快照（第 30 天系统状态）
    rules_path = os.path.join(OUTPUT_DIR, exp_dir, "rules_snapshot.json")
    eng_final = RuleEngine()
    try:
        eng_final.load_snapshot(rules_path)
    except Exception:
        eng_final = None
    agree_final, agree_final_n = agree_using_engine(eng_final) if eng_final \
        else (0.0, 0)

    # 只用前 70% 天离线重蒸馏（零泄漏下界）
    eng_warm = RuleEngine()
    di = Distiller(eng_warm, llm_client=None)
    try:
        di.distill([traces[i] for i in warm_idx])
    except Exception:
        pass
    agree_warm, agree_warm_n = agree_using_engine(eng_warm)

    return {
        "ar_eval_window": ar_eval,
        "eval_n": len(eval_idx),
        "eval_first_day": split_day,
        "agree_warm_rules": agree_warm,
        "agree_warm_n": agree_warm_n,
        "agree_final_rules": agree_final,
        "agree_final_n": agree_final_n,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}
    ar_matrix = []      # blocks(数据集) × treatments(基线)
    ccr_matrix = []
    dataset_names = []

    for exp_dir, label in EXPERIMENTS:
        try:
            traces = load_traces(exp_dir)
        except FileNotFoundError:
            print(f"[SKIP] {exp_dir} traces not found (legacy experiment not re-run).")
            continue
        day_bounds = load_day_bounds(exp_dir)
        baselines = make_baselines(seed=42,
                                   dataset_key=dataset_key_of(exp_dir))
        res = run_baseline_comparison(traces, baselines, seed=42,
                                      train_ratio=0.7, day_bounds=day_bounds)

        ar_row = [res[name]["autonomy_rate"] for name in baselines]
        ccr_row = [res[name]["cloud_call_reduction"] for name in baselines]
        ar_matrix.append(ar_row)
        ccr_matrix.append(ccr_row)
        dataset_names.append(label)

        # Bootstrap CI for AR（基于逐条 trace 的 local/cloud 二值）
        local_flags = [
            1 if t.get("execution", {}).get("mode") == "local" else 0
            for t in traces
        ]
        # AR: two definitions
        # 1) Full AR = local / all (conservative, counts LLM-idle traces too)
        # 2) Actionable AR = local / (local + cloud_with_toolcall)
        #    Only counts interactions where LLM actually took action.
        #    LLM-idle traces ("status ok") have no action to learn or match.
        cloud_tc_flags = [
            1 if (t.get("execution", {}).get("mode") == "cloud"
                  and t.get("llm_response", {}).get("tool_calls"))
            else 0
            for t in traces
        ]
        actionable_total = sum(local_flags) + sum(cloud_tc_flags)

        lo, hi = bootstrap_ci(local_flags, seed=42)

        ours_agree, ours_agree_n = compute_ours_decision_agreement(traces, exp_dir)
        ours_eval = compute_ours_eval_window(traces, exp_dir, day_bounds)

        all_results[label] = {
            "traces": len(traces),
            "system_autonomy_rate": round(sum(local_flags) / max(1, len(local_flags)) * 100, 1),
            "system_autonomy_rate_eval_window": ours_eval["ar_eval_window"],
            "system_actionable_ar": round(sum(local_flags) / max(1, actionable_total) * 100, 1),
            "cloud_with_toolcall": sum(cloud_tc_flags),
            "cloud_idle": sum(1 for t in traces
                if t.get("execution", {}).get("mode") == "cloud"
                and not t.get("llm_response", {}).get("tool_calls")),
            "bootstrap_95ci_ar": [round(lo * 100, 2), round(hi * 100, 2)],
            "system_decision_agreement_pct": ours_agree,
            "system_decision_agreement_n": ours_agree_n,
            "eval_window": ours_eval,
            "baselines": res,
        }
        print(f"[{label}] {len(traces)} traces, "
              f"AR_full={all_results[label]['system_autonomy_rate']}% "
              f"AR_act={all_results[label]['system_actionable_ar']}% "
              f"AGREE={ours_agree}% (ours)")
        for name in baselines:
            r = res[name]
            dn = r.get("display_name", name)
            print(f"    {name:<24s} AR={r['autonomy_rate']:6.1f}%  "
                  f"AGREE={r.get('decision_agreement_pct', 0):6.1f}%  ({dn})")

    # Append Ours (system AR) to each row
    for i, label in enumerate(dataset_names):
        ar_row = ar_matrix[i]
        ours_ar = all_results[label]["system_autonomy_rate_eval_window"]
        ar_row.append(ours_ar)
        # Also add to CCR matrix (same value for AR=CCR)
        ccr_matrix[i].append(ours_ar)

    baseline_names = list(make_baselines().keys()) + ["Ours"]

    # Friedman + Nemenyi（AR）
    friedman_ar = friedman_test(ar_matrix, higher_is_better=True)
    nemenyi_ar = nemenyi_posthoc(
        friedman_ar["mean_ranks"], friedman_ar["n_blocks"], baseline_names)

    # Friedman + Nemenyi（CCR）
    friedman_ccr = friedman_test(ccr_matrix, higher_is_better=True)
    nemenyi_ccr = nemenyi_posthoc(
        friedman_ccr["mean_ranks"], friedman_ccr["n_blocks"], baseline_names)

    stats = {
        "datasets": dataset_names,
        "baselines": baseline_names,
        "ar_matrix_percent": ar_matrix,
        "ccr_matrix_percent": ccr_matrix,
        "friedman_ar": friedman_ar,
        "nemenyi_ar": nemenyi_ar,
        "friedman_ccr": friedman_ccr,
        "nemenyi_ccr": nemenyi_ccr,
        "note": "All methods evaluated on the same held-out window "
                "(days int(30*0.7)+1..30). Batch baselines train on the first 70% "
                "of days only; online baselines (Online-DT, ESP-Claw, Ours) learn "
                "incrementally with past-only information. decision_agreement is "
                "computed separately on 60 teacher-replay snapshots per dataset by "
                "merge_statistics.py (see teacher_replay_results.json). "
                "Friedman p<0.05 = significant overall difference; Nemenyi pair "
                "significant if rank_diff > critical_difference.",
    }

    with open(os.path.join(OUTPUT_DIR, "baseline_results_4x.json"),
              "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, "statistics_4x_baselines.json"),
              "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n=== FRIEDMAN (AR) ===")
    print(json.dumps(friedman_ar, ensure_ascii=False, indent=2))
    print("\nSaved: output/baseline_results_4x.json + "
          "output/statistics_4x_baselines.json")

    # === v10.6: MCU-feasible Sensor-Vector Cache 多阈值扫参 ===
    print("\n" + "=" * 60)
    print("  Sensor-Vector Cache Threshold Sweep")
    print("=" * 60)
    seed42_traces = load_traces("run4b_seed42_seed42")
    sweep_baselines = make_semantic_sweep(seed=42)
    sweep_res = run_baseline_comparison(
        seed42_traces, sweep_baselines, seed=42, train_ratio=1.0)
    print(f"\n  {'Baseline':<28s} {'AR%':>6s} {'AGREE%':>8s} {'Cache':>6s}")
    print(f"  {'-'*52}")
    sweep_summary = {}
    for name in sweep_baselines:
        r = sweep_res[name]
        dn = r.get("display_name", name)
        print(f"  {name:<28s} {r['autonomy_rate']:6.1f}  "
              f"{r.get('decision_agreement_pct', 0):7.1f}  "
              f"{r.get('cache_size', 0):5d}  ({dn})")
        sweep_summary[name] = {
            "ar": r["autonomy_rate"],
            "agree": r.get("decision_agreement_pct", 0),
            "cache_size": r.get("cache_size", 0),
        }
    # Save sweep
    sweep_path = os.path.join(OUTPUT_DIR, "semantic_sweep_results.json")
    # UCI V3 真实数据对照
    uci3_traces = load_traces("run4b_uci_seed42")
    uci3_sweep_res = run_baseline_comparison(
        uci3_traces, sweep_baselines, seed=42, train_ratio=1.0)
    sweep_data = {
        "seed42": sweep_summary,
        "uci_v3": {name: {
            "ar": uci3_sweep_res[name]["autonomy_rate"],
            "agree": uci3_sweep_res[name].get("decision_agreement_pct", 0),
            "cache_size": uci3_sweep_res[name].get("cache_size", 0),
        } for name in sweep_baselines},
    }
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_data, f, indent=2, ensure_ascii=False)
    print(f"\n  UCI V3 sweep:")
    for name in sweep_baselines:
        r = uci3_sweep_res[name]
        print(f"  {name:<28s} AR={r['autonomy_rate']:5.1f}%  AGREE={r.get('decision_agreement_pct', 0):5.1f}%")
    print(f"\n  Sweep saved: {sweep_path}")


if __name__ == "__main__":
    main()
