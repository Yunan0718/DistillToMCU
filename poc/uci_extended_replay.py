"""
UCI extended teacher-replay (v10.6 follow-up)
============================================
Runs the same teacher-consistency + all-methods precision/recall/agreement
evaluation as teacher_replay.py, but ONLY for UCI and with a larger snapshot
count (240 instead of 60) to narrow the Wilson intervals behind the
"calibrated abstention vs aggressive imitation" claim.

Outputs (kept separate so the canonical 60-snapshot files are untouched):
  output/llm_consistency_results_uci240.json
  output/uci_extended_replay_results.json
"""

import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm_client
from llm_consistency_experiment import (
    DATASETS,
    load_trace_items,
    select_diverse_snapshots,
    extract_action,
    _one_call,
)
from run_full_analysis import load_traces, load_day_bounds
from teacher_replay import (
    _teacher_data,
    _action_key,
    _ours_engine_from_snapshot,
    _ours_engine_warm,
    _engine_action,
    _dt_action,
    _eval_method_on_snapshots,
    _make_oneshot,
    _oneshot_action,
    _userdef_action,
    _esp_action,
    _exact_action,
    _eval_online_dt,
    _chronological_snapshot_order,
    TRAIN_RATIO,
)
from baselines import (
    ExactCacheBaseline,
    UserDefinedRulesBaseline,
    DecisionTreeBaseline,
    OnlineDailyRefitDecisionTreeBaseline,
    ESPClawStyleBaseline,
    extract_cloud_action,
    trace_day_labels,
)
from teacher_replay import USER_RULES_BY_LABEL

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
N_SNAPSHOTS = 240
LABEL = "uci_v3"


def run_uci_consistency(n: int = N_SNAPSHOTS, seed: int = 42,
                        tag: str = None, workers: int = 20):
    cfg = DATASETS[LABEL]
    items = load_trace_items(cfg["dir"])
    selected = select_diverse_snapshots(items, n, seed=seed)
    print(f"[consistency] loaded {len(items)} traces, selected {len(selected)} snapshots")

    import concurrent.futures

    calls = []
    for si, snap in enumerate(selected):
        for _ in range(3):
            calls.append((si, snap, cfg["prompt"], 0.0, "deepseek-v4-flash"))
        for _ in range(2):
            calls.append((si, snap, cfg["prompt"], 0.7, "deepseek-v4-flash"))
        calls.append((si, snap, cfg["prompt"], 0.0, "qwen3.7-flash-2026-07-15"))

    responses = [None] * len(calls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one_call, c[1:]): i for i, c in enumerate(calls)}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                responses[i] = fut.result()
            except Exception:
                responses[i] = (None, 0)
            done += 1
            if done % 240 == 0:
                print(f"    [{done}/{len(calls)}] done")

    rows = []
    for si, snap in enumerate(selected):
        base = si * 6
        ds_t0 = [responses[base + k][0] for k in range(3)]
        ds_t07 = [responses[base + 3 + k][0] for k in range(2)]
        qwen = responses[base + 5][0]
        c0 = Counter(a for a in ds_t0 if a is not None)
        majority = c0.most_common(1)[0][0] if c0 else None
        rows.append({
            "snapshot_idx": si,
            "sensors": {k: round(v, 2) if isinstance(v, float) else v
                        for k, v in snap.get("sensors", {}).items()
                        if v is not None},
            "user_input": snap.get("user_input", ""),
            "ds_t0_actions": ds_t0,
            "ds_t07_actions": ds_t07,
            "qwen_action": qwen,
            "ds_t0_majority": majority,
            "ds_t0_consistent": len(set(a for a in ds_t0 if a is not None)) <= 1,
            "ds_t07_consistent": len(set(a for a in ds_t07 if a is not None)) <= 1,
            "avg_latency_ms": round(sum(r[1] for r in responses[base:base + 6]) / 6),
        })

    payload = {
        "summary": {
            "total_api_calls": len(calls),
            "total_snapshots": len(selected),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "details": {
            LABEL: {
                "dataset": cfg["dir"],
                "system_prompt": cfg["prompt"],
                "n_snapshots_selected": len(selected),
                "rows": rows,
            },
        },
    }
    tag = tag or f"{len(selected)}_seed{seed}"
    p = os.path.join(OUT, f"llm_consistency_results_{LABEL}_{tag}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[consistency] wrote {p}")
    return payload["details"]


def evaluate_uci(details, tag: str = None):
    snapshots = _teacher_data(details, LABEL)
    if not snapshots:
        raise SystemExit("no UCI teacher data produced")

    cfg = DATASETS[LABEL]
    exp_dir = cfg["dir"]
    traces = load_traces(exp_dir)
    day_bounds = load_day_bounds(exp_dir)
    day_labels = trace_day_labels(len(traces), day_bounds)
    n_days = max(day_labels)
    split_day = int(n_days * TRAIN_RATIO) + 1
    warm_traces = [t for t, d in zip(traces, day_labels) if d < split_day]
    warm_cloud = [t for t in warm_traces
                  if t.get("execution", {}).get("mode") == "cloud"]

    ceil_n = sum(1 for s in snapshots if s["self_consistent"])
    ceiling = round(ceil_n / max(1, len(snapshots)) * 100, 1)

    eng_warm = _ours_engine_warm(traces, day_labels, split_day)

    dt = DecisionTreeBaseline(seed=42)
    dt.train(warm_cloud)
    dt_feature_names = list(dt._feature_names)

    warm_enriched = []
    for t in warm_cloud:
        e = dict(t)
        e["_cloud_action"] = extract_cloud_action(t)
        warm_enriched.append(e)

    exact = ExactCacheBaseline(seed=42)
    for t in warm_cloud:
        act = extract_cloud_action(t)
        if act:
            key = frozenset(
                (k, round(v, 1) if isinstance(v, float) else v)
                for k, v in sorted(t.get("sensors", {}).items())
                if v is not None)
            exact.cache[key] = act

    ud = UserDefinedRulesBaseline(
        seed=42,
        rules=USER_RULES_BY_LABEL.get(LABEL))
    esp = ESPClawStyleBaseline(seed=42)
    for t in warm_cloud:
        act = extract_cloud_action(t)
        if act:
            esp._learn_from_cloud(t.get("sensors", {}), act)

    oneshot = _make_oneshot()

    methods = {
        "Ours (warm rules)": lambda s, e=eng_warm: _engine_action(e, s),
        "Decision Tree (batch)": lambda s, t=dt:
            _dt_action(t._tree, s, dt_feature_names),
        "User-defined Rules": lambda s: _userdef_action(ud, s),
        "LLM One-shot": lambda s: _oneshot_action(oneshot, s),
        "ESP-Claw-style": lambda s: _esp_action(esp, s),
        "Exact Cache": lambda s: _exact_action(exact, s),
        "Pure Cloud": lambda s: None,
    }
    method_rows = {name: _eval_method_on_snapshots(fn, snapshots)
                   for name, fn in methods.items()}

    online_dt = OnlineDailyRefitDecisionTreeBaseline(seed=42)
    online_dt.train(warm_enriched)
    order = _chronological_snapshot_order(traces, snapshots)
    method_rows["Decision Tree (online refit)"] = _eval_online_dt(
        online_dt, traces, day_labels, order, snapshots)

    result = {
        "exp_dir": exp_dir,
        "n_snapshots": len(snapshots),
        "teacher_self_agreement_pct": ceiling,
        "teacher_self_agreement_n": ceil_n,
        "methods": method_rows,
    }
    tag = tag or "extended"
    p = os.path.join(OUT, f"{LABEL}_extended_replay_results_{tag}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] wrote {p}")
    return result


def _wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (centre - half, centre + half)


def _fmt_method(name, r):
    prec = r["precision_pct"]
    if prec is None:
        ci = "-"
    else:
        lo, hi = _wilson(prec / 100, r["n_local"])
        ci = f"{lo*100:.1f}-{hi*100:.1f}%"
    return (f"{name:28s} P={prec}% (Wilson {ci})  R={r['recall_pct']}%  "
            f"n_teacher={r['n_teacher_act']}  n_local={r['n_local']}  "
            f"agree={r['agree_pct']}%")


def _agg(values):
    if not values:
        return None
    m = sum(values) / len(values)
    if len(values) == 1:
        return f"{m:.1f}%"
    sd = (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    return f"{m:.1f}% ± {sd:.1f}%"


if __name__ == "__main__":
    seeds = [42, 123, 999]
    n = 480
    results = {}
    for seed in seeds:
        tag = f"{n}_seed{seed}"
        details = run_uci_consistency(n, seed=seed, tag=tag)
        res = evaluate_uci(details, tag=tag)
        results[seed] = res
        print("\n" + "=" * 70)
        print(f"UCI seed={seed} ({res['n_snapshots']} snapshots)  "
              f"teacher self-agreement {res['teacher_self_agreement_pct']}%")
        print("=" * 70)
        for name, r in res["methods"].items():
            print(_fmt_method(name, r))

    # 跨 seed 汇总（按方法聚合 precision / recall / n_local / n_teacher_act）
    print("\n" + "#" * 70)
    print("Across-seed summary (precision mean +/- sd; recall; pooled n)")
    print("#" * 70)
    method_names = list(results[seeds[0]]["methods"].keys())
    pooled = {}
    for mn in method_names:
        precs = [results[s]["methods"][mn]["precision_pct"]
                 for s in seeds
                 if results[s]["methods"][mn]["precision_pct"] is not None]
        recs = [results[s]["methods"][mn]["recall_pct"] for s in seeds]
        n_teacher = sum(results[s]["methods"][mn]["n_teacher_act"] for s in seeds)
        n_local = sum(results[s]["methods"][mn]["n_local"] for s in seeds)
        pooled[mn] = {
            "precision": _agg(precs),
            "recall": _agg(recs),
            "n_teacher": n_teacher,
            "n_local": n_local,
        }
    for mn in method_names:
        p = pooled[mn]
        print(f"{mn:28s} P={p['precision']}  R={p['recall']}  "
              f"pooled n_teacher={p['n_teacher']}  pooled n_local={p['n_local']}")

    agg_path = os.path.join(OUT, "uci_extended_replay_results_multiseed.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "n": n, "results": results,
                   "pooled": pooled}, f, indent=2, ensure_ascii=False)
    print(f"\n[aggregate] wrote {agg_path}")
