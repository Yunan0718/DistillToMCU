"""
Oracle Replay: Full-Information Rule Capacity Upper Bound
===========================================================
Measures the maximum achievable AR if rules were pre-learned
from ALL traces (full batch distillation, same-trace replay).

This is NOT the online/incremental AR. It's the capacity ceiling:
"if the system had perfect knowledge of all past and future
LLM decisions, what fraction could it handle locally?"

Usage:
    python oracle_replay.py --dataset seed42
    python oracle_replay.py --all

Output: output/oracle_replay_<dataset>.json
"""

import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from rule_engine import RuleEngine
from distiller import Distiller
from trace_store import TraceStore

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

DATASETS = {
    "seed42":        ("seed42",         "output/seed42/traces.jsonl"),
    "seed123":       ("run_seed123",    "output/run_seed123/traces.jsonl"),
    "seed999":       ("run_seed999",    "output/run_seed999/traces.jsonl"),
    "seed777":       ("seed777",        "output/seed777/traces.jsonl"),
    "strands":       ("strands_seed42", "output/strands_seed42/traces.jsonl"),
    "uci_v3":        ("uci_v3_seed42",  "output/uci_v3_seed42/traces.jsonl"),
    "sml2010":       ("run4b_sml2010_seed42", "output/run4b_sml2010_seed42/traces.jsonl"),
    "steel":         ("run4b_steel_seed42",   "output/run4b_steel_seed42/traces.jsonl"),
    "airquality":    ("run4b_airquality_seed42", "output/run4b_airquality_seed42/traces.jsonl"),
}


def oracle_replay(exp_dir: str) -> dict:
    """Full batch distill + same-trace replay on one dataset."""
    traces_path = os.path.join(OUTPUT_DIR, exp_dir, "traces.jsonl")
    if not os.path.exists(traces_path):
        raise FileNotFoundError(traces_path)

    # Load all traces
    with open(traces_path, "r", encoding="utf-8") as f:
        traces = [json.loads(l) for l in f if l.strip()]

    # Full batch distillation from ALL traces
    engine = RuleEngine()
    distiller = Distiller(engine, llm_client=None)
    distiller.distill(traces)

    # Replay same traces through learned rules
    local = 0
    cloud_w_tc = 0
    cloud_idle = 0

    for t in traces:
        sensors = t.get("sensors", {})
        matches = engine.match(sensors)
        best = engine.resolve_conflict(matches)
        if best:
            local += 1
        else:
            mode = t.get("execution", {}).get("mode", "cloud")
            tc = t.get("llm_response", {}).get("tool_calls")
            if mode == "cloud":
                if tc:
                    cloud_w_tc += 1
                else:
                    cloud_idle += 1

    total = len(traces)
    actionable = local + cloud_w_tc
    ar_full = local / max(1, total) * 100
    ar_act = local / max(1, actionable) * 100
    rules = engine.rules

    return {
        "dataset": exp_dir,
        "traces_total": total,
        "local": local,
        "cloud_with_toolcall": cloud_w_tc,
        "cloud_idle": cloud_idle,
        "ar_full_pct": round(ar_full, 1),
        "ar_actionable_pct": round(ar_act, 1),
        "rules_total": len(rules),
        "rules_active": sum(1 for r in rules.values() if r.state == "active"),
        "rules_verified": sum(1 for r in rules.values() if r.state == "verified"),
        "rule_conditions_range": [
            min(len(r.conditions) for r in rules.values()) if rules else 0,
            max(len(r.conditions) for r in rules.values()) if rules else 0,
        ],
        "method_note": "Full batch distill from ALL traces, same-trace replay. "
                       "This is the ORACLE upper bound, NOT incremental/online AR.",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Oracle replay: full-information rule capacity upper bound")
    parser.add_argument("--dataset", default=None, help="Dataset key (seed42, uci_v3, ...)")
    parser.add_argument("--all", action="store_true", help="Run on all datasets")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.all:
        targets = DATASETS
    elif args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset '{args.dataset}'. Available: {list(DATASETS.keys())}")
            sys.exit(1)
        targets = {args.dataset: DATASETS[args.dataset]}
    else:
        print("Usage: --dataset <name> or --all")
        sys.exit(1)

    all_results = {}
    for key, (label, _) in targets.items():
        result = oracle_replay(label)
        all_results[key] = result
        print(f"{key}: rules={result['rules_total']} conds={result['rule_conditions_range']} "
              f"AR_full={result['ar_full_pct']}% AR_act={result['ar_actionable_pct']}% "
              f"local={result['local']}/{result['traces_total']}")

        out_path = os.path.join(OUTPUT_DIR, f"oracle_replay_{key}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Save combined
    combined_path = os.path.join(OUTPUT_DIR, "oracle_replay_all.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved individual results + combined: {combined_path}")


if __name__ == "__main__":
    main()
