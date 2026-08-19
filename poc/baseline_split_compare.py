"""
DistillToMCU — D1: baseline in-sample vs held-out comparison (v10.6)
====================================================================
Same 24 run4b dirs, two train/eval protocols for trainable baselines:
  - in-sample (train_ratio=1.0): train on all traces, replay all
  - held-out  (train_ratio=0.7): train/warm-up on the first 70% of days,
    evaluate ONLY on the last 30% of days (true held-out window)
Ours (system AR) is identical in both rows.

Output: output/baseline_split_compare.json

Usage:
    python baseline_split_compare.py
"""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_full_analysis import build_experiments, make_baselines, \
    dataset_key_of, load_traces, load_day_bounds
from baselines import run_baseline_comparison

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    exps = build_experiments()
    rows = {"in_sample": {}, "held_out": {}}
    for ratio, key in [(1.0, "in_sample"), (0.7, "held_out")]:
        for exp_dir, label in exps:
            traces = load_traces(exp_dir)
            day_bounds = load_day_bounds(exp_dir)
            baselines = make_baselines(seed=42,
                                       dataset_key=dataset_key_of(exp_dir))
            res = run_baseline_comparison(traces, baselines, seed=42,
                                          train_ratio=ratio,
                                          day_bounds=day_bounds)
            rows[key][label] = {
                name: {"ar": r["autonomy_rate"],
                       "agree": r.get("decision_agreement_pct", 0.0)}
                for name, r in res.items()
            }
        print(f"{key}: {len(rows[key])} runs done")

    # Summarize means per method
    summary = {}
    names = list(rows["held_out"][list(rows["held_out"])[0]].keys())
    for key in ["in_sample", "held_out"]:
        summary[key] = {}
        for n in names:
            ars = [rows[key][l][n]["ar"] for l in rows[key]]
            agrees = [rows[key][l][n]["agree"] for l in rows[key]]
            summary[key][n] = {
                "ar_mean": round(statistics.mean(ars), 1),
                "ar_std": round(statistics.stdev(ars), 1) if len(ars) > 1 else 0,
                "agree_mean": round(statistics.mean(agrees), 1),
            }

    print("\n=== AR mean (24 runs) ===")
    for n in names:
        s1 = summary["in_sample"][n]
        s2 = summary["held_out"][n]
        print(f"  {n:<26s} in={s1['ar_mean']:6.1f}%  held={s2['ar_mean']:6.1f}%  "
              f"(delta {s2['ar_mean']-s1['ar_mean']:+.1f})")

    out = {"protocol": {
        "in_sample": "train all traces, replay all (in-sample)",
        "held_out": "train/warm-up first 70% of days, evaluate last 30% of days only",
        "ours": "incremental daily distillation (identical in both rows)"},
        "per_run": rows, "summary": summary}
    p = os.path.join(OUT, "baseline_split_compare.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
