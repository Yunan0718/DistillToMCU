"""
DistillToMCU — 4x repeat sweep analysis
=======================================
Reads run4_* output dirs and produces:
  - per-dataset AR mean/std + bootstrap 95% CI (n=4)
  - growth curves (day-by-day) mean +/- CI
  - per-dataset rule counts / active / final states
  - JSON summary for figures

Usage:
    python analyze_4x.py [--out figures/data_4x.json]
"""

import argparse
import json
import os
import random
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "poc", "output")
SEEDS = [42, 123, 999, 777]

DATASETS = {
    "seed42":  {"dirs": [f"run4b_seed42_seed{s}" for s in SEEDS],
                "label": "Synthetic seed-42 data"},
    "seed123": {"dirs": [f"run4b_seed123_seed{s}" for s in SEEDS],
                "label": "Synthetic seed-123 data"},
    "seed999": {"dirs": [f"run4b_seed999_seed{s}" for s in SEEDS],
                "label": "Synthetic seed-999 data"},
    "seed777": {"dirs": [f"run4b_seed777_seed{s}" for s in SEEDS],
                "label": "Synthetic seed-777 data"},
    "strands": {"dirs": [f"run4b_strands_seed{s}" for s in SEEDS],
                "label": "STRANDS Aruba-1"},
    "uci_v3":  {"dirs": [f"run4b_uci_seed{s}" for s in SEEDS],
                "label": "UCI V3 (Real sensors)"},
}

# v10.6: UCI 追加 2 次独立重复（n=6）
EXTRA_SEEDS = {"uci_v3": [2026, 31415]}

for _k, _extra in EXTRA_SEEDS.items():
    DATASETS[_k]["dirs"] += [f"run4b_uci_seed{s}" for s in _extra]


def load_metrics(exp_dir):
    p = os.path.join(OUT, exp_dir, "metrics.jsonl")
    if not os.path.exists(p):
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.05):
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in values]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return round(lo, 2), round(hi, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "data_4x.json"))
    args = ap.parse_args()

    summary = {}
    for key, cfg in DATASETS.items():
        ars, rules, active = [], [], []
        curves = []
        for d in cfg["dirs"]:
            m = load_metrics(d)
            if not m:
                print(f"  [skip] {d}: no metrics")
                continue
            ars.append(m[-1]["autonomy_rate"])
            rules.append(m[-1].get("total_rules"))
            active.append(m[-1].get("active_rules"))
            curves.append(m)
        if not ars:
            continue
        n = max(len(c) for c in curves)
        daily_mean, daily_lo, daily_hi = [], [], []
        for day in range(1, n + 1):
            vals = [c[day - 1]["autonomy_rate"] for c in curves
                    if len(c) >= day]
            daily_mean.append(round(statistics.mean(vals), 2))
            lo, hi = bootstrap_ci(vals)
            daily_lo.append(lo if lo is not None else None)
            daily_hi.append(hi if hi is not None else None)
        mean = statistics.mean(ars)
        sd = statistics.stdev(ars) if len(ars) > 1 else 0.0
        lo, hi = bootstrap_ci(ars)
        summary[key] = {
            "label": cfg["label"],
            "n_runs": len(ars),
            "ar_runs": ars,
            "ar_mean": round(mean, 2),
            "ar_std": round(sd, 2),
            "ar_ci95": [lo, hi],
            "rules_mean": round(statistics.mean(rules), 1) if rules else None,
            "active_mean": round(statistics.mean(active), 1) if active else None,
            "growth": {"mean": daily_mean, "ci_lo": daily_lo, "ci_hi": daily_hi},
        }
        print(f"{key}: AR {ars} -> {mean:.1f}% ± {sd:.1f}  "
              f"CI95={lo}-{hi}  rules={summary[key]['rules_mean']} "
              f"active={summary[key]['active_mean']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
