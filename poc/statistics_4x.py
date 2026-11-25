"""
DistillToMCU — 4x repeat sweep statistics (v10.5)
=================================================
Friedman + Nemenyi over 6 datasets x 4 seeds using the run4b AR values,
plus per-dataset mean/std/bootstrap CI. Produces output/statistics_4x.json
and reuses statistics_tests.py.

Usage:
    python statistics_4x.py
"""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statistics_tests import friedman_test, nemenyi_posthoc, bootstrap_ci

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SEEDS = [42, 123, 999, 777]

# dataset key -> run4b dir template
DATASETS = {
    "seed42":  "run4b_seed42_seed{}",
    "seed123": "run4b_seed123_seed{}",
    "seed999": "run4b_seed999_seed{}",
    "seed777": "run4b_seed777_seed{}",
    "strands": "run4b_strands_seed{}",
    "uci_v3":  "run4b_uci_seed{}",
    "sml2010": "run4b_sml2010_seed{}",
    "steel":   "run4b_steel_seed{}",
    "airquality": "run4b_airquality_seed{}",
}

# v10.6: UCI 真实数据方差大（9.2-55%），追加 2 次独立重复 → n=6。
EXTRA_SEEDS = {
    "uci_v3": [2026, 31415],
}


def load_ar(exp_dir):
    p = os.path.join(OUT, exp_dir, "metrics.jsonl")
    ar = None
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ar = json.loads(line).get("autonomy_rate")
    except Exception:
        return None
    return ar


def main():
    keys = list(DATASETS)
    matrix = []  # rows = datasets, cols = seeds
    stats = {}
    for k in keys:
        ars = []
        for s in SEEDS:
            v = load_ar(DATASETS[k].format(s))
            if v is not None:
                ars.append(v)
        for s in EXTRA_SEEDS.get(k, []):
            v = load_ar(DATASETS[k].format(s))
            if v is not None:
                ars.append(v)
        # Friedman 需要平衡设计：跨种子检验只用前 4 个种子；
        # UCI 的均值/方差/CI 用全部 n=6。
        matrix.append(ars[:4])
        lo, hi = bootstrap_ci(ars)
        stats[k] = {
            "runs": ars,
            "mean": round(statistics.mean(ars), 2),
            "std": round(statistics.stdev(ars), 2) if len(ars) > 1 else 0.0,
            "ci95": [lo, hi],
        }
        print(f"{k}: {ars} -> {stats[k]['mean']} ± {stats[k]['std']} "
              f"CI={stats[k]['ci95']}")

    # Friedman: rows = 6 datasets (blocks), cols = 4 seeds (treatments)
    # Question: do the 4 repeat runs differ? (should NOT — stability)
    fr = friedman_test(matrix)
    nem = nemenyi_posthoc(fr["mean_ranks"], len(matrix), SEEDS)
    print(f"\nFriedman (4 repeats): p={fr['p_value']:.2e} "
          f"chi2={fr['chi2']:.3f}")
    print(f"Nemenyi CD={nem['critical_difference']:.3f}")

    out = {
        "design": ("6 datasets x 4 fixed seeds (42/123/999/777), v10.2; "
                   "UCI augmented with 2 extra independent LLM repeats (seeds "
                   "2026/31415) -> n=6, v10.6"),
        "per_dataset": stats,
        "friedman_across_seeds": fr,
        "nemenyi_across_seeds": nem,
    }
    p = os.path.join(OUT, "statistics_4x.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
