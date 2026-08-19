"""Build figures/data_4x.json for all 9 datasets (4x growth curves).

Reads run4b_{dataset}_seed{42,123,999,777}/metrics.jsonl and computes per-day
mean AR with t-distribution 95% CI (n=4), plus final AR stats and rule counts.
"""

import json
import math
import os
import statistics

FIG = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(FIG), "poc", "output")
T95_N4 = 3.182  # t(3, 0.975)

DATASETS = {
    "seed42": "Synthetic 42",
    "seed123": "Synthetic 123",
    "seed999": "Synthetic 999",
    "seed777": "Synthetic 777",
    "strands": "STRANDS Aruba-1",
    "uci_v3": "UCI V3 (Real)",
    "sml2010": "SML2010 (Real)",
    "steel": "Steel Ind. (Real)",
    "airquality": "Air Quality (Real)",
}
SEEDS = [42, 123, 999, 777]
DIR_MAP = {"uci_v3": "uci"}  # run4b 目录名与数据集 key 的映射


def load_days(dataset):
    per_seed = []
    for s in SEEDS:
        dname = DIR_MAP.get(dataset, dataset)
        p = os.path.join(OUT, f"run4b_{dname}_seed{s}", "metrics.jsonl")
        days = []
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        days.append(json.loads(line))
        per_seed.append(days)
    return per_seed


def build(dataset):
    per_seed = load_days(dataset)
    n_days = max((len(d) for d in per_seed), default=0)
    if n_days == 0:
        return None
    mean, ci_lo, ci_hi = [], [], []
    ar_runs = []
    rules_last, active_last = [], []
    for d in range(1, n_days + 1):
        vals = []
        for days in per_seed:
            m = next((x for x in days if x["day"] == d), None)
            if m is not None:
                vals.append(m["autonomy_rate"])
        if vals:
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            mean.append(round(m, 2))
            ci_lo.append(round(m - T95_N4 * sd / math.sqrt(len(vals)), 2))
            ci_hi.append(round(m + T95_N4 * sd / math.sqrt(len(vals)), 2))
        else:
            mean.append(None)
            ci_lo.append(None)
            ci_hi.append(None)
    for days in per_seed:
        if days:
            ar_runs.append(days[-1]["autonomy_rate"])
            rules_last.append(days[-1].get("total_rules", 0))
            active_last.append(days[-1].get("active_rules", 0))
    return {
        "label": DATASETS[dataset],
        "n_runs": len(ar_runs),
        "ar_runs": ar_runs,
        "ar_mean": round(statistics.mean(ar_runs), 2),
        "ar_std": round(statistics.stdev(ar_runs), 2) if len(ar_runs) > 1 else 0.0,
        "rules_mean": round(statistics.mean(rules_last), 1) if rules_last else 0,
        "active_mean": round(statistics.mean(active_last), 1) if active_last else 0,
        "growth": {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi},
    }


def main():
    out = {}
    for k in DATASETS:
        d = build(k)
        if d:
            out[k] = d
            print(f"[{k}] ar_runs={d['ar_runs']} mean={d['ar_mean']}")
    p = os.path.join(FIG, "data_4x.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"Saved: {p} ({len(out)} datasets)")


if __name__ == "__main__":
    main()
