"""
DistillToMCU — merge AR (24-block held-out) + AGREE (60 teacher-replay snapshots)
================================================================================
statistics_4x_baselines.json 最终版：
  - AR / CCR: 24 blocks (6 datasets x 4 seeds) x 8 methods, held-out window
  - decision agreement: 6 datasets x 8 methods, measured on the SAME 60
    teacher-replay snapshots per dataset (ground truth = DeepSeek majority)
  - Friedman + Nemenyi for each metric

Usage:
    python run_full_analysis.py       # regenerates AR part
    python merge_statistics.py        # merges teacher-replay AGREE part
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statistics_tests import friedman_test, nemenyi_posthoc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

TR_METHOD_TO_BASELINE = {
    "B1 Pure Cloud": "Pure Cloud",
    "B2 Exact Cache": "Exact Cache",
    "B3 User-defined Rules": "User-defined Rules",
    "B5 Decision Tree": "Decision Tree (batch)",
    "B5b Decision Tree (online refit)": "Decision Tree (online refit)",
    "B6 ESP-Claw-style": "ESP-Claw-style",
    "B4 LLM One-shot": "LLM One-shot",
    "Ours": "Ours (warm rules)",
}

DATASET_ORDER = [
    "synthetic_seed42", "synthetic_seed123", "synthetic_seed999",
    "synthetic_seed777", "strands_aruba1", "uci_v3",
    "sml2010", "steel", "airquality",
]


def main():
    p = os.path.join(OUT, "statistics_4x_baselines.json")
    stats = json.load(open(p, encoding="utf-8"))
    tr_path = os.path.join(OUT, "teacher_replay_results.json")
    if not os.path.exists(tr_path):
        print(f"[SKIP] {tr_path} not found — run the teacher-replay "
              f"experiment first.")
        return
    tr = json.load(open(tr_path, encoding="utf-8"))

    baselines = stats["baselines"]
    agree_matrix = []
    for ds in DATASET_ORDER:
        d = tr.get(ds)
        if not d:
            continue
        methods = d["methods"]
        row = []
        for name in baselines:
            tr_name = TR_METHOD_TO_BASELINE[name]
            row.append(methods.get(tr_name, {}).get("agree_pct", 0.0))
        agree_matrix.append(row)

    friedman_agree = friedman_test(agree_matrix, higher_is_better=True)
    nemenyi_agree = nemenyi_posthoc(
        friedman_agree["mean_ranks"], friedman_agree["n_blocks"], baselines)

    stats["decision_agreement_matrix_percent"] = agree_matrix
    stats["decision_agreement_datasets"] = [
        ds for ds in DATASET_ORDER if ds in tr]
    stats["friedman_decision_agreement"] = friedman_agree
    stats["nemenyi_decision_agreement"] = nemenyi_agree
    stats["note"] += (" decision_agreement = method local action (device+command) "
                      "== DeepSeek majority decision on the same 60 teacher-replay "
                      "snapshots per dataset (6 blocks x 8 methods); Pure Cloud never "
                      "acts locally (0 by definition), Exact Cache replays exact "
                      "teacher decisions seen during warm-up.")

    json.dump(stats, open(p, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    print(f"AGREE Friedman (6 blocks x {len(baselines)} methods): "
          f"p={friedman_agree['p_value']} mean_ranks={friedman_agree['mean_ranks']}")
    print(f"Saved merged: {p}")


if __name__ == "__main__":
    main()
