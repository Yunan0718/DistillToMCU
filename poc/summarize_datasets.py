"""Summarize the 9-dataset results for reporting."""

import json
import os
import statistics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    br = json.load(open(os.path.join(OUT, "baseline_results_4x.json"),
                        encoding="utf-8"))
    tr = json.load(open(os.path.join(OUT, "teacher_replay_results.json"),
                        encoding="utf-8"))
    stats = json.load(open(os.path.join(OUT, "statistics_4x_baselines.json"),
                           encoding="utf-8"))

    print("=== Teacher-replay P/R per dataset (n_act | Ours | DT batch | ESP-Claw) ===")
    order = ["synthetic_seed42", "synthetic_seed123", "synthetic_seed999",
             "synthetic_seed777", "strands_aruba1", "uci_v3",
             "sml2010", "steel", "airquality"]
    for k in order:
        d = tr.get(k)
        if not d:
            continue
        m = d["methods"]
        o = m["Ours (warm rules)"]
        dt = m["Decision Tree (batch)"]
        esp = m["ESP-Claw-style"]
        print(f"{k:20s} n_act={o['n_teacher_act']:3d}  "
              f"Ours P={o['precision_pct']} R={o['recall_pct']}  "
              f"DT P={dt['precision_pct']} R={dt['recall_pct']}  "
              f"ESP P={esp['precision_pct']} R={esp['recall_pct']}")

    print("\n=== Held-out AR (days 22-30) per dataset (4 seeds) ===")
    prefix_map = {
        "seed42": "synthetic_seed42", "seed123": "synthetic_seed123",
        "seed999": "synthetic_seed999", "seed777": "synthetic_seed777",
        "strands": "strands", "uci_v3": "uci_v3",
        "sml2010": "sml2010", "steel": "steel", "airquality": "airquality",
    }
    for k, prefix in prefix_map.items():
        runs = []
        for key, d in br.items():
            if key.startswith(prefix + "_") and d:
                runs.append(d["system_autonomy_rate_eval_window"])
        if runs:
            m = statistics.mean(runs)
            sd = statistics.stdev(runs) if len(runs) > 1 else 0.0
            print(f"{k:12s} runs={runs}  mean={m:.1f} std={sd:.1f}")

    print("\n=== Friedman / Nemenyi (36 blocks x 8 methods) ===")
    print("AR friedman p:", stats["friedman_ar"]["p_value"])
    print("AR mean ranks:", [round(x, 2) for x in
                             stats["friedman_ar"]["mean_ranks"]])
    print("Nemenyi AR CD:", stats["nemenyi_ar"]["critical_difference"])
    print("Agree friedman p:",
          stats["friedman_decision_agreement"]["p_value"])
    print("Agree mean ranks:", [round(x, 2) for x in
                                stats["friedman_decision_agreement"]["mean_ranks"]])


if __name__ == "__main__":
    main()
