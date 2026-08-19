"""
DistillToMCU — D2 (v10.6): AGREE reference frame with n=60 + bootstrap CI
=========================================================================
Ours decision-agreement vs the teacher LLM's own self-agreement ceiling,
per dataset, both measured on the SAME 60 teacher-replay snapshots:

  - ours_agree  = P(rule action == teacher majority | teacher acted)
  - ceiling     = P(teacher's 3 T=0 repeats agree) over all snapshots
  - efficiency  = ours_agree / ceiling

Bootstrap 95% CI over the 60 snapshots for ours_agree and ceiling.
Covers all 6 datasets (adds seed123/seed999 that were missing in v10.5f).

Output: output/agree_reference.json
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def bootstrap_mean(vals, n_boot=10000, seed=42):
    rng = random.Random(seed)
    if not vals:
        return None, None
    means = [sum(rng.choice(vals) for _ in vals) / len(vals)
             for _ in range(n_boot)]
    means.sort()
    return (round(means[int(0.025 * n_boot)], 1),
            round(means[int(0.975 * n_boot)], 1))


def main():
    p = os.path.join(OUT, "teacher_replay_results.json")
    if not os.path.exists(p):
        print(f"[SKIP] {p} not found.")
        return
    tr = json.load(open(p, encoding="utf-8"))
    rows = {}
    for label, d in tr.items():
        snaps = d.get("snapshots", [])
        agree_flags, ceil_flags = [], []
        for s in snaps:
            ceil_flags.append(1 if s.get("self_consistent") else 0)
            majority = s.get("majority")
            if majority is not None:
                # 与 teacher_replay 的 agree 口径一致：教师有动作才计入
                agree_flags.append(1 if s.get("_ours_warm_match") else 0)
        ours = round(sum(agree_flags) / max(1, len(agree_flags)) * 100, 1)
        ceiling = round(sum(ceil_flags) / max(1, len(ceil_flags)) * 100, 1)
        ours_ci = bootstrap_mean(agree_flags)
        ceil_ci = bootstrap_mean(ceil_flags)
        rows[label] = {
            "ours_agree": ours,
            "ours_agree_ci95": ours_ci,
            "llm_self_agree": ceiling,
            "llm_self_agree_ci95": ceil_ci,
            "fidelity_efficiency": round(ours / ceiling * 100, 1)
            if ceiling else None,
            "n_action_snapshots": len(agree_flags),
            "n_snapshots": len(snaps),
        }
        print(f"{label}: ours={ours}% CI={ours_ci} | ceiling={ceiling}% "
              f"CI={ceil_ci} | efficiency={rows[label]['fidelity_efficiency']}%")

    json.dump({
        "note": "Ours AGREE vs LLM self-agreement ceiling, paired on the same "
                "60 snapshots per dataset under the original run prompt; "
                "bootstrap 95% CI (n=60).",
        "per_dataset": rows,
    }, open(os.path.join(OUT, "agree_reference.json"), "w", encoding="utf-8"),
        indent=1, ensure_ascii=False)
    print(f"\nSaved: {os.path.join(OUT, 'agree_reference.json')}")


if __name__ == "__main__":
    main()
