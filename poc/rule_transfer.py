"""
DistillToMCU — D7 (v10.6): rule transfer summary from teacher-replay
====================================================================
Thin reader over teacher_replay_results.json (produced by the 2160-call
teacher-replay experiment). All numbers are paired on the SAME 60 snapshots
per dataset, under the run's original prompt:

  - fidelity:  DeepSeek-distilled rules vs DeepSeek majority decision
  - transfer:  DeepSeek-distilled rules vs Qwen decision
  - reference: model-model agreement (DeepSeek majority vs Qwen) on the same
               snapshots

This replaces the v10.5f rule_transfer.py whose seed-pairing was misaligned
and whose first-match-in-list resolution produced the uninterpretable
"seed777 agree=1.2%" result.

Output: output/rule_transfer.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    p = os.path.join(OUT, "teacher_replay_results.json")
    if not os.path.exists(p):
        print(f"[SKIP] {p} not found — run llm_consistency_experiment.py "
              f"then teacher_replay.py first.")
        return
    tr = json.load(open(p, encoding="utf-8"))
    out = {}
    for label, d in tr.items():
        out[label] = {
            "n_snapshots": d.get("n_snapshots"),
            "fidelity_to_deepseek_pct": d.get(
                "fidelity_ours_warm_to_deepseek_pct"),
            "transfer_to_qwen_pct": d.get("transfer_ours_warm_to_qwen_pct"),
            "transfer_to_qwen_device_pct": d.get(
                "transfer_ours_warm_to_qwen_device_pct"),
            "model_model_agreement_pct": d.get("model_model_agreement_pct"),
            "model_model_agreement_device_pct": d.get(
                "model_model_agreement_device_pct"),
            "teacher_self_agreement_pct": d.get("teacher_self_agreement_pct"),
            "note": "paired on the same 60 snapshots; original run prompt; "
                    "engine resolution; zero additional API cost.",
        }
        print(f"{label}: fidelity={out[label]['fidelity_to_deepseek_pct']}% "
              f"transfer={out[label]['transfer_to_qwen_pct']}% "
              f"model-model={out[label]['model_model_agreement_pct']}% "
              f"(n={out[label]['n_snapshots']})")
    json.dump(out, open(os.path.join(OUT, "rule_transfer.json"), "w",
                        encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nSaved: {os.path.join(OUT, 'rule_transfer.json')}")


if __name__ == "__main__":
    main()
