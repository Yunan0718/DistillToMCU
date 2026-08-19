"""
DistillToMCU — same-data cross-model online runs (v10.6)
=========================================================
After the generator fix (interaction schedule pre-generated before the loop),
DeepSeek and Qwen online runs share bit-identical sensor/user_input sequences.
This script verifies alignment and summarizes the AR comparison per dataset.

Pairs:
  - seed42:  xrun_ds_seed42  vs xrun_qwen_seed42
  - seed777: xrun_ds_seed777 vs xrun_qwen_seed777
  - uci_v3:  run4b_uci_seed42 vs qwen_uci_v3 (data already aligned, 600/600)

Output: output/cross_model_same_data.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

PAIRS = [
    ("seed42", "xrun_ds_seed42", "xrun_qwen_seed42"),
    ("seed777", "xrun_ds_seed777", "xrun_qwen_seed777"),
    ("uci_v3", "run4b_uci_seed42", "qwen_uci_v3"),
]


def load_traces(d):
    p = os.path.join(OUT, d, "traces.jsonl")
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def final_ar(d):
    p = os.path.join(OUT, d, "metrics.jsonl")
    ar = None
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ar = json.loads(line).get("autonomy_rate")
    return ar


def main():
    out = {}
    for label, ds_dir, qw_dir in PAIRS:
        if not os.path.exists(os.path.join(OUT, ds_dir, "metrics.jsonl")) or \
           not os.path.exists(os.path.join(OUT, qw_dir, "metrics.jsonl")):
            print(f"[SKIP] {label} (runs not finished)")
            continue
        a, b = load_traces(ds_dir), load_traces(qw_dir)
        n = min(len(a), len(b))
        sensor_aligned = sum(1 for i in range(n)
                             if a[i].get("sensors") == b[i].get("sensors"))
        full_aligned = sum(1 for i in range(n)
                           if a[i].get("sensors") == b[i].get("sensors")
                           and a[i].get("user_input") == b[i].get("user_input"))
        ar_ds, ar_qw = final_ar(ds_dir), final_ar(qw_dir)
        out[label] = {
            "deepseek_dir": ds_dir,
            "qwen_dir": qw_dir,
            "deepseek_ar": ar_ds,
            "qwen_ar": ar_qw,
            "ar_delta_pp": round(ar_qw - ar_ds, 1),
            "traces_ds": len(a),
            "traces_qwen": len(b),
            "sensor_aligned": sensor_aligned,
            "sensor_plus_user_aligned": full_aligned,
            "aligned_total": n,
            "note": "generator fix (v10.6): interaction schedule pre-generated "
                    "before the LLM loop -> same seed = identical data across "
                    "backends (synthetic); UCI data is fixed/chronological so "
                    "sensors are identical while the older Qwen run used a "
                    "slightly different query formatting. AR delta isolates "
                    "the teacher-model effect.",
        }
        print(f"{label}: DS={ar_ds}% Qwen={ar_qw}% (delta {out[label]['ar_delta_pp']:+.1f}pp) "
              f"sensor-align {sensor_aligned}/{n} full {full_aligned}/{n}")
    json.dump(out, open(os.path.join(OUT, "cross_model_same_data.json"), "w",
                        encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nSaved: {os.path.join(OUT, 'cross_model_same_data.json')}")


if __name__ == "__main__":
    main()
