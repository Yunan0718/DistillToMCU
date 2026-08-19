"""
把新数据集的 teacher-replay 结果（_fixed 文件）合并进主表
teacher_replay_results.json，供 merge_statistics.py 统一做统计。
"""

import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

NEW = {
    "sml2010": "sml2010_extended_replay_results_60_seed42_fixed.json",
    "steel": "steel_extended_replay_results_60_seed42_fixed.json",
    "airquality": "airquality_extended_replay_results_60_seed42_fixed.json",
}


def main():
    p = os.path.join(OUT, "teacher_replay_results.json")
    tr = json.load(open(p, encoding="utf-8"))
    for label, fn in NEW.items():
        fp = os.path.join(OUT, fn)
        if not os.path.exists(fp):
            print(f"[SKIP] {fn} not found")
            continue
        d = json.load(open(fp, encoding="utf-8"))
        tr[label] = {
            "exp_dir": d["exp_dir"],
            "n_snapshots": d["n_snapshots"],
            "teacher_self_agreement_pct": d["teacher_self_agreement_pct"],
            "teacher_self_agreement_n": d["teacher_self_agreement_n"],
            "methods": d["methods"],
            "fidelity_ours_warm_to_deepseek_pct":
                d["methods"]["Ours (warm rules)"]["agree_pct"],
            "fidelity_efficiency_ours_warm": round(
                d["methods"]["Ours (warm rules)"]["agree_pct"]
                / max(0.1, d["teacher_self_agreement_pct"]) * 100, 1),
        }
        print(f"[merge] {label}: n={d['n_snapshots']} "
              f"ceiling={d['teacher_self_agreement_pct']}%")
    json.dump(tr, open(p, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    print(f"Saved: {p} ({len(tr)} datasets)")


if __name__ == "__main__":
    main()
