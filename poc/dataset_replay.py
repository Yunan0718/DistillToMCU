"""
通用数据集 teacher-replay（v10.6 扩展）
======================================
对任意已注册数据集跑 teacher 一致性 + 全方法 precision/recall/agreement。
复用 uci_extended_replay 的 evaluate 逻辑，通过 --label 指定数据集。

Usage:
    python dataset_replay.py --label sml2010 --n 60 --seed 42
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uci_extended_replay as uer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--consistency-file", default=None,
                    help="reuse an existing consistency JSON instead of re-running LLM calls")
    args = ap.parse_args()

    uer.LABEL = args.label
    tag = args.tag or f"{args.n}_seed{args.seed}"
    if args.consistency_file:
        import json as _json
        payload = _json.load(open(args.consistency_file, encoding="utf-8"))
        details = payload["details"]
    else:
        details = uer.run_uci_consistency(args.n, seed=args.seed, tag=tag)
    res = uer.evaluate_uci(details, tag=tag)

    print("\n" + "=" * 70)
    print(f"{args.label} teacher-replay ({res['n_snapshots']} snapshots)  "
          f"self-agreement {res['teacher_self_agreement_pct']}%")
    print("=" * 70)
    for name, r in res["methods"].items():
        print(uer._fmt_method(name, r))


if __name__ == "__main__":
    main()
