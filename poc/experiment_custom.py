"""
DistillToMCU — 自定义数据源实验 (v7.0)
======================================
读取由 H5 列映射生成的标准快照 JSONL（每行 {"sensors":{...},"user_input":"..."}），
复用 UCIExecutor 的"传感器直发 LLM + 规则蒸馏"闭环，输出与其它实验一致的
metrics.jsonl / rules_snapshot.json / traces.jsonl。

Usage:
    python experiment_custom.py --data snapshots.jsonl --seed 42 --days 30 --output-dir output/exp_x
"""

import json
import os
import random
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from trace_store import TraceStore
from rule_engine import RuleEngine
from distiller import Distiller
import llm_client
from experiment_uci import UCIExecutor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="快照 JSONL 路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    random.seed(args.seed)

    snaps = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    if not snaps:
        print("ERROR: 快照为空")
        sys.exit(1)

    odir = args.output_dir or os.path.join(os.path.dirname(__file__), "output", "custom")
    os.makedirs(odir, exist_ok=True)

    ts = TraceStore(output_dir=odir)
    re = RuleEngine()
    di = Distiller(re, llm_client=llm_client)
    ex = UCIExecutor(re, ts)

    per_day = max(1, len(snaps) // args.days)
    by_day = [snaps[i * per_day:(i + 1) * per_day] for i in range(args.days)]
    print(f"Custom dataset: {len(snaps)} snapshots, {len(by_day[0])}/day x {args.days}d")

    daily_m = []
    t0 = time.time()
    for d in range(args.days):
        dt0 = time.time()
        for s in by_day[d]:
            sensors = s.get("sensors", {})
            ui = s.get("user_input", "")
            # UCIExecutor.handle(sensors) 直接发 LLM；带 user_input 时走合成交互
            if ui:
                sensors = dict(sensors)
                sensors["user_input"] = ui
            ex.handle(sensors)
        re.update_all_freshness()
        re.gc()
        nr, _ = di.distill(ts.traces)
        sm = ex.summary()
        ar = sm["ar"]
        daily_m.append({
            "day": d + 1, "autonomy_rate": ar,
            "cloud_calls": sm["cloud"], "local_calls": sm["local"],
            "total": sm["total"],
            "active_rules": re.stats().get("active_count", 0),
            "total_rules": re.stats()["total"], "new_rules_today": nr,
            "avg_local_lat_ms": sm["loc_ms"], "avg_cloud_lat_ms": sm["cld_ms"],
        })
        print(f"  Day {d+1:2d}  [+{nr}r] AR={ar:5.1f}%  ({time.time()-dt0:.0f}s)")

    fin = ex.summary()
    print(f"\n  FINAL: AR={fin['ar']:.1f}% | {fin['total']} int | "
          f"Local:{fin['local']} Cloud:{fin['cloud']} | "
          f"Cost:~{fin['cost']:.2f} CNY | Time:{time.time()-t0:.0f}s")

    with open(os.path.join(odir, "metrics.jsonl"), "w", encoding="utf-8") as f:
        for m in daily_m:
            f.write(json.dumps(m) + "\n")
    re.save_snapshot(os.path.join(odir, "rules_snapshot.json"))
    with open(os.path.join(odir, "traces.jsonl"), "w", encoding="utf-8") as f:
        for t in ts.traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
