"""
DistillToMCU — MCU on-board 4x validation runner
================================================
Runs the physical ESP32-S3 (COM6) against 4 different seed traces per
dataset, measuring execution stability (AR, active rules, SRAM).

Phase 1: UCI rules (already flashed) x UCI seeds 42/123/999/777
Phase 2: seed42 rules (flashed by caller before phase 2) x seed42 seeds

Each run: first 200 traces of the run4b output, 6s injection interval
(~20 min each).

Usage:
    python mcu_run_4x.py --port COM6 --dataset uci_v3
    python mcu_run_4x.py --port COM6 --dataset seed42
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serial_mcu_exp import MCU, load_sensors

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "poc", "output")
SEEDS = [42, 123, 999, 777]


def load_run4b_traces(dataset_key, seed, count=200):
    """Load sensor snapshots from run4b_<key>_seed<seed>/traces.jsonl."""
    # directory template matches run_4x.py: run4b_{key}_seed{seed}
    # with key = seed42/seed123/seed999/seed777/strands/uci
    key = dataset_key.replace("_v3", "")  # uci_v3 -> uci
    exp_dir = f"run4b_{key}_seed{seed}"
    path = os.path.join(OUT, exp_dir, "traces.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line).get("sensors", {})
                if s:
                    out.append(s)
    return out[:count]


def load_dir_traces(traces_dir, count=200):
    """Load sensor snapshots from an arbitrary output/<traces_dir>."""
    path = os.path.join(OUT, traces_dir, "traces.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line).get("sensors", {})
                if s:
                    out.append(s)
    return out[:count]


def run_board(dataset_key, port, count, delay, seeds=None):
    mcu = MCU(port)
    s0 = mcu.stats()
    print(f"Init: rules={s0.get('rules_total')} active={s0.get('rules_active')} "
          f"SRAM={s0.get('free_sram')}")

    results = []
    for seed in (seeds or SEEDS):
        if dataset_key.startswith("dir:"):
            traces_dir = dataset_key[4:]
            sensors = load_dir_traces(traces_dir, count)
            label = traces_dir
        else:
            sensors = load_run4b_traces(dataset_key, seed, count)
            label = f"{dataset_key}_seed{seed}"
        print(f"\n=== {label}: {len(sensors)} samples ===")
        lost = 0
        for i, s in enumerate(sensors):
            ok, status = mcu.inject(s)
            if not ok:
                lost += 1
            time.sleep(max(0.0, delay - 1.5))
            if (i + 1) % 50 == 0:
                st = mcu.stats()
                print(f"  [{i+1:4d}] t={st.get('total')} L={st.get('local')} "
                      f"C={st.get('cloud')} AR={st.get('ar_pct')}% "
                      f"active={st.get('rules_active')} lost={lost}")
        st = mcu.stats()
        results.append({
            "dataset": dataset_key, "seed": seed, "n": len(sensors),
            "final": st, "inject_lost": lost,
        })
        print(f"  FINAL {label}: AR={st.get('ar_pct')}% "
              f"L={st.get('local')}/{st.get('total')} active={st.get('rules_active')}")

    # v10.5f (D3): capture real match latency in the SAME serial session
    # immediately after the run — the board has been observed to reboot
    # shortly after injection stops, wiping telemetry counters.
    lat = mcu.latstats()
    results.append({"dataset": "latency", "latstats": lat})
    mcu.close()
    # v10.5b: dataset_key may be "dir:<name>" — ':' is an NTFS ADS separator
    # on Windows and silently redirects the file into an alternate data
    # stream (data lost from normal listing & git). Sanitize it.
    safe_key = dataset_key.replace(":", "_").replace("dir_", "")
    out_path = os.path.join(OUT, f"mcu_4x_{safe_key}_{int(time.time())}.json")
    json.dump(results, open(out_path, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--dataset", default="uci_v3", choices=["uci_v3", "seed42"])
    ap.add_argument("--traces-dir", default=None,
                    help="Load traces from poc/output/<dir> instead of run4b "
                         "(e.g. --traces-dir qwen_seed42)")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--delay", type=float, default=6.0)
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 999")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    ds = a.dataset
    if a.traces_dir:
        ds = "dir:" + a.traces_dir
        seeds = [None]  # single pass over that directory
    run_board(ds, a.port, a.count, a.delay, seeds=seeds)
