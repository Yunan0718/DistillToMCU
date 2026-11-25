#!/usr/bin/env python3
"""
DistillToMCU Phase 0 re-run driver (2026-08-02)
==============================================
Re-runs all currently reproducible Phase 0 experiments with the FIXED code,
then re-runs baselines / statistics / ablations / dashboard data generation.

Why re-run:
  - output/ still contains traces produced before the 2026-08-01 fixes
    (evidence accumulation, date advance, parameterization). Those traces
    are not valid for the paper.
  - UCI V2 (speech translation layer) was deliberately removed from the code
    base (v3 rewrite). It cannot be reproduced; its legacy traces are moved
    to output/legacy_20260802/ and excluded from formal analysis.

Real API cost estimate: ~3,500 DeepSeek calls, roughly 17-20 CNY.
Runtime: about 1.5-3.5 h depending on API latency.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import re


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POC = os.path.join(ROOT, "poc")
OUTPUT = os.path.join(POC, "output")
LEGACY = os.path.join(OUTPUT, "legacy_20260802")
LOGS = os.path.join(OUTPUT, "rerun_logs_20260802")
SUMMARY = os.path.join(OUTPUT, "rerun_20260802_summary.json")
PY = sys.executable
DRIVER_LOG = os.path.join(LOGS, "driver.log")

CANONICAL_DIRS = [
    "seed42",
    "run_seed123",
    "run_seed999",
    "strands_seed42",
    "uci_seed42",
    "uci_v3_seed42",
]

EXPERIMENTS = [
    ("experiment.py",        ["--real", "--seed", "42",  "--days", "30", "--output-dir", os.path.join(OUTPUT, "seed42")],        "synthetic_seed42"),
    ("experiment.py",        ["--real", "--seed", "123", "--days", "30", "--output-dir", os.path.join(OUTPUT, "run_seed123")],   "synthetic_seed123"),
    ("experiment.py",        ["--real", "--seed", "999", "--days", "30", "--output-dir", os.path.join(OUTPUT, "run_seed999")],   "synthetic_seed999"),
    ("experiment_strands.py", ["--seed", "42", "--days", "30", "--output-dir", os.path.join(OUTPUT, "strands_seed42")],         "strands_aruba1"),
    ("experiment_uci.py",    ["--seed", "42", "--days", "30", "--output-dir", os.path.join(OUTPUT, "uci_v3_seed42")],           "uci_v3_real_sensors"),
]

ANALYSIS_STEPS = [
    ("run_full_analysis.py", [], "baselines_statistics"),
    ("ablation_runner.py",   ["--data-dir", os.path.join(OUTPUT, "seed42")], "ablations"),
    ("generate_dashboard_data.py", [], "dashboard_data"),
]


def log(msg):
    line = time.strftime("[%H:%M:%S] ") + str(msg)
    print(line, flush=True)
    try:
        os.makedirs(LOGS, exist_ok=True)
        with open(DRIVER_LOG, "a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
    except Exception:
        pass


def archive_old():
    """Move stale canonical experiment dirs to legacy_20260802/ (recoverable)."""
    os.makedirs(LEGACY, exist_ok=True)
    for name in CANONICAL_DIRS:
        src = os.path.join(OUTPUT, name)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(LEGACY, name)
        if os.path.exists(dst):
            dst = os.path.join(LEGACY, name + "_" + time.strftime("%H%M%S"))
        log(f"[archive] {src} -> {dst}")
        shutil.move(src, dst)


def run_step(script, args, label):
    cmd = [PY, os.path.join(POC, script)] + [str(a) for a in args]
    os.makedirs(LOGS, exist_ok=True)
    log_file = os.path.join(LOGS, label + ".log")
    log(f"[run] {label}: {' '.join(cmd)}")
    t0 = time.time()
    with open(log_file, "w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.run(cmd, cwd=POC, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    log(f"[done] {label}: exit={p.returncode}, {dt/60:.1f} min, log={log_file}")
    if p.returncode != 0:
        log(f"[ERROR] {label} failed with exit code {p.returncode}")
        raise RuntimeError(f"{label} failed")
    return log_file


def parse_metrics(exp_dir):
    path = os.path.join(OUTPUT, exp_dir, "metrics.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    ar = rows[-1].get("autonomy_rate") if rows else None
    # metrics.jsonl 中 cloud_calls / local_calls 是累计值，取最后一行
    cloud = rows[-1].get("cloud_calls", 0) if rows else 0
    local = rows[-1].get("local_calls", 0) if rows else 0
    return {"days": len(rows), "final_ar": ar,
            "cloud_calls": cloud, "local_calls": local}


def write_summary(timings):
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Fixed-code Phase 0 re-run. UCI V2 excluded (code removed).",
        "experiments": {},
        "timings_min": timings,
    }
    for script, args, label in EXPERIMENTS:
        odir = args[-1] if "--output-dir" in args else None
        exp_key = os.path.basename(odir) if odir else label
        m = parse_metrics(exp_key)
        trace_path = os.path.join(OUTPUT, exp_key, "traces.jsonl")
        n_traces = 0
        if os.path.exists(trace_path):
            with open(trace_path, "r", encoding="utf-8") as f:
                n_traces = sum(1 for _ in f)
        m["traces"] = n_traces
        log_file = os.path.join(LOGS, label + ".log")
        m["est_cost_rmb"] = parse_est_cost(log_file)
        summary["experiments"][label] = m
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"[summary] written to {SUMMARY}")
    return summary


def parse_est_cost(log_file):
    """Parse the 'Est. API Cost: ~X CNY' line from an experiment log."""
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        m = re.search(r"Est\. API Cost:\s*[~¥]?\s*([\d.]+)", text)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def main():
    timings = {}
    try:
        archive_old()
        for script, args, label in EXPERIMENTS:
            t0 = time.time()
            run_step(script, args, label)
            timings[label] = round((time.time() - t0) / 60, 1)
        for script, args, label in ANALYSIS_STEPS:
            t0 = time.time()
            run_step(script, args, label)
            timings[label] = round((time.time() - t0) / 60, 1)
    except Exception as e:
        log(f"[FATAL] {e}")
        write_summary(timings)
        sys.exit(1)
    summary = write_summary(timings)
    log("[OK] Phase 0 re-run complete.")
    for k, v in summary["experiments"].items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
