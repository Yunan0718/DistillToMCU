"""
DistillToMCU v10.2 — 4x Repeat Experiment Runner
================================================
Run all 6 datasets x 4 fixed seeds (42/123/999/777) with the CURRENT
(v10.2: pruning + discriminative filtering) algorithm, 4-way parallel.

Why 4x:
  - Each dataset gets n=4 independent runs -> mean/std, bootstrap CI,
    cross-seed stability (addresses the old 58.9% -> 13.4% repro concern)
  - All runs use fixed seeds for reproducibility
  - Old output dirs (seed42/, run_seed123/, ...) are preserved as legacy;
    new runs go to run4_* directories.

Usage:
    python run_4x.py --workers 4            # full 24-run sweep
    python run_4x.py --smoke                 # 2 runs only (validation)
    python run_4x.py --datasets seed42,uci_v3  # subset by dataset key
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POC = os.path.join(ROOT, "poc")
OUT = os.path.join(POC, "output")
LOG_DIR = os.path.join(OUT, "rerun_logs_4x")
PY = r"C:\Espressif\tools\python\v5.2.6\venv\Scripts\python.exe"

# dataset key -> (script, output-dir template, needs --real flag)
DATASETS = {
    # FIX v4b: output dir must encode BOTH dataset key and repeat seed,
    # otherwise synthetic datasets overwrite each other's traces/rules.
    "seed42":   ("experiment.py",        "run4b_seed42_seed{seed}",  True),
    "seed123":  ("experiment.py",        "run4b_seed123_seed{seed}", True),
    "seed999":  ("experiment.py",        "run4b_seed999_seed{seed}", True),
    "seed777":  ("experiment.py",        "run4b_seed777_seed{seed}", True),
    "strands":  ("experiment_strands.py","run4b_strands_seed{seed}", False),
    "uci_v3":   ("experiment_uci.py",    "run4b_uci_seed{seed}",    False),
    "sml2010":  ("experiment_sml2010.py","run4b_sml2010_seed{seed}", False),
    "steel":    ("experiment_steel.py",   "run4b_steel_seed{seed}",   False),
    "airquality":("experiment_airquality.py","run4b_airquality_seed{seed}", False),
}

SEEDS = [42, 123, 999, 777]


def build_tasks(datasets=None):
    tasks = []
    keys = datasets or list(DATASETS)
    for key in keys:
        script, odir_tmpl, real = DATASETS[key]
        for seed in SEEDS:
            tasks.append({
                "key": key,
                "seed": seed,
                "script": os.path.join(POC, script),
                "out": os.path.join(OUT, odir_tmpl.format(seed=seed)),
                "real": real,
            })
    return tasks


def run_one(task):
    os.makedirs(LOG_DIR, exist_ok=True)
    label = f"{task['key']}_seed{task['seed']}"
    log_path = os.path.join(LOG_DIR, f"{label}.log")
    cmd = [PY, task["script"], "--seed", str(task["seed"]),
           "--days", "30", "--output-dir", task["out"]]
    if task["real"]:
        cmd.append("--real")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=POC)
    dt = time.time() - t0
    return {"task": task, "exit": proc.returncode,
            "seconds": round(dt, 1), "log": log_path}


def final_ar(out_dir):
    p = os.path.join(out_dir, "metrics.jsonl")
    if not os.path.exists(p):
        return None
    ar = None
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ar = json.loads(line).get("autonomy_rate")
    except Exception:
        return None
    return ar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datasets", default=None,
                    help="comma-separated subset, e.g. seed42,uci_v3")
    ap.add_argument("--smoke", action="store_true",
                    help="run 2 tasks only (validation)")
    args = ap.parse_args()

    tasks = build_tasks(
        [d.strip() for d in args.datasets.split(",")] if args.datasets else None)
    if args.smoke:
        tasks = tasks[:2]
    print(f"Tasks: {len(tasks)} | workers: {args.workers}")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            done += 1
            ar = final_ar(r["task"]["out"])
            status = "OK" if r["exit"] == 0 else f"FAIL({r['exit']})"
            print(f"[{done}/{len(tasks)}] {r['task']['key']} "
                  f"seed{r['task']['seed']} {status} AR={ar}% "
                  f"{r['seconds']:.0f}s")
            results.append({**r, "final_ar": ar})

    summary_path = os.path.join(LOG_DIR, "summary.json")
    json.dump(results, open(summary_path, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"\nSaved: {summary_path}")
    fails = [r for r in results if r["exit"] != 0]
    print(f"OK={len(results)-len(fails)} FAIL={len(fails)}")


if __name__ == "__main__":
    main()
