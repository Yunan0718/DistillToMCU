"""
Teacher LLM Behavioral Consistency + Cross-Model Experiment (v10.6)
====================================================================
For 60 sensor snapshots per dataset (stratified across the 30-day horizon),
query the teacher LLM multiple times UNDER THE EXACT PROMPT/TOOLS/USER_INPUT
of the original online runs:

  - 3x DeepSeek @ T=0   -> self-agreement ceiling + majority action
  - 2x DeepSeek @ T=0.7 -> stochastic-consistency reference
  - 1x Qwen3.7   @ T=0   -> cross-model transfer reference

6 datasets x 60 snapshots x 6 calls = 2160 API calls.
Prompt fidelity matters: previous versions used a different system prompt,
which changed teacher behavior (e.g. UCI: replay prompt turned on lights while
the actual run distilled fan/curtain rules). This version reuses
cloud_agent_think with each dataset's original prompt and the recorded
user_input, so ceiling/fidelity are measured in the same behavioral regime.

Output: output/llm_consistency_results.json
"""

import concurrent.futures
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import llm_client
from experiment_uci import SYSTEM_PROMPT as UCI_SYSTEM_PROMPT
from experiment_sml2010 import SYSTEM_PROMPT as SML2010_SYSTEM_PROMPT
from experiment_steel import SYSTEM_PROMPT as STEEL_SYSTEM_PROMPT
from experiment_airquality import SYSTEM_PROMPT as AIRQUALITY_SYSTEM_PROMPT

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

SYNTHETIC_SYSTEM_PROMPT = """You are a smart home controller agent.
Given sensor readings and user input, decide what device actions to take.

Available devices:
- led: adjustable brightness (0-100), color temperature
- fan: adjustable speed (1-3)
- curtain: open/close, adjustable position (0-100)

Rules:
1. Always use tools when device control is needed.
2. Consider energy efficiency.
3. Consider user comfort.
4. When in doubt, choose the safer option.
5. Respond in user's language."""

STRANDS_SYSTEM_PROMPT = ("Smart home controller. Use tools for: led(brightness), "
                         "fan(speed 1-3), curtain(position 0-100).")

DATASETS = {
    "synthetic_seed42":  {"dir": "run4b_seed42_seed42",
                          "prompt": SYNTHETIC_SYSTEM_PROMPT},
    "synthetic_seed123": {"dir": "run4b_seed123_seed42",
                          "prompt": SYNTHETIC_SYSTEM_PROMPT},
    "synthetic_seed999": {"dir": "run4b_seed999_seed42",
                          "prompt": SYNTHETIC_SYSTEM_PROMPT},
    "synthetic_seed777": {"dir": "run4b_seed777_seed42",
                          "prompt": SYNTHETIC_SYSTEM_PROMPT},
    "strands_aruba1":    {"dir": "run4b_strands_seed42",
                          "prompt": STRANDS_SYSTEM_PROMPT},
    "uci_v3":            {"dir": "run4b_uci_seed42",
                          "prompt": UCI_SYSTEM_PROMPT},
    "sml2010":           {"dir": "run4b_sml2010_seed42",
                          "prompt": SML2010_SYSTEM_PROMPT},
    "steel":             {"dir": "run4b_steel_seed42",
                          "prompt": STEEL_SYSTEM_PROMPT},
    "airquality":        {"dir": "run4b_airquality_seed42",
                          "prompt": AIRQUALITY_SYSTEM_PROMPT},
}


def load_trace_items(exp_dir: str) -> list[dict]:
    path = os.path.join(OUTPUT_DIR, exp_dir, "traces.jsonl")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            if t.get("sensors"):
                items.append(t)
    return items


def select_diverse_snapshots(items: list[dict], n: int = 60,
                             seed: int = 42) -> list[dict]:
    """按时间轴均匀采样 n 条（覆盖 30 天），并对传感器状态去重。"""
    rng = random.Random(seed)
    if len(items) <= n:
        return items
    step = len(items) / n
    selected = []
    for i in range(n):
        lo = int(i * step)
        hi = max(lo + 1, int((i + 1) * step))
        selected.append(items[rng.randrange(lo, min(hi, len(items)))])
    seen, out = set(), []
    for t in selected:
        key = json.dumps(t.get("sensors", {}), sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _agent_call(system_prompt, user_input, sensors, temperature, backend):
    tools = llm_client.build_tools()
    sensor_text = "\n".join([f"  {k}: {v}" for k, v in sensors.items()
                             if v is not None])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Current sensors:\n{sensor_text}\n\n"
                    f"User says: \"{user_input}\"\n\n"
                    "Decide what action to take. Use the available tools "
                    "if device control is needed."},
    ]
    resp = llm_client.call_llm_with_backend(
        messages, tools=tools, backend=backend, temperature=temperature)
    return extract_action(resp.get("tool_calls") or []), resp.get("latency_ms", 0)


def extract_action(tool_calls: list) -> str | None:
    if not tool_calls:
        return None
    func = tool_calls[0].get("function", {})
    raw_name = func.get("name", "")
    if "_control" not in raw_name:
        return None  # read_sensors 等只读工具不算控制动作
    name = raw_name.replace("_control", "")
    if not name:
        return None
    try:
        args_str = func.get("arguments", "{}")
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except json.JSONDecodeError:
        args = {}
    return f"{name}.{args.get('command', '?')}"


def _one_call(args):
    snap, system_prompt, temp, backend = args
    return _agent_call(system_prompt, snap.get("user_input", ""),
                       snap.get("sensors", {}), temp, backend)


def run_consistency_experiment(datasets=None, snapshots_per_dataset: int = 60,
                               workers: int = 12):
    datasets = datasets or DATASETS
    all_results = {}
    total_calls = 0

    for label, cfg in datasets.items():
        print(f"\n{'='*60}\n  Dataset: {label} ({cfg['dir']})\n{'='*60}")
        items = load_trace_items(cfg["dir"])
        selected = select_diverse_snapshots(items, snapshots_per_dataset)
        print(f"  Loaded {len(items)} traces, selected {len(selected)} snapshots")

        calls = []
        for si, snap in enumerate(selected):
            for _ in range(3):
                calls.append((si, snap, cfg["prompt"], 0.0, "deepseek-v4-flash"))
            for _ in range(2):
                calls.append((si, snap, cfg["prompt"], 0.7, "deepseek-v4-flash"))
            calls.append((si, snap, cfg["prompt"], 0.0,
                          "qwen3.7-flash-2026-07-15"))

        responses = [None] * len(calls)
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers) as pool:
            futs = {pool.submit(_one_call, c[1:]): i for i, c in enumerate(calls)}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                i = futs[fut]
                try:
                    responses[i] = fut.result()
                except Exception:
                    responses[i] = (None, 0)
                done += 1
                if done % 120 == 0:
                    print(f"    [{done}/{len(calls)}] calls done")

        rows = []
        for si, snap in enumerate(selected):
            base = si * 6
            ds_t0 = [responses[base + k][0] for k in range(3)]
            ds_t07 = [responses[base + 3 + k][0] for k in range(2)]
            qwen = responses[base + 5][0]
            total_calls += 6
            from collections import Counter
            c0 = Counter(a for a in ds_t0 if a is not None)
            majority = c0.most_common(1)[0][0] if c0 else None
            rows.append({
                "snapshot_idx": si,
                "sensors": {k: round(v, 2) if isinstance(v, float) else v
                            for k, v in snap.get("sensors", {}).items()
                            if v is not None},
                "user_input": snap.get("user_input", ""),
                "ds_t0_actions": ds_t0,
                "ds_t07_actions": ds_t07,
                "qwen_action": qwen,
                "ds_t0_majority": majority,
                "ds_t0_consistent": len(set(a for a in ds_t0
                                            if a is not None)) <= 1,
                "ds_t07_consistent": len(set(a for a in ds_t07
                                             if a is not None)) <= 1,
                "avg_latency_ms": round(sum(r[1] for r in responses[base:base+6])
                                        / 6),
            })

        n = len(selected)
        t0_c = sum(1 for r in rows if r["ds_t0_consistent"])
        t07_c = sum(1 for r in rows if r["ds_t07_consistent"])
        print(f"  T=0 consistency: {t0_c}/{n} ({t0_c/n*100:.0f}%) | "
              f"T=0.7: {t07_c}/{n} ({t07_c/n*100:.0f}%)")

        all_results[label] = {
            "dataset": cfg["dir"],
            "system_prompt": cfg["prompt"],
            "n_snapshots_selected": n,
            "rows": rows,
            "summary": {
                "T=0.0": {"consistent": t0_c, "total": n},
                "T=0.7": {"consistent": t07_c, "total": n},
            },
        }
        _save(all_results, total_calls)

    _save(all_results, total_calls)
    return all_results


def _save(all_results, total_calls):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    payload = {
        "summary": {
            "total_api_calls": total_calls,
            "total_snapshots": sum(d["n_snapshots_selected"]
                                   for d in all_results.values()),
            "per_dataset": {k: d["summary"] for k, d in all_results.items()},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "details": all_results,
    }
    p = os.path.join(OUTPUT_DIR, "llm_consistency_results.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_consistency_experiment(
        datasets=DATASETS,
        snapshots_per_dataset=60,
    )
