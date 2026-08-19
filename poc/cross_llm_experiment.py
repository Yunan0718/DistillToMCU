"""
DistillToMCU v8 — Cross-LLM Validation Experiment
=================================================
将同一批传感器快照分别发给 DeepSeek V4 Flash 和 GPT-4o-mini，
比较两者的 tool_call 决策一致性，验证规则蒸馏是 LLM-agnostic 的。

指标:
  - Decision Agreement: 两个 LLM 对同一传感器状态给出相同 (device, command) 的比例
  - Tool Call Rate: 各自的 tool_call 触发率
  - Action Distribution: device/command 分布对比
  - Per-sample comparison: 逐条对比，标注一致/不一致/单方无动作

输出: output/cross_llm_results.json

Usage:
    # 需要两个 API Key:
    export DEEPSEEK_API_KEY=sk-xxx
    export OPENAI_API_KEY=sk-xxx

    python cross_llm_experiment.py --max-samples 100
    python cross_llm_experiment.py --dry-run  # 先跑 5 条测试
"""

import json
import os
import sys
import time
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from llm_client import call_llm_with_backend

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# v10.5: per-dataset trace source (run4b = v10.2 full 30-day runs)
DATASET_TRACES = {
    "seed42":  "run4b_seed42_seed42",
    "seed777": "run4b_seed777_seed42",
    "uci_v3":  "run4b_uci_seed42",
}

SYSTEM_PROMPT = """You are a smart home controller. Given sensor readings,
decide what device actions to take.

Available devices:
- led: on/off, brightness (0-100)
- fan: on/off, speed (1-3)
- curtain: on/off/set, position (0-100)

Rules:
- If it's dark, turn on the light
- If it's too hot, turn on the fan
- If the sensor values are comfortable, do nothing
- Respond with a tool call ONLY when action is needed"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "led_control",
            "description": "Control LED light",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "enum": ["on", "off"]},
                    "brightness": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fan_control",
            "description": "Control fan",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "enum": ["on", "off"]},
                    "speed": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "curtain_control",
            "description": "Control curtain",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "enum": ["on", "off", "set"]},
                    "position": {"type": "integer"},
                },
            },
        },
    },
]


def load_traces(exp_dir: str = "run4b_seed42_seed42") -> list:
    path = os.path.join(OUTPUT_DIR, exp_dir, "traces.jsonl")
    if not os.path.exists(path):
        path = os.path.join(OUTPUT_DIR, "traces.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_action(tool_calls: list) -> tuple | None:
    """Extract (device, command) from tool_calls. Returns None if no action."""
    if not tool_calls:
        return None
    tc = tool_calls[0]
    func = tc.get("function", {})
    name = func.get("name", "").replace("_control", "")
    if not name:
        return None
    try:
        args = json.loads(func.get("arguments", "{}")) if isinstance(
            func.get("arguments"), str) else func.get("arguments", {})
    except json.JSONDecodeError:
        args = {}
    return name, args.get("command", "on")


def query_llm(sensors: dict, backend: str, temperature: float = 0.0) -> dict:
    """Send sensors to LLM, return raw response."""
    sensor_str = json.dumps(sensors, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Current sensor readings:\n{sensor_str}\n\nDecide what action to take."},
    ]
    result = call_llm_with_backend(
        messages, tools=TOOLS, backend=backend, temperature=temperature)
    return result


def run_cross_llm(max_samples: int = 100, dry_run: bool = False, seed: int = 42,
                  dataset: str = "seed42",
                  backend_a: str = "deepseek-v4-flash",
                  backend_b: str = "qwen3.7-flash-2026-07-15"):
    """Main Cross-LLM experiment."""
    import random
    random.seed(seed)

    # Load traces
    traces = load_traces(DATASET_TRACES.get(dataset, DATASET_TRACES["seed42"]))
    cloud_traces = [t for t in traces
                    if t.get("execution", {}).get("mode") == "cloud"
                    and t.get("llm_response", {}).get("tool_calls")]
    print(f"[{dataset}] Loaded {len(traces)} traces, "
          f"{len(cloud_traces)} with tool_calls")

    # Select diverse sensor snapshots
    # Sort by temperature range to get diversity
    cloud_traces.sort(key=lambda t: t.get("sensors", {}).get("temperature", 25))
    # Sample evenly
    n = min(max_samples, len(cloud_traces))
    if dry_run:
        n = min(5, n)
    step = max(1, len(cloud_traces) // n)
    sampled = cloud_traces[::step][:n]

    print(f"\nRunning Cross-LLM ({backend_a} vs {backend_b}) on {n} samples...")
    if dry_run:
        print("  (DRY RUN mode)")

    results = []
    agree_count = 0
    disagree_count = 0
    both_no_action = 0
    only_deepseek = 0
    only_gpt = 0
    total_cost_cny = 0.0

    for i, trace in enumerate(sampled):
        sensors = trace.get("sensors", {})
        original_action = extract_action(
            trace.get("llm_response", {}).get("tool_calls", []))

        ds_result = query_llm(sensors, backend_a)
        ds_action = extract_action(ds_result.get("tool_calls") or [])

        gpt_result = query_llm(sensors, backend_b)
        gpt_action = extract_action(gpt_result.get("tool_calls") or [])

        # Compare
        if ds_action == gpt_action:
            if ds_action is None:
                both_no_action += 1
                status = "both_no_action"
            else:
                agree_count += 1
                status = "agree"
        elif ds_action is None and gpt_action is not None:
            only_gpt += 1
            status = "only_gpt"
        elif ds_action is not None and gpt_action is None:
            only_deepseek += 1
            status = "only_deepseek"
        else:
            disagree_count += 1
            status = "disagree"

        # Cost estimate
        total_cost_cny += 0.0007 * 2  # 2 calls per sample

        results.append({
            "sensors": {k: round(v, 2) if isinstance(v, float) else v
                       for k, v in sensors.items()},
            "original_action": list(original_action) if original_action else None,
            "deepseek": {
                "action": list(ds_action) if ds_action else None,
                "latency_ms": ds_result.get("latency_ms", 0),
                "model": ds_result.get("model", "deepseek-v4-flash"),
            },
            "backend_b": {
                "action": list(gpt_action) if gpt_action else None,
                "latency_ms": gpt_result.get("latency_ms", 0),
                "model": gpt_result.get("model", backend_b),
            },
            "status": status,
        })

        if (i + 1) % 10 == 0 or dry_run:
            pct = (i + 1) / n * 100
            print(f"  [{i+1}/{n}] {pct:.0f}%  "
                  f"agree={agree_count} disagree={disagree_count} "
                  f"ds_only={only_deepseek} gpt_only={only_gpt} "
                  f"both_none={both_no_action}")

        # Rate limit: avoid hitting API too fast
        if not dry_run:
            time.sleep(0.3)

    # Summary statistics
    total = n
    agreement_pct = round((agree_count + both_no_action) / total * 100, 1)
    action_agreement_pct = round(
        agree_count / max(1, agree_count + disagree_count + only_deepseek + only_gpt) * 100, 1)

    summary = {
        "n_samples": total,
        "agree": agree_count,
        "disagree": disagree_count,
        "both_no_action": both_no_action,
        "only_deepseek": only_deepseek,
        "only_backend_b": only_gpt,
        "agreement_pct": agreement_pct,
        "action_agreement_pct": action_agreement_pct,
        "est_cost_cny": round(total_cost_cny, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\n" + "=" * 60)
    print("  Cross-LLM Validation Results")
    print("=" * 60)
    print(f"  Samples:              {total}")
    print(f"  Agree (same action):  {agree_count}")
    print(f"  Disagree (different): {disagree_count}")
    print(f"  Both no-action:       {both_no_action}")
    print(f"  Only DeepSeek acts:   {only_deepseek}")
    print(f"  Only {backend_b} acts:    {only_gpt}")
    print(f"  Overall agreement:    {agreement_pct}%")
    print(f"  Action agreement:     {action_agreement_pct}%")
    print(f"  Est. API cost:        ¥{total_cost_cny:.4f}")

    # Interpretation
    print(f"\n  Interpretation:")
    if agree_count + both_no_action >= total * 0.7:
        print(f"  ✅ High agreement ({agreement_pct}%) → rules transfer across LLMs")
    elif agree_count + both_no_action >= total * 0.5:
        print(f"  ⚠️  Moderate agreement ({agreement_pct}%) → partial transfer")
    else:
        print(f"  ❌ Low agreement ({agreement_pct}%) → LLM-dependent")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR,
                            f"cross_llm_{dataset}_{backend_b}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "backend_a": backend_a,
                   "backend_b": backend_b, "dataset": dataset,
                   "details": results}, f,
                  indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    return summary, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistillToMCU Cross-LLM Validation")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Number of sensor snapshots to test (default: 100)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test with only 5 samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="seed42",
                        choices=["seed42", "seed777", "uci_v3"])
    parser.add_argument("--backend-b", default="qwen3.7-flash-2026-07-15",
                        help="Second LLM backend (default qwen3.7-flash)")
    args = parser.parse_args()

    # Check API keys
    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    b_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if len(ds_key) < 10:
        print("ERROR: DEEPSEEK_API_KEY not set. Export it first.")
        sys.exit(1)
    if len(b_key) < 10:
        print(f"ERROR: key for {args.backend_b} not set.")
        sys.exit(1)

    run_cross_llm(max_samples=args.max_samples, dry_run=args.dry_run,
                  seed=args.seed, dataset=args.dataset,
                  backend_b=args.backend_b)
