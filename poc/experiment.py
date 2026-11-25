"""
DistillToMCU Phase 0a — 主实验脚本 (v2: 真实 LLM + 包容区间学习)
==============================================================
运行：python experiment.py --real     # 真实 DeepSeek API
      python experiment.py --mock     # 内置 mock（快速验证）
      python experiment.py --real --days 10  # 缩短实验用于调试

输出：output/traces.jsonl、rules_snapshot.json、metrics.jsonl
"""

import os
import sys
import json
import gc
import random
import argparse
from datetime import datetime, timedelta

from config import (
    SIMULATION_DAYS, SEED, OUTPUT_DIR, METRICS_FILE,
    TRACE_FILE, RULES_FILE,
)
from simulator import SensorSimulator, InteractionGenerator
from trace_store import TraceStore
from rule_engine import RuleEngine
from distiller import Distiller


# ============================================================
# Real Executor — 调用真实 DeepSeek API
# ============================================================

class RealExecutor:
    """使用真实 LLM API 的执行器"""

    def __init__(self, rule_engine, trace_store, llm_client_module):
        self.engine = rule_engine
        self.traces = trace_store
        self.llm = llm_client_module
        self.actuator_states = {}
        self.metrics = {
            "total": 0, "local": 0, "cloud": 0,
            "local_lat_sum": 0, "cloud_lat_sum": 0,
        }
        self._total_cost_cny = 0.0  # 估算 API 花费（人民币，v6 修复单位 bug）
        self._total_tokens = 0

    def handle(self, interaction, current_time=None):
        self.metrics["total"] += 1
        sensors = interaction["sensors"]
        user_input = interaction["user_input"]

        # Step 1: 规则匹配
        matches = self.engine.match(sensors, current_time)
        rule = self.engine.resolve_conflict(matches, self.actuator_states)

        if rule:
            # === 本地执行 ===
            lat = random.randint(3, 15)  # simulated, not measured on MCU
            self.metrics["local"] += 1
            self.metrics["local_lat_sum"] += lat

            action = rule.action
            self._apply_action(action)
            self.engine.update_on_execution(rule.id, "accepted")

            # 记录 trace（标记为 local）
            self.traces.add(
                sensors, user_input,
                llm_response={
                    "reasoning": f"Local rule: {rule.id}",
                    "tool_calls": [{
                        "id": f"local_{rule.id}",
                        "type": "function",
                        "function": {
                            "name": f"{action['device']}_control",
                            "arguments": json.dumps(
                                dict(command=action["command"],
                                     **action.get("params", {}))
                            ),
                        }
                    }],
                    "model": "local_rule_engine",
                    "latency_ms": lat,
                },
                execution_mode="local",
                rule_id=rule.id,
            )
            return {"mode": "local", "rule_id": rule.id, "latency_ms": lat}

        # Step 2: 未命中 → 真实 LLM API 调用
        import time as _time
        t0 = _time.time()

        response = self.llm.cloud_agent_think(
            system_prompt="""You are a smart home controller agent.
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
5. Respond in user's language.""",
            user_input=user_input,
            sensors=sensors,
        )

        cloud_latency = response.get("latency_ms", 2000)
        self.metrics["cloud"] += 1
        self.metrics["cloud_lat_sum"] += cloud_latency

        # 估算费用（DeepSeek V4 Flash: ¥1/M input, ¥2/M output）
        # 粗略估算：每次调用约 500 input + 100 output tokens → ¥0.0007/次
        # ⚠️ v6 修复：此值已是人民币，不再乘汇率。
        est_cost = (500 * 1 + 100 * 2) / 1_000_000
        self._total_cost_cny += est_cost
        self._total_tokens += 600

        # 解析 tool_calls → 执行
        tool_calls = response.get("tool_calls") or []
        action_taken = None
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                if "_control" not in name:
                    continue
                device = name.replace("_control", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    continue
                command = args.get("command", "on")
                params = {k: v for k, v in args.items() if k != "command"}
                action_taken = {"device": device, "command": command, "params": params}
                self._apply_action(action_taken)

        # 记录 trace
        self.traces.add(
            sensors, user_input,
            llm_response={
                "reasoning": response.get("content", ""),
                "tool_calls": tool_calls,
                "model": response.get("model", "deepseek-v4-flash"),
                "latency_ms": cloud_latency,
            },
            execution_mode="cloud",
            rule_id=None,
        )

        return {
            "mode": "cloud",
            "rule_id": None,
            "latency_ms": cloud_latency,
            "action": action_taken,
        }

    def _apply_action(self, action):
        import time as _time
        now = _time.time()
        dev = action["device"]
        cmd = action["command"]
        if cmd == "on":
            self.actuator_states[dev] = {"state": "on", "last_actuated": now}
        elif cmd == "off":
            self.actuator_states[dev] = {"state": "off", "last_actuated": now}
        elif cmd == "set":
            self.actuator_states[dev] = {"state": "custom", "last_actuated": now}

    def get_summary(self):
        m = self.metrics
        t = max(1, m["total"])
        lc = max(1, m["local"])
        cc = max(1, m["cloud"])
        return {
            "autonomy_rate": round(m["local"] / t * 100, 1),
            "cloud_call_reduction": round((1 - m["cloud"] / t) * 100, 1),
            "avg_local_latency_ms": round(m["local_lat_sum"] / lc, 1),
            "avg_cloud_latency_ms": round(m["cloud_lat_sum"] / cc, 1),
            "total": t, "local": m["local"], "cloud": m["cloud"],
        }


# ============================================================
# Mock Executor（无 API Key 时的 fallback）
# ============================================================

class MockExecutor:
    """使用启发式规则的执行器。仅用于框架验证，不用于正式实验。"""

    def __init__(self, rule_engine, trace_store):
        self.engine = rule_engine
        self.traces = trace_store
        self.actuator_states = {}
        self.metrics = {
            "total": 0, "local": 0, "cloud": 0,
            "local_lat_sum": 0, "cloud_lat_sum": 0,
        }

    def handle(self, interaction, current_time=None):
        self.metrics["total"] += 1
        sensors = interaction["sensors"]
        user_input = interaction["user_input"]
        temp = sensors.get("temperature", 25)
        light = sensors.get("light", 500)

        matches = self.engine.match(sensors, current_time)
        rule = self.engine.resolve_conflict(matches, self.actuator_states)

        if rule:
            lat = random.randint(3, 15)
            self.metrics["local"] += 1
            self.metrics["local_lat_sum"] += lat
            self._apply(rule.action)
            self.engine.update_on_execution(rule.id, "accepted")
            self.traces.add(
                sensors, user_input,
                llm_response=self._build_response(rule.action, lat, f"rule:{rule.id}"),
                execution_mode="local", rule_id=rule.id,
            )
            return {"mode": "local", "rule_id": rule.id}

        action = self._mock_cloud_decision(sensors, user_input)
        lat = random.randint(800, 2500)
        self.metrics["cloud"] += 1
        self.metrics["cloud_lat_sum"] += lat
        if action:
            self._apply(action)
        self.traces.add(
            sensors, user_input,
            llm_response=self._build_response(action, lat, "mock"),
            execution_mode="cloud", rule_id=None,
        )
        return {"mode": "cloud", "rule_id": None}

    def _mock_cloud_decision(self, sensors, user_input):
        inp = user_input.lower()
        temp = sensors.get("temperature", 25)
        light = sensors.get("light", 500)
        motion = sensors.get("motion", 0)

        if any(w in inp for w in ["热", "hot", "闷", "warm", "太暖"]):
            return {"device": "fan", "command": "on", "params": {"speed": 2}}
        if any(w in inp for w in ["暗", "dark", "灯", "light", "亮"]):
            return {"device": "led", "command": "on",
                    "params": {"brightness": random.choice([50, 60, 70, 80])}}
        if any(w in inp for w in ["冷", "cold", "凉"]):
            return {"device": "fan", "command": "off", "params": {}}
        if any(w in inp for w in ["窗帘", "curtain", "morning", "起床"]):
            return {"device": "curtain", "command": "on",
                    "params": {"position": random.choice([80, 90, 100])}}
        if any(w in inp for w in ["睡", "sleep", "night", "晚安"]):
            return {"device": "led", "command": "off", "params": {}}
        if any(w in inp for w in ["温度", "几度", "status"]):
            return None
        if temp > 32:
            return {"device": "fan", "command": "on", "params": {"speed": 3}}
        if light < 30 and motion == 1:
            return {"device": "led", "command": "on",
                    "params": {"brightness": random.choice([50, 60])}}
        return None

    def _apply(self, action):
        import time as _time
        dev = action["device"]
        cmd = action["command"]
        now = _time.time()
        if cmd == "on":
            self.actuator_states[dev] = {"state": "on", "last_actuated": now}
        elif cmd == "off":
            self.actuator_states[dev] = {"state": "off", "last_actuated": now}

    def _build_response(self, action, latency_ms, model):
        return {
            "reasoning": f"Decision by {model}",
            "tool_calls": [{
                "id": f"call_{random.randint(1000,9999)}",
                "type": "function",
                "function": {
                    "name": f"{action['device']}_control" if action else "read_sensors",
                    "arguments": json.dumps(
                        dict(command=action["command"], **action.get("params", {}))
                    ) if action else "{}",
                }
            }] if action else [],
            "model": model, "latency_ms": latency_ms,
        }

    def get_summary(self):
        m = self.metrics
        t = max(1, m["total"])
        lc = max(1, m["local"])
        cc = max(1, m["cloud"])
        return {
            "autonomy_rate": round(m["local"] / t * 100, 1),
            "cloud_call_reduction": round((1 - m["cloud"] / t) * 100, 1),
            "avg_local_latency_ms": round(m["local_lat_sum"] / lc, 1),
            "avg_cloud_latency_ms": round(m["cloud_lat_sum"] / cc, 1),
            "total": t, "local": m["local"], "cloud": m["cloud"],
        }


# ============================================================
# 主实验
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DistillToMCU Phase 0a Experiment")
    parser.add_argument("--real", action="store_true",
                        help="Use real DeepSeek API (requires env var DEEPSEEK_API_KEY)")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Use built-in mock executor (default)")
    parser.add_argument("--days", type=int, default=SIMULATION_DAYS,
                        help=f"Number of simulation days (default: {SIMULATION_DAYS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stop after first LLM call (test mode)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for traces and metrics")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--selector", choices=["deterministic", "pms"],
                        default="deterministic",
                        help="Rule conflict resolution: deterministic sorting "
                             "(default) or PMS (exploration-aware, v10)")
    parser.add_argument("--backend", default=None,
                        help="LLM backend (default deepseek-v4-flash; "
                             "e.g. qwen3.7-flash-2026-07-15)")
    args = parser.parse_args()

    if args.backend:
        import llm_client
        llm_client._ACTIVE_BACKEND = args.backend

    # 设置全局种子
    exp_seed = args.seed
    random.seed(exp_seed)

    if args.real:
        use_real = True
    elif args.mock:
        use_real = False
    else:
        use_real = False  # default to mock

    print("=" * 60)
    if use_real:
        try:
            from llm_client import _ACTIVE_BACKEND
            backend_name = _ACTIVE_BACKEND or "deepseek-v4-flash"
        except Exception:
            backend_name = "deepseek-v4-flash"
        mode_str = f"REAL LLM ({backend_name})"
    else:
        mode_str = "MOCK (heuristic)"
    print(f"  DistillToMCU Phase 0a — Growing Autonomy Experiment")
    print(f"  Mode: {mode_str}")
    print(f"  Days: {args.days}")
    print(f"  Seed: {exp_seed}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 初始化 ----
    sensor_sim = SensorSimulator(seed=exp_seed)
    trace_store = TraceStore(output_dir=args.output_dir)
    trace_store.start_new_session()  # v7: 显式清空旧数据
    rule_engine = RuleEngine(selector=args.selector)

    if use_real:
        # 检查 API Key
        import os as _os
        api_key = _os.environ.get("DEEPSEEK_API_KEY", "")
        # v7 fix: also reject obvious placeholder keys
        placeholder_patterns = ["sk-your-key", "your-key-here", "sk-xxxxxxxx",
                                "your_api_key", "change-me", "placeholder"]
        if len(api_key) < 10 or any(p in api_key.lower() for p in placeholder_patterns):
            print("\n  ERROR: DEEPSEEK_API_KEY is missing or looks like a placeholder!")
            print("  Set a real key or run with --mock mode.")
            sys.exit(1)

        import llm_client
        executor = RealExecutor(rule_engine, trace_store, llm_client)
        distiller = Distiller(rule_engine, llm_client=llm_client)
        interaction_gen = InteractionGenerator(sensor_sim, llm_client=None, seed=exp_seed)
        # mock 住户输入（住户模拟器额外花 API 调用，Phase 0b 再启用）
    else:
        executor = MockExecutor(rule_engine, trace_store)
        distiller = Distiller(rule_engine, llm_client=None)
        interaction_gen = InteractionGenerator(sensor_sim, llm_client=None, seed=exp_seed)

    # ---- 实验主循环 ----
    daily_metrics = []

    print(f"\n  Starting {args.days}-day simulation...\n")
    import time as _time
    exp_start = _time.time()

    # v10.6: 交互序列一次性预生成，与 LLM 反馈解耦。
    # 此前逐日 generate_day() 与本地执行延迟共用全局 RNG：LLM 行为不同 →
    # 本地命中次数不同 → 后续随机数流偏移 → 同一"固定种子"在不同后端下
    # 数据序列分叉（DeepSeek/Qwen 在线运行不可复现的根因）。
    all_interactions = [
        interaction_gen.generate_day(day) for day in range(args.days)
    ]

    for day in range(args.days):
        t_day_start = _time.time()
        print(f"  Day {day+1:2d}/{args.days}  ", end="", flush=True)

        interactions = all_interactions[day]

        for interaction in interactions:
            current_time = datetime(2026, 7, 1, interaction["hour"], 0, 0) + timedelta(days=day)
            executor.handle(interaction, current_time)

        # 每日维护
        gc.collect()  # 释放 LLM response 对象
        rule_engine.update_all_freshness()
        rule_engine.gc()
        new_rules, _ = distiller.distill(trace_store.traces)

        if new_rules > 0:
            print(f"[+{new_rules}r] ", end="", flush=True)

        # 收集指标
        summary = executor.get_summary()
        stats = rule_engine.stats()
        daily_metrics.append({
            "day": day + 1,
            "autonomy_rate": summary["autonomy_rate"],
            "cloud_calls": summary["cloud"],
            "local_calls": summary["local"],
            "total": summary["total"],
            "avg_local_lat_ms": summary["avg_local_latency_ms"],
            "avg_cloud_lat_ms": summary["avg_cloud_latency_ms"],
            "active_rules": stats["active_count"],
            "total_rules": stats["total"],
            "new_rules_today": new_rules,
        })

        day_elapsed = _time.time() - t_day_start
        print(f"AR={summary['autonomy_rate']:.0f}%  "
              f"({day_elapsed:.0f}s)")

        # Dry run: stop after first successful LLM call
        if args.dry_run and summary["cloud"] > 0:
            print(f"\n  [DRY RUN] First LLM call successful. Stopping.")
            break

    # ---- 最终报告 ----
    elapsed = _time.time() - exp_start
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)

    final = executor.get_summary()
    final_stats = rule_engine.stats()

    print(f"\n  Mode:                     {mode_str}")
    print(f"  Autonomy Rate:            {final['autonomy_rate']:.1f}%")
    print(f"  Cloud Call Reduction:     {final['cloud_call_reduction']:.1f}%")
    print(f"  Avg Local Latency:        {final['avg_local_latency_ms']:.1f} ms")
    print(f"  Avg Cloud Latency:        {final['avg_cloud_latency_ms']:.1f} ms")
    print(f"  Total Interactions:       {final['total']}")
    print(f"  Local Executions:         {final['local']}")
    print(f"  Cloud Executions:         {final['cloud']}")
    print(f"  Rules — Total:            {final_stats['total']}")
    print(f"  Rules — Active:           {final_stats['active_count']}")
    print(f"  Rules — By State:         {final_stats['by_state']}")
    print(f"  Experiment Time:          {elapsed:.0f}s ({elapsed/60:.1f}m)")

    if use_real and hasattr(executor, '_total_cost_cny'):
        print(f"  Est. API Cost:            ~{executor._total_cost_cny:.2f} CNY")

    # ---- 保存 ----
    rule_engine.save_snapshot(
        os.path.join(args.output_dir, RULES_FILE)
    )
    with open(os.path.join(args.output_dir, METRICS_FILE), "w", encoding="utf-8") as f:
        for m in daily_metrics:
            f.write(json.dumps(m) + "\n")

    # ---- 自主率增长曲线 ----
    print(f"\n  Growing Autonomy Curve:")
    print(f"  {'Day':>4s} {'AR%':>6s} {'Rules':>6s}  Curve")
    for m in daily_metrics[::max(1, len(daily_metrics) // 15)]:
        bar = "#" * int(m["autonomy_rate"] // 5)
        print(f"  {m['day']:4d} {m['autonomy_rate']:5.1f}% {m['active_rules']:5d}   {bar}")

    print(f"\n  Output: {os.path.abspath(args.output_dir)}/")
    print(f"    {TRACE_FILE}   — interaction traces (JSONL)")
    print(f"    {RULES_FILE} — final rule snapshot (JSON)")
    print(f"    {METRICS_FILE}  — daily metrics (JSONL)")

    # ---- 判定 ----
    ar = final["autonomy_rate"]
    if use_real:
        print(f"\n  NOTE: Real LLM experiment. AR={ar:.0f}% is actual data.")
    else:
        print(f"\n  NOTE: Mock mode. AR={ar:.0f}% is NOT real — use --real for actual experiment.")

    return ar


if __name__ == "__main__":
    main()
