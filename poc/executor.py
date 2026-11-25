"""
DistillToMCU Phase 0 PoC — 执行引擎 (v7: 已弃用)
================================================
⚠️ 此模块已不再被任何实验脚本使用。实验使用 experiment.py 内联的
   RealExecutor / MockExecutor。保留此文件供参考（S1 上下文选择
   机制的实现示例），但不纳入正式实验管道。

原用途：模拟 MCU 上的 L5 执行层：规则匹配 → 本地执行 / 未匹配 → 调云端 LLM。
"""

import json
from config import SAFETY_L3


SYSTEM_PROMPT = """
You are a smart home controller agent. Your job is to interpret the user's
natural language request, consider the current sensor readings, and decide
what device actions to take.

Available devices:
- led (adjustable brightness, color temperature) — safety: comfort
- fan (adjustable speed) — safety: comfort
- curtain (open/close, adjustable position) — safety: comfort

Rules:
1. Always use the available tools when device control is needed.
2. Consider energy efficiency (don't run fan if temperature is comfortable).
3. Consider user comfort (appropriate brightness for time of day).
4. If no device action is needed, just respond naturally.
5. When in doubt, choose the safer option (off rather than on).
"""


class Executor:
    """
    执行决策引擎。
    模拟 MCU 主循环：事件到达 → 规则匹配 → 本地执行/云端 LLM 执行。
    """

    def __init__(self, rule_engine, llm_client, trace_store):
        self.engine = rule_engine
        self.llm = llm_client
        self.traces = trace_store
        self.actuator_states = {}  # {device: {"state": "on"/"off", "last_actuated": ts}}
        self.metrics = {
            "total_interactions": 0,
            "local_executions": 0,
            "cloud_executions": 0,
            "local_latency_sum": 0,
            "cloud_latency_sum": 0,
        }

    def handle(self, interaction, current_time=None, weekday=None):
        """
        处理一次用户交互。
        interaction = {sensors, user_input, day, hour, ...}
        返回执行结果。
        """
        self.metrics["total_interactions"] += 1
        sensors = interaction["sensors"]
        user_input = interaction["user_input"]

        # Step 1: 规则匹配
        matches = self.engine.match(sensors, current_time, weekday)
        rule = self.engine.resolve_conflict(matches, self.actuator_states)

        if rule:
            # 命中规则 → 本地执行
            local_latency = self._simulate_local_latency()
            self.metrics["local_executions"] += 1
            self.metrics["local_latency_sum"] += local_latency

            action = rule.action
            self._apply_action(action)

            # 更新规则统计
            self.engine.update_on_execution(rule.id, feedback="accepted")

            # 记录 trace
            trace = self.traces.add(
                sensors=sensors,
                user_input=user_input,
                llm_response={
                    "reasoning": f"Local rule matched: {rule.id}",
                    "tool_calls": [{
                        "id": f"local_{rule.id}",
                        "type": "function",
                        "function": {
                            "name": f"{action['device']}_control",
                            "arguments": json.dumps(
                                dict(command=action["command"], **action.get("params", {}))
                            ),
                        }
                    }],
                    "model": "local_rule_engine",
                    "latency_ms": local_latency,
                },
                execution_mode="local",
                rule_id=rule.id,
            )
            return {
                "mode": "local",
                "rule_id": rule.id,
                "action": action,
                "latency_ms": local_latency,
                "matches_count": len(matches),
            }

        # Step 2: 未命中 → 调云端 LLM
        # 使用支撑机制 S1：上下文选择（fallback_to_llm 内部）
        result = self._cloud_execute(sensors, user_input, interaction)
        return result

    def _cloud_execute(self, sensors, user_input, interaction):
        """
        云端 LLM 决策。
        包含 S1 支撑机制：上下文选择策略。
        """
        import time as _time

        # S1: 上下文选择（支撑机制 A）
        context = self._select_context(sensors)

        t0 = _time.time()
        response = self.llm.cloud_agent_think(
            system_prompt=SYSTEM_PROMPT,
            user_input=user_input,
            sensors=sensors,
            history_summary=context,
        )
        cloud_latency = response.get("latency_ms", 2000)

        self.metrics["cloud_executions"] += 1
        self.metrics["cloud_latency_sum"] += cloud_latency

        # 解析 tool_calls
        tool_calls = response.get("tool_calls") or []
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                if "_control" not in name:
                    continue
                device = name.replace("_control", "")
                import json as _json
                try:
                    args = _json.loads(func.get("arguments", "{}"))
                except _json.JSONDecodeError:
                    continue
                command = args.get("command", "on")
                params = {k: v for k, v in args.items() if k != "command"}
                self._apply_action({"device": device, "command": command, "params": params})

        # 记录 trace
        trace = self.traces.add(
            sensors=sensors,
            user_input=user_input,
            llm_response={
                "reasoning": response.get("content", ""),
                "tool_calls": tool_calls,
                "model": response.get("model", ""),
                "latency_ms": cloud_latency,
            },
            execution_mode="cloud",
            rule_id=None,
        )

        return {
            "mode": "cloud",
            "rule_id": None,
            "action": None,
            "latency_ms": cloud_latency,
            "matches_count": 0,
        }

    def _select_context(self, sensors):
        """
        S1 支撑机制：上下文选择。
        轻量离散评分（keyword overlap + time decay + heuristic MI）。
        不需要 embedding 模型，MCU 可算。
        """
        # 收集最近 50 条 trace + 相关规则
        recent_traces = self.traces.get_last_n(50)
        active_rules = [r for r in self.engine.rules.values()
                        if r.state in ("active", "verified")]

        if not recent_traces:
            return ""

        # 简单评分：时间越近 + 传感器相似 → 分数越高
        scored = []
        active_sensors = [s for s, v in sensors.items() if v is not None]
        for i, t in enumerate(reversed(recent_traces)):
            t_sensors = t.get("sensors", {})
            # sensor overlap score
            overlap = sum(1 for s in active_sensors if s in t_sensors) / max(1, len(active_sensors))
            # time decay
            time_score = 1.0 / (i + 1)  # 最近的最重要
            score = 0.5 * overlap + 0.5 * time_score
            scored.append((score, t))

        # Top-10
        scored.sort(key=lambda x: -x[0])
        top = scored[:10]

        # 构建精简上下文
        lines = []
        for _, t in top:
            llm = t.get("llm_response", {})
            tool_names = []
            for tc in (llm.get("tool_calls") or []):
                fn = tc.get("function", {})
                fn_name = fn.get("name", "").replace("_control", "")
                if fn_name:
                    tool_names.append(fn_name)
            if tool_names:
                lines.append(f"  User: '{t.get('user_input','')}' → [{', '.join(tool_names)}]")

        return "Past relevant interactions:\n" + "\n".join(lines[-5:])

    def _apply_action(self, action):
        """模拟执行器状态更新（MCU 上走 GPIO/PWM）"""
        device = action["device"]
        command = action["command"]
        import time as _time
        now = _time.time()

        if command == "on":
            self.actuator_states[device] = {"state": "on", "last_actuated": now}
        elif command == "off":
            self.actuator_states[device] = {"state": "off", "last_actuated": now}
        elif command == "set":
            self.actuator_states[device] = {"state": "custom", "last_actuated": now}

    def _simulate_local_latency(self):
        """模拟 MCU 本地规则匹配 + GPIO 控制延迟：3-15ms"""
        import random
        return random.randint(3, 15)

    def get_summary(self):
        m = self.metrics
        total = max(1, m["total_interactions"])
        return {
            "autonomy_rate": round(m["local_executions"] / total * 100, 1),
            "cloud_call_reduction": round((1 - m["cloud_executions"] / total) * 100, 1),
            "avg_local_latency_ms": round(m["local_latency_sum"] / max(1, m["local_executions"]), 1),
            "avg_cloud_latency_ms": round(m["cloud_latency_sum"] / max(1, m["cloud_executions"]), 1),
            "total": total,
            "local": m["local_executions"],
            "cloud": m["cloud_executions"],
        }
