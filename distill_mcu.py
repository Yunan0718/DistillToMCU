#!/usr/bin/env python3
"""
DistillToMCU — PC ↔ MCU 蒸馏管道
=================================
从 ESP32 板子读取 LLM 行为轨迹，在 PC 端蒸馏规则，下发回 MCU。

完整闭环:
  1. WS 连接板子 (192.168.2.6:18789)
  2. 发送 trace_read → 获取所有 traces (JSONL)
  3. 运行 Python distiller → 学习包容区间规则
  4. 通过 WS rule_push 推回 MCU
  5. MCU 收到规则 → 存 SPIFFS → 之后传感器匹配命中 → 本地执行

Usage:
    python distill_mcu.py                    # 读 traces + 蒸馏 + 推规则
    python distill_mcu.py --watch --interval 30  # 每 30 秒循环一次
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'poc'))

import websocket
from distiller import Distiller
from rule_engine import RuleEngine
from config import SAFETY_L1


class MCUDistiller:
    def __init__(self, host="192.168.2.6", port=18789):
        self.host = host
        self.port = port
        self.ws = None
        self.engine = RuleEngine()
        self.distiller = Distiller(self.engine, llm_client=None)

    def connect(self):
        url = f"ws://{self.host}:{self.port}/"
        self.ws = websocket.create_connection(url, timeout=10)
        print(f"[OK] Connected to {url}")

    def disconnect(self):
        if self.ws:
            self.ws.close()
            self.ws = None

    def _send_recv(self, msg, timeout=5):
        """Send WS message and wait for one response."""
        self.ws.send(json.dumps(msg))
        self.ws.settimeout(timeout)
        try:
            return json.loads(self.ws.recv())
        except Exception:
            return None

    def read_traces(self):
        """从 MCU 读取 trace 数据。"""
        print("[read] Requesting traces from MCU...")
        resp = self._send_recv({"type": "trace_read"}, timeout=10)
        if not resp:
            print("[ERROR] No response from MCU")
            return []
        data = resp.get("data", [])
        if isinstance(data, list):
            print(f"[OK] Received {len(data)} traces")
            return data
        print("[ERROR] Unexpected response format")
        return []

    def distill_rules(self, traces):
        """在 PC 端蒸馏规则（与 experiment.py 完全相同的逻辑）。"""
        if not traces:
            print("[distill] No traces to distill")
            return []

        # 灌入 trace_store 格式：Python distiller 期望 traces 为 dict list
        # 每个 dict 必须有 execution.mode + llm_response.tool_calls + sensors
        print(f"[distill] Running distillation on {len(traces)} traces...")
        self.distiller._last_distill_ts = 0
        new_count, total = self.distiller.distill(traces)

        # 获取所有规则（活跃 + 候选）
        all_rules = []
        for rule in self.engine.rules.values():
            all_rules.append({
                "id": rule.id,
                "state": rule.state,
                "conditions": rule.conditions,
                "action": rule.action,
                "confidence": rule.confidence,
                "source": rule.source,
                "evidence_count": rule.evidence_count,
                "positive_feedback": rule.positive_feedback,
                "negative_feedback": rule.negative_feedback,
            })
        print(f"[distill] {new_count} new rules from {total} candidates "
              f"(total {len(self.engine.rules)} in engine)")
        return all_rules

    def push_rule(self, rule):
        """通过 WS 推送一条规则到 MCU。"""
        # 构造 rule_store_add 需要的格式
        payload = {
            "id": rule.get("id", f"rule_{int(time.time())}"),
            "state": rule.get("state", "candidate"),
            "conditions": rule.get("conditions", []),
            "action": rule.get("action", {}),
            "source": rule.get("source", "distill"),
            "confidence": rule.get("confidence", 0.7),
            "safety_level": 1,
            "evidence_count": rule.get("evidence_count", 3),
            "positive_feedback": rule.get("positive_feedback", 3),
            "negative_feedback": rule.get("negative_feedback", 0),
        }
        resp = self._send_recv({"type": "rule_push", "rule": payload}, timeout=5)
        if resp:
            status = resp.get("data", {}).get("status", "?")
            if status == "rule_added":
                print(f"  [MCU] Rule {payload['id']} accepted")
                return True
        print(f"  [MCU] Rule push failed for {payload['id']}")
        return False

    def clear_traces(self):
        """蒸馏完成后清空 MCU trace 文件（避免重复蒸馏）。"""
        self._send_recv({"type": "trace_clear"}, timeout=5)
        print("[OK] Traces cleared on MCU")

    def run_once(self):
        """执行一次完整蒸馏循环。"""
        print("\n" + "=" * 60)
        print("  DistillToMCU — PC↔MCU Distillation Cycle")
        print("=" * 60)

        # Step 1: 读 traces
        traces = self.read_traces()
        if not traces:
            print("[skip] No traces available. Inject sensor data first.")
            return 0

        # Step 2: 蒸馏
        rules = self.distill_rules(traces)

        # Step 3: 推规则回 MCU
        pushed = 0
        for rule in rules:
            if self.push_rule(rule):
                pushed += 1

        # Step 4: 清空 MCU traces（已处理的不再保留）
        if pushed > 0:
            self.clear_traces()

        print(f"\n[DONE] Cycle complete: {len(traces)} traces → "
              f"{len(rules)} rules → {pushed} pushed to MCU")
        return pushed


def main():
    parser = argparse.ArgumentParser(description="DistillToMCU PC↔MCU Pipeline")
    parser.add_argument("--host", default="192.168.2.6")
    parser.add_argument("--port", type=int, default=18789)
    parser.add_argument("--watch", action="store_true",
                        help="Continuously watch and distill")
    parser.add_argument("--interval", type=int, default=30,
                        help="Interval between cycles (seconds)")
    args = parser.parse_args()

    d = MCUDistiller(host=args.host, port=args.port)

    try:
        d.connect()
    except Exception as e:
        print(f"[FATAL] Cannot connect to MCU: {e}")
        print("  Check: 1) ESP32 is powered on and WiFi connected")
        print("         2) IP is correct (default: 192.168.2.6)")
        sys.exit(1)

    if args.watch:
        print(f"[watch] Continuous mode, interval={args.interval}s")
        cycle = 0
        while True:
            cycle += 1
            print(f"\n=== Cycle {cycle} ===")
            d.run_once()
            time.sleep(args.interval)
    else:
        d.run_once()

    d.disconnect()


if __name__ == "__main__":
    main()
