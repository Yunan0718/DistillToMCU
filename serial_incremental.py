"""
DistillToMCU — MCU Incremental Autonomy Experiment
===================================================
Simulates multi-day deployment on real ESP32-S3 hardware.
Each "day": feed sensor data → MCU processes → PC distills
traces → pushes rules → MCU gets smarter → AR grows.

Usage:
    python serial_incremental.py --port COM6 --dataset seed42 --days 10

Output: output/mcu_incremental_<ts>.jsonl
"""

import serial
import time
import json
import os
import sys
import argparse
import random

# Ensure poc is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "poc"))


def load_sensors(dataset_name):
    base = os.path.join(os.path.dirname(__file__), "poc", "output")
    paths = {
        "seed42": os.path.join(base, "seed42", "traces.jsonl"),
        "seed123": os.path.join(base, "run_seed123", "traces.jsonl"),
        "seed999": os.path.join(base, "run_seed999", "traces.jsonl"),
        "uci_v3": os.path.join(base, "uci_v3_seed42", "traces.jsonl"),
    }
    path = paths.get(dataset_name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    sensors_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                s = t.get("sensors", {})
                if s:
                    sensors_list.append(s)
    return sensors_list


class MCU:
    def __init__(self, port):
        self.ser = serial.Serial(port, 115200, timeout=0.3)
        time.sleep(6)
        self._drain()

    def _drain(self):
        while self.ser.in_waiting:
            self.ser.read(self.ser.in_waiting)

    def cmd(self, text):
        self.ser.write((text + "\r\n").encode())
        self.ser.flush()

    def inject(self, sensors):
        js = json.dumps(sensors, separators=(",", ":"))
        self.cmd(f"sensor {js}")
        time.sleep(2.8)

    def stats(self):
        self._drain()
        self.cmd("stats")
        time.sleep(0.5)
        buf = ""
        t0 = time.time()
        while time.time() - t0 < 2:
            if self.ser.in_waiting:
                buf += self.ser.read(self.ser.in_waiting).decode("utf-8", "ignore")
            if "{" in buf and "}" in buf:
                break
            time.sleep(0.05)
        try:
            s = buf.index("{")
            e = buf.rindex("}") + 1
            return json.loads(buf[s:e])
        except (ValueError, json.JSONDecodeError):
            return {}

    def rules(self):
        self._drain()
        self.cmd("rules")
        time.sleep(0.3)
        buf = ""
        t0 = time.time()
        while time.time() - t0 < 1.5:
            if self.ser.in_waiting:
                buf += self.ser.read(self.ser.in_waiting).decode("utf-8", "ignore")
            time.sleep(0.05)
        return buf

    def push_rule(self, rule_dict):
        seen = set()
        uniq = []
        for c in rule_dict.get("conditions", []):
            k = (c["sensor"], c["op"], c["value"])
            if k not in seen:
                seen.add(k)
                uniq.append({"sensor": c["sensor"], "op": c["op"], "value": c["value"]})
        mini = {
            "conditions": uniq,
            "action": rule_dict["action"],
            "id": rule_dict["id"],
            "state": rule_dict.get("state", "candidate"),
        }
        js = json.dumps(mini, separators=(",", ":"))
        self.cmd(f"rule add {js}")
        time.sleep(0.5)
        self._drain()

    def close(self):
        self.ser.close()


def run_incremental(dataset, port, days, interactions_per_day=17):
    sensors = load_sensors(dataset)
    print(f"Dataset: {dataset}, {len(sensors)} readings, {days} days, ~{interactions_per_day}/day")

    mcu = MCU(port)
    initial = mcu.stats()
    print(f"MCU init: {initial.get('rules_total', 0)} rules, "
          f"SRAM={initial.get('free_sram', 0)}")

    # Import distiller
    from distiller import Distiller
    from rule_engine import RuleEngine

    output_dir = os.path.join(os.path.dirname(__file__), "poc", "output")
    log_path = os.path.join(output_dir, f"mcu_incremental_{dataset}_{int(time.time())}.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")

    sensor_idx = 0
    day_results = []

    for day in range(days):
        # Feed sensors for this "day"
        batch_cloud = 0
        batch_local = 0
        for _ in range(interactions_per_day):
            if sensor_idx >= len(sensors):
                sensor_idx = 0  # cycle
            mcu.inject(sensors[sensor_idx])
            sensor_idx += 1

        time.sleep(2)
        stats_now = mcu.stats()
        local = stats_now.get("local", 0)
        cloud = stats_now.get("cloud", 0)
        total = local + cloud
        ar = local / max(1, total) * 100

        # Distill: read rules from MCU, build engine, distill from known traces
        # We can't read MCU traces via serial, so we use the PC traces as proxy
        # (the MCU interacts with same sensor data)
        engine = RuleEngine()
        d = Distiller(engine, llm_client=None)

        # Build proxy traces from PC dataset
        pc_traces = []
        seen = min(sensor_idx, len(sensors))
        for i in range(seen):
            s = sensors[i]
            # Load matching cloud trace for action
            # Simplified: use the original trace's LLM response if sensor matches
            pc_traces.append({"sensors": s, "llm_response": {"tool_calls": []},
                              "execution": {"mode": "cloud"}})

        d.distill(pc_traces[:max(1, seen)])

        # Push new rules to MCU
        pushed = 0
        for r in engine.rules.values():
            mcu.push_rule(r.to_dict())
            pushed += 1

        record = {
            "day": day + 1,
            "total": total,
            "local": local,
            "cloud": cloud,
            "ar_pct": round(ar, 1),
            "rules_pushed": pushed,
            "rules_total": stats_now.get("rules_total", 0),
            "rules_active": stats_now.get("rules_active", 0),
            "free_sram": stats_now.get("free_sram", 0),
        }
        day_results.append(record)
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

        print(f"  Day {day+1:2d}/{days}  total={total} local={local} cloud={cloud} "
              f"AR={ar:.0f}%  rules={record['rules_total']} pushed={pushed}")

    log_f.close()

    # Summary
    print(f"\n--- MCU Incremental Experiment Summary ---")
    print(f"  Days: {days}")
    print(f"  Dataset: {dataset}")
    for r in day_results:
        print(f"  D{r['day']:2d}: AR={r['ar_pct']:5.1f}% rules={r['rules_total']}")

    final = mcu.stats()
    print(f"\n  Final MCU: {final.get('rules_total',0)} rules, "
          f"AR={final.get('ar_pct',0)}%, SRAM={final.get('free_sram',0)}")

    mcu.close()
    return day_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCU Incremental Autonomy Experiment")
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--dataset", default="seed42")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--per-day", type=int, default=17)
    args = parser.parse_args()
    run_incremental(args.dataset, args.port, args.days, args.per_day)
