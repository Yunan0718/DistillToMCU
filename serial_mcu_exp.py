"""
DistillToMCU — MCU Autonomy Verification Experiment
===================================================
对比实验：MCU有规则 vs MCU无规则，验证规则驱动的本地自主执行。

Usage:
    python serial_mcu_exp.py --port COM6 --dataset seed42 --count 100
"""

import serial
import time
import json
import os
import sys
import argparse


def load_sensors(dataset_name):
    base = os.path.join(os.path.dirname(__file__), "poc", "output")
    paths = {"seed42": os.path.join(base, "seed42", "traces.jsonl"),
             "seed123": os.path.join(base, "run_seed123", "traces.jsonl"),
             "seed999": os.path.join(base, "run_seed999", "traces.jsonl"),
             "seed777": os.path.join(base, "seed777", "traces.jsonl"),
             "uci_v3": os.path.join(base, "uci_v3_seed42", "traces.jsonl")}
    path = paths.get(dataset_name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(dataset_name)
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line).get("sensors", {})
                if s: out.append(s)
    return out


class MCU:
    def __init__(self, port):
        self.ser = serial.Serial(port, 115200, timeout=0.3)
        time.sleep(8)
        self._drain()

    def _drain(self):
        while self.ser.in_waiting:
            self.ser.read(self.ser.in_waiting)

    def _write_line(self, line: str, chunk: int = 64, gap: float = 0.02):
        """Write a command in small chunks to avoid overflowing the
        USB-Serial-JTAG RX buffer (~256B default). A single large write
        silently drops bytes and truncates long JSON commands."""
        data = line.encode()
        for i in range(0, len(data), chunk):
            self.ser.write(data[i:i + chunk])
            self.ser.flush()
            if gap > 0:
                time.sleep(gap)

    def inject(self, sensors, retries: int = 2):
        js = json.dumps(sensors, separators=(",", ":"))
        for attempt in range(retries + 1):
            self.ser.reset_input_buffer()
            self._write_line(f"sensor {js}\r\n")
            # Wait for MCU echo "[OK]" or "ERROR" (max 1.5s)
            buf, t0 = "", time.time()
            ok = False
            while time.time() - t0 < 1.5:
                if self.ser.in_waiting:
                    buf += self.ser.read(self.ser.in_waiting).decode("utf-8", "ignore")
                    if "[OK]" in buf or "ERROR" in buf:
                        ok = True
                        break
                time.sleep(0.02)
            if ok:
                return True, "OK" if "[OK]" in buf else "ERROR"
        return False, "NO_ACK"

    def push_rule(self, rule: dict):
        """Push one rule to the MCU via CLI `rule add <json>`."""
        r = {
            "conditions": rule.get("conditions", []),
            "action": rule.get("action", {}),
            "state": rule.get("state", "verified"),
            "safety_level": rule.get("safety_level", 1),
            "confidence": rule.get("confidence", 0.9),
        }
        js = json.dumps(r, separators=(",", ":"))
        self.ser.reset_input_buffer()
        self._write_line(f"rule add {js}\r\n")
        buf, t0 = "", time.time()
        while time.time() - t0 < 2.0:
            if self.ser.in_waiting:
                buf += self.ser.read(self.ser.in_waiting).decode("utf-8", "ignore")
                if "[OK]" in buf or "ERROR" in buf:
                    break
            time.sleep(0.02)
        return "[OK]" in buf

    def stats(self):
        self._drain()
        self.ser.write(b"stats\r\n")
        self.ser.flush()
        time.sleep(0.5)
        buf, t0 = "", time.time()
        while time.time() - t0 < 2:
            if self.ser.in_waiting:
                buf += self.ser.read(self.ser.in_waiting).decode("utf-8", "ignore")
            if "{" in buf and "}" in buf: break
            time.sleep(0.05)
        try: return json.loads(buf[buf.index("{"):buf.rindex("}")+1])
        except: return {}

    def latstats(self):
        """Read match-latency distribution (p50/p95/p99) via `latstats`."""
        self._drain()
        self.ser.write(b"latstats\r\n")
        self.ser.flush()
        buf, t0 = "", time.time()
        while time.time() - t0 < 3:
            if self.ser.in_waiting:
                buf += self.ser.read(self.ser.in_waiting).decode("utf-8", "ignore")
            if "}" in buf:
                break
            time.sleep(0.05)
        try:
            return json.loads(buf[buf.index("{"):buf.rindex("}")+1])
        except Exception:
            return {}

    def close(self):
        self.ser.close()


def load_rules_for_push(dataset_name: str) -> list[dict]:
    """Load distilled rules from output/<exp>/rules_v10.json (preferred) or
    rules_snapshot.json (legacy online snapshot)."""
    base = os.path.join(os.path.dirname(__file__), "poc", "output")
    dirs = {"seed42": "seed42", "seed123": "run_seed123",
            "seed999": "run_seed999", "seed777": "seed777",
            "uci_v3": "uci_v3_seed42", "strands": "strands_seed42"}
    exp_dir = dirs.get(dataset_name)
    if not exp_dir:
        return []
    path_v10 = os.path.join(base, exp_dir, "rules_v10.json")
    path = path_v10 if os.path.exists(path_v10) \
        else os.path.join(base, exp_dir, "rules_snapshot.json")
    if not os.path.exists(path):
        print("  rules_v10.json / rules_snapshot.json missing; "
              "run oracle_replay first")
        return []
    d = json.load(open(path, encoding="utf-8"))
    rules = d if isinstance(d, list) else d.get("rules", [])
    if path == path_v10:
        print(f"  using v10 distilled rules ({len(rules)} rules)")
    # keep only verified/active rules (candidate rules have too little evidence)
    keep = [r for r in rules if r.get("state") in ("verified", "active")]
    if not keep:
        keep = rules[:3]
    return keep[:20]


def run(dataset, port, count, push_rules=False, delay=3.5):
    sensors = load_sensors(dataset)
    count = min(count, len(sensors))
    print(f"{dataset}: {len(sensors)} sensors, running {count}")

    mcu = MCU(port)
    if push_rules:
        rules = load_rules_for_push(dataset)
        print(f"Pushing {len(rules)} rules to MCU...")
        pushed = 0
        for r in rules:
            if mcu.push_rule(r):
                pushed += 1
            else:
                print(f"  FAILED to push rule: {r.get('action')}")
        print(f"Pushed {pushed}/{len(rules)} rules")
        time.sleep(1.0)

    s0 = mcu.stats()
    rules_n = s0.get("rules_total", 0)
    rules_a = s0.get("rules_active", 0)
    sram0 = s0.get("free_sram", 0)
    print(f"Init: rules={rules_n} active={rules_a} SRAM={sram0}")

    batch = 20
    log = []
    lost = 0
    for i in range(count):
        ok, status = mcu.inject(sensors[i])
        if not ok:
            lost += 1
        time.sleep(max(0.0, delay - 1.5))  # keep steady cadence
        if (i + 1) % batch == 0:
            s = mcu.stats()
            t = s.get("total", 0)
            l = s.get("local", 0)
            c = s.get("cloud", 0)
            ar = l / max(1, l + c) * 100
            ra = s.get("rules_active", 0)
            sr = s.get("free_sram", 0)
            print(f"  [{i+1:4d}/{count}] t={t} L={l} C={c} AR={ar:.0f}% "
                  f"active={ra} SRAM={sr} lost={lost}")
            log.append({"n": i + 1, "total": t, "local": l, "cloud": c,
                        "ar": round(ar, 1), "active": ra, "sram": sr,
                        "lost": lost})

    sf = mcu.stats()
    print(f"\nFinal: AR={sf.get('ar_pct',0)}% rules={sf.get('rules_total',0)}"
          f" active={sf.get('rules_active',0)} SRAM={sf.get('free_sram',0)}")
    print(f"Result: {sf.get('local',0)}/{sf.get('total',0)} local executions, "
          f"AR={sf.get('ar_pct',0)}%")

    out_dir = os.path.join(os.path.dirname(__file__), "poc", "output")
    out_path = os.path.join(out_dir, f"mcu_exp_{dataset}_{int(time.time())}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "count": count, "history": log,
                   "final": sf, "inject_lost": lost, "pushed_rules": push_rules},
                  f, indent=2)
    print(f"Saved: {out_path}")

    mcu.close()
    return log


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM6")
    p.add_argument("--dataset", default="seed42")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--push-rules", action="store_true",
                   help="Push distilled rules to MCU before running")
    p.add_argument("--delay", type=float, default=3.5,
                   help="Seconds between injections (default 3.5)")
    a = p.parse_args()
    run(a.dataset, a.port, a.count, push_rules=a.push_rules, delay=a.delay)
