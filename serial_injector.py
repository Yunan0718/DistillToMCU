#!/usr/bin/env python3
"""
DistillToMCU Phase 1 — USB Serial Sensor Injector
==================================================
PC-side script that reads sensor datasets (UCI Occupancy Detection, CASAS, etc.)
and sends them line-by-line to ESP32-S3 via USB serial port.

The ESP32 firmware uses the 'sensor <json>' CLI command to receive injected data.
Each sensor reading is sent as a serial command, and the ESP32's agent loop
processes it: rule matching → local execution / cloud LLM.

Usage:
    # Auto-detect COM port and send UCI enriched data
    python serial_injector.py --dataset uci --baud 115200

    # Specify port explicitly
    python serial_injector.py --port COM5 --dataset uci

    # Send one snapshot every 5 seconds (default)
    python serial_injector.py --dataset uci --interval 5

    # Send all snapshots as fast as possible (batch mode)
    python serial_injector.py --dataset uci --batch

    # List available COM ports
    python serial_injector.py --list

Dataset options:
    uci     — UCI Occupancy Detection enriched snapshots (605 snapshots)
    casas   — CASAS Aruba-1 snapshots (5,376 snapshots)
    synth   — Synthetic sine-wave sensor data (infinite generation)
    custom  — Custom JSONL file
"""

import json
import os
import sys
import time
import argparse
import random
import math
from typing import Optional

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "poc")


def find_serial_port() -> Optional[str]:
    """Auto-detect ESP32 serial port."""
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        # ESP32 typically shows as Silicon Labs CP210x or CH340/CH341
        if any(kw in p.description.lower() for kw in
               ("cp210", "ch340", "ch341", "silicon", "esp32", "usb serial")):
            print(f"[INFO] Auto-detected: {p.device} ({p.description})")
            return p.device
    # Fallback: return first available
    if ports:
        print(f"[INFO] No ESP32 detected, trying first port: {ports[0].device}")
        return ports[0].device
    return None


def list_ports():
    """List all available serial ports."""
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print(f"{'Port':<20s} {'Description':<40s} {'HWID'}")
    print("-" * 80)
    for p in ports:
        print(f"{p.device:<20s} {p.description[:40]:<40s} {p.hwid[:20] if p.hwid else 'N/A'}")


# ==================== Data Sources ====================

def load_uci_snapshots() -> list:
    """Load UCI Occupancy Detection enriched snapshots."""
    path = os.path.join(POC_DIR, "data", "uci", "snapshots_enriched.json")
    if not os.path.exists(path):
        print(f"[ERROR] UCI data not found: {path}")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} UCI snapshots")
    return data


def load_casas_snapshots() -> list:
    """Load CASAS Aruba-1 snapshots."""
    path = os.path.join(POC_DIR, "data", "casas", "aruba_snapshots.json")
    if not os.path.exists(path):
        print(f"[ERROR] CASAS data not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} CASAS snapshots")
    return data


def generate_synthetic_snapshot(day: int = 0) -> dict:
    """Generate one synthetic sensor snapshot (sine-wave based)."""
    hour = (day * 24) % 24 + random.uniform(0, 1)
    # Simulate daily temperature cycle: peak at 14:00, low at 2:00
    temp = 24.0 + 6.0 * math.sin(2 * math.pi * (hour - 8) / 24) + random.gauss(0, 1.0)
    humidity = 55.0 + 15.0 * math.sin(2 * math.pi * (hour - 16) / 24) + random.gauss(0, 5.0)
    light = max(0, 800 * math.sin(2 * math.pi * (hour - 6) / 24)) + random.gauss(0, 50)
    motion = 1 if random.random() < 0.3 else 0

    return {
        "temperature": round(temp, 1),
        "humidity": round(max(0, min(100, humidity)), 1),
        "light": round(max(0, min(2000, light)), 1),
        "motion": motion,
        "hour": round(hour % 24, 1),
    }


# ==================== Serial Communication ====================

def send_snapshot(ser, snapshot: dict, dry_run: bool = False) -> bool:
    """Send one sensor snapshot to ESP32 via serial CLI 'sensor' command."""
    # v6 fix: esp_console 会把带空格的 JSON 拆成多参数并剥掉引号，
    # 必须用紧凑格式（无空格）让整个 JSON 成为单个参数。
    json_str = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    cmd = f"sensor {json_str}\r\n"

    if dry_run:
        print(f"  [DRY RUN] {cmd.strip()[:100]}...")
        return True

    try:
        ser.write(cmd.encode("utf-8"))
        # Read response
        time.sleep(0.3)
        response = b""
        while ser.in_waiting:
            response += ser.read(ser.in_waiting)
            time.sleep(0.1)
        return True
    except Exception as e:
        print(f"  [ERROR] Serial write failed: {e}")
        return False


def interactive_loop(ser):
    """Interactive mode: type sensor JSON manuellly, see responses."""
    print("\n=== Interactive Sensor Injection ===")
    print("Type JSON and press Enter to inject. Examples:")
    print('  {"temperature":23.5,"humidity":45,"light":300}')
    print('  {"temperature":30.1,"humidity":60,"light":800,"co2":1200}')
    print("Type 'auto' to run 10 UCI snapshots automatically.")
    print("Type 'quit' to exit.\n")

    uci_data = None
    while True:
        try:
            line = input("sensor> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        if line.lower() == "auto":
            if uci_data is None:
                try:
                    uci_data = load_uci_snapshots()
                except SystemExit:
                    continue
            print(f"  Sending 10 UCI snapshots...")
            for i, snap in enumerate(uci_data[:10]):
                print(f"  [{i+1}/10] {json.dumps(snap, ensure_ascii=False)[:80]}...")
                send_snapshot(ser, snap)
                time.sleep(2)
            continue

        # Parse as JSON
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"  Invalid JSON: {line}")
            continue

        ok = send_snapshot(ser, data)
        if not ok:
            print("  [WARN] Write failed, check connection")


def main():
    parser = argparse.ArgumentParser(
        description="DistillToMCU — USB Serial Sensor Injector")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (e.g. COM5, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument("--dataset", type=str, default="uci",
                        choices=["uci", "casas", "synth", "custom"],
                        help="Dataset to replay (default: uci)")
    parser.add_argument("--custom-file", type=str, default=None,
                        help="Path to custom JSONL file (for --dataset custom)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Delay between snapshots in seconds (default: 5)")
    parser.add_argument("--batch", action="store_true",
                        help="Send all snapshots without delay (batch mode)")
    parser.add_argument("--max", type=int, default=None,
                        help="Max snapshots to send (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--list", action="store_true",
                        help="List available serial ports and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without sending (debug)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode: type JSON manually")

    args = parser.parse_args()

    # --list
    if args.list:
        list_ports()
        return

    # Random seed
    random.seed(args.seed)

    # Find port
    port = args.port
    if port is None:
        port = find_serial_port()
        if port is None and not args.dry_run:
            print("[ERROR] No serial port found. Use --port or --list.")
            sys.exit(1)

    # Load dataset
    if args.dataset == "uci":
        snapshots = load_uci_snapshots()
    elif args.dataset == "casas":
        snapshots = load_casas_snapshots()
    elif args.dataset == "synth":
        n = args.max or 100
        snapshots = [generate_synthetic_snapshot(i) for i in range(n)]
        print(f"[INFO] Generated {n} synthetic snapshots")
    elif args.dataset == "custom":
        if not args.custom_file or not os.path.exists(args.custom_file):
            print("[ERROR] --custom-file required for custom dataset")
            sys.exit(1)
        with open(args.custom_file, encoding="utf-8") as f:
            snapshots = [json.loads(l) for l in f if l.strip()]
        print(f"[INFO] Loaded {len(snapshots)} custom snapshots")
    else:
        snapshots = []
        print("[ERROR] Unknown dataset")
        sys.exit(1)

    # Trim
    if args.max and args.max < len(snapshots):
        snapshots = snapshots[:args.max]

    print(f"[INFO] Will send {len(snapshots)} snapshots to {port}")
    print(f"[INFO] Interval: {'batch (no delay)' if args.batch else f'{args.interval}s'}")
    print()

    if args.dry_run:
        # Dry run: just print, no serial
        print(f"=== DRY RUN ({len(snapshots)} snapshots) ===\n")
        for i, snap in enumerate(snapshots[:10]):
            send_snapshot(None, snap, dry_run=True)
        if len(snapshots) > 10:
            print(f"  ... and {len(snapshots) - 10} more snapshots")
        return

    # Open serial
    import serial
    try:
        ser = serial.Serial(port, args.baud, timeout=2)
        # Wait for ESP32 to boot
        print("[INFO] Waiting for ESP32 to boot...")
        time.sleep(3)
        # Flush boot messages
        while ser.in_waiting:
            boot_line = ser.readline()
            try:
                print(f"  [BOOT] {boot_line.decode('utf-8', errors='replace').strip()}")
            except Exception:
                pass
            time.sleep(0.1)

        if args.interactive:
            interactive_loop(ser)
            ser.close()
            return

        # Batch injection loop
        print(f"\n=== Starting injection ({len(snapshots)} snapshots) ===\n")
        success = 0
        failed = 0

        for i, snap in enumerate(snapshots):
            # Clean snapshot: remove non-sensor keys that the ESP32 can handle
            # Keep only numeric sensor fields
            sensor_only = {
                k: v for k, v in snap.items()
                if isinstance(v, (int, float, bool)) and not isinstance(v, bool)
            }
            if not sensor_only:
                sensor_only = snap  # fallback to original

            ok = send_snapshot(ser, sensor_only)

            if ok:
                success += 1
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [{i+1}/{len(snapshots)}] {json.dumps(sensor_only, ensure_ascii=False)[:80]}... [OK]")
            else:
                failed += 1
                print(f"  [{i+1}/{len(snapshots)}] FAILED")

            if not args.batch:
                time.sleep(args.interval)

        print(f"\n=== Done ===")
        print(f"  Sent: {success}, Failed: {failed}")

        ser.close()

    except serial.SerialException as e:
        print(f"[ERROR] Cannot open {port}: {e}")
        print("Check: 1) ESP32 is plugged in  2) Correct port  3) No other program using port")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Closing serial...")
        if 'ser' in locals():
            ser.close()


if __name__ == "__main__":
    main()
