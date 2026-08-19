"""
DistillToMCU — 串口探测/验证脚本
================================
烧录后自动验证：
  1. 启动日志（SPIFFS 挂载、规则加载、WiFi 连接、LED 自检）
  2. LED 自检命令（ledtest）
  3. WS2812 直接控制（ws r g b）
  4. 规则列表（rules）
  5. 注入一条 UCI 风格传感器数据，验证规则匹配闭环（stats 应出现 local）

用法：python tools/serial_probe.py --port COM6
"""

import argparse
import json
import sys
import time

import serial


def read_until(ser, timeout=1.0):
    end = time.time() + timeout
    buf = b""
    while time.time() < end:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
        else:
            time.sleep(0.05)
    return buf.decode("utf-8", errors="replace")


def send_cmd(ser, cmd, wait=1.2):
    print(f"\n>>> {cmd}")
    ser.write((cmd + "\r\n").encode("utf-8"))
    time.sleep(0.3)
    out = read_until(ser, wait)
    print(out.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=2)
    ser.reset_input_buffer()
    print(f"[INFO] Connected to {args.port}, reading boot log...")
    time.sleep(1.0)
    boot = read_until(ser, 3.0)
    print(boot)

    send_cmd(ser, "ledtest")
    send_cmd(ser, "ws 255 0 0")
    time.sleep(0.5)
    send_cmd(ser, "ws 0 255 0")
    time.sleep(0.5)
    send_cmd(ser, "ws 0 0 255")
    time.sleep(0.5)
    send_cmd(ser, "ws 0 0 0")
    send_cmd(ser, "rules")
    send_cmd(ser, "stats")
    send_cmd(ser, "heap")

    # 注入一条符合预置规则（co2 高 → fan on → 绿灯）的传感器数据
    snap = {
        "temperature": 22.0, "humidity": 27.0, "light": 400.0,
        "co2": 1200.0, "motion": 1, "hour": 14.0, "temp_trend": 0.0,
    }
    # v6 fix: 紧凑 JSON（无空格），否则 esp_console 会拆分参数并剥掉引号
    send_cmd(ser, f"sensor {json.dumps(snap, separators=(',', ':'))}", wait=0.8)
    print("\n[INFO] Waiting for agent loop to process (rule match should turn FAN=GREEN)...")
    time.sleep(4.0)
    send_cmd(ser, "stats")
    send_cmd(ser, "rules")

    ser.close()
    print("\n[INFO] Probe done. Expected: boot self-test colors, ws colors, "
          "5 preloaded rules, stats local>0 after injection.")


if __name__ == "__main__":
    main()
