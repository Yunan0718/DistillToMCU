"""串口日志监听：抓取 ESP32 串口输出（用于崩溃现场/启动日志）。
用法：python tools/serial_log.py [端口] [秒数] [输出文件]"""

import serial
import sys
import time


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM6"
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    out = sys.argv[3] if len(sys.argv) > 3 else "serial_trace.txt"
    s = serial.Serial(port, 115200, timeout=2)
    s.reset_input_buffer()
    buf = b""
    end = time.time() + secs
    last_flush = time.time()
    while time.time() < end:
        try:
            n = s.in_waiting
            if n:
                buf += s.read(n)
            else:
                time.sleep(0.1)
            if time.time() - last_flush >= 5:
                with open(out, "wb") as f:
                    f.write(buf)
                last_flush = time.time()
        except Exception:
            break
    with open(out, "wb") as f:
        f.write(buf)
    print(f"[serial_log] wrote {len(buf)} bytes to {out}")


if __name__ == "__main__":
    main()
