"""
DistillToMCU — WebSocket 对话链路测试（纯标准库，无第三方依赖）
================================================================
模拟 H5 仪表盘的 AI 对话通道：
  1. 连接 ESP32 WS 服务（默认 192.168.2.6:18789）
  2. 发送 {"type":"chat","content":"..."}
  3. 打印返回的 chat 消息（mode/reply/action/latency）

用法：python tools/ws_chat_test.py [--ip 192.168.2.6] [--port 18789]
"""

import argparse
import base64
import json
import os
import socket
import struct
import sys


def ws_connect(ip, port):
    s = socket.create_connection((ip, port), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET / HTTP/1.1\r\n"
        f"Host: {ip}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    )
    s.sendall(req.encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        chunk = s.recv(4096)
        if not chunk:
            raise ConnectionError("handshake failed")
        resp += chunk
    if b" 101 " not in resp.split(b"\r\n", 1)[0]:
        raise ConnectionError(resp.decode("utf-8", "replace")[:200])
    return s


def ws_send_text(s, text):
    payload = text.encode("utf-8")
    mask = os.urandom(4)
    header = bytearray([0x81])  # FIN + text
    n = len(payload)
    if n < 126:
        header.append(0x80 | n)
    elif n < 65536:
        header.append(0x80 | 126)
        header += struct.pack(">H", n)
    else:
        header.append(0x80 | 127)
        header += struct.pack(">Q", n)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    s.sendall(bytes(header) + mask + masked)


def ws_recv_frame(s):
    hdr = _recv_exact(s, 2)
    opcode = hdr[0] & 0x0F
    length = hdr[1] & 0x7F
    masked = hdr[1] & 0x80
    if length == 126:
        length = struct.unpack(">H", _recv_exact(s, 2))[0]
    elif length == 127:
        length = struct.unpack(">Q", _recv_exact(s, 8))[0]
    mask = _recv_exact(s, 4) if masked else None
    payload = _recv_exact(s, length)
    if mask:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return opcode, payload


def _recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.2.6")
    ap.add_argument("--port", type=int, default=18789)
    ap.add_argument("--pre-rules", action="store_true",
                    help="先发送 rules 请求（模拟仪表盘连接行为，回归测试崩溃修复）")
    ap.add_argument("--with-hist", action="store_true",
                    help="首条消息携带模拟会话历史（多会话上下文回归）")
    ap.add_argument("--sid", default=None,
                    help="附带会话 ID，验证固件原样回显 sid")
    ap.add_argument("msgs", nargs="*", default=[
        "开灯",                # -> led.on（红灯）
        "关掉它",              # -> led.off（上下文指代灯）
        "风扇开到最大",        # -> fan speed 3（绿灯最亮）
        "调低一档",            # -> fan speed 2（依赖设备状态）
        "窗帘开到30%",         # -> curtain position 30（蓝灯 30%）
    ])
    args = ap.parse_args()

    s = ws_connect(args.ip, args.port)
    print(f"[OK] connected to ws://{args.ip}:{args.port}")
    if args.pre_rules:
        print("\n>>> request rules (dashboard connect behavior)")
        ws_send_text(s, json.dumps({"type": "message", "content": "rules"}))
        s.settimeout(5)
        try:
            op, payload = ws_recv_frame(s)
            if op == 1:
                m = json.loads(payload.decode("utf-8"))
                print(f"  [rules] received {len(m.get('data', []))} rules")
        except Exception as e:
            print(f"  [!!] rules request failed: {e}")
        s.settimeout(10)
    for text in args.msgs:
        print(f"\n>>> chat: {text}")
        payload = {"type": "chat", "content": text}
        if args.sid:
            payload["sid"] = args.sid
        if args.with_hist and text == args.msgs[0]:
            payload["history"] = [
                {"role": "user", "content": "把灯关掉"},
                {"role": "assistant", "content": "好的，已关闭灯。"},
            ]
            print("  [hist] 附带 2 条历史消息")
        ws_send_text(s, json.dumps(payload, ensure_ascii=False))
        deadline = 30
        got = False
        while deadline > 0:
            import time
            s.settimeout(min(deadline, 10))
            try:
                op, payload = ws_recv_frame(s)
            except socket.timeout:
                break
            if op == 1:
                try:
                    m = json.loads(payload.decode("utf-8"))
                except Exception:
                    print("  [frame] " + payload.decode("utf-8", "replace")[:200])
                    continue
                if m.get("type") == "chat":
                    d = m.get("data", {})
                    print(f"  [chat] mode={d.get('mode')} action={d.get('action')} "
                          f"lat={d.get('latency_us')}us rule={d.get('rule_id')} sid={d.get('sid')}")
                    print(f"  [reply] {d.get('reply', '(none)')}")
                    got = True
                    break
                elif m.get("type") == "response":
                    print(f"  [resp] {json.dumps(m.get('data', {}), ensure_ascii=False)}")
                else:
                    print(f"  [{m.get('type')}] {payload.decode('utf-8','replace')[:150]}")
            deadline -= 1
        if not got:
            print("  [!!] no chat reply within timeout")
    s.close()


if __name__ == "__main__":
    main()
