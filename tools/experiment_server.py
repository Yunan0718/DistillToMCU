#!/usr/bin/env python3
"""
DistillToMCU 本地实验服务 (v7.0)
=================================
让 H5 仪表盘可以直接：导入数据源 → 新建实验 → 自动跑实验（PC 模拟 或 MCU 真机喂送）→
结果自动进仪表盘下拉列表；同时管理"对话实验"（AI 对话会话自动保存 + 统计）。

技术说明：
  - 纯标准库实现（CSV/JSON/XLSX 解析、WebSocket 客户端、DeepSeek 列映射均内置），
    用项目 Python 环境即可运行，无需额外 pip 包。
  - PC 模式：调用 poc/experiment*.py 跑真实 LLM 实验。
  - MCU 模式：通过 WebSocket 把数据源按节奏喂给 ESP32，执行完拉取 trace 计算指标。
  - 端口 18800，只监听 127.0.0.1（本机 H5 访问）。

用法：
    python tools/experiment_server.py [--port 18800]
"""

import argparse
import base64
import csv
import io
import json
import math
import os
import random
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POC = os.path.join(ROOT, "poc")
OUTPUT = os.path.join(POC, "output")
UPLOADS = os.path.join(POC, "data", "uploads")
REGISTRY = os.path.join(OUTPUT, "experiments.json")
CHAT_FILE = os.path.join(OUTPUT, "chat_experiments.json")
FIGURES_DIR = os.path.join(ROOT, "figures")

PORT = 18800

FIGURE_META = {
    "fig_architecture": {
        "title": "系统架构图",
        "desc": "论文 Figure 1：数据源 → PC 蒸馏 → MCU 本地执行 / 云端 LLM 回退的闭环（实线在线流、虚线蒸馏反馈）。",
        "metrics": ["System architecture"],
    },
    "fig_rule_lifecycle": {
        "title": "规则生命周期状态机",
        "desc": "candidate → verified → active → degraded → retired，以及置信度/新鲜度阈值（含负反馈降级与 14 天退役）。",
        "metrics": ["Rule lifecycle"],
    },
    "fig_ar_learning": {
        "title": "自主率增长曲线",
        "desc": "(a) 3 个合成种子的均值±95% CI；(b) STRANDS / UCI V2 / UCI V3。对应 Autonomy Rate。",
        "metrics": ["Autonomy Rate"],
    },
    "fig_baselines": {
        "title": "基线与云调用削减对比",
        "desc": "8 条基线 + DistillToMCU（朱红高亮）在 6 个数据集上的均值±SD。",
        "metrics": ["Autonomy Rate", "Cloud Call Reduction"],
    },
    "fig_latency": {
        "title": "本地 vs 云端延迟",
        "desc": "全部 trace 的延迟分布（对数轴）+ 各数据集 p50。对应 Latency local/cloud。",
        "metrics": ["Latency local p50", "Latency cloud p50"],
    },
    "fig_precision_recall": {
        "title": "Precision / Recall",
        "desc": "系统在各数据集上的规则执行精度与召回。",
        "metrics": ["Precision", "Recall"],
    },
    "fig_rules_size": {
        "title": "规则库规模增长",
        "desc": "蒸馏规则总数随实验天数的增长。对应 Rule Store Size。",
        "metrics": ["Rule Store Size"],
    },
    "fig_nemenyi_cd": {
        "title": "Friedman + Nemenyi 显著性 CD 图",
        "desc": "8 基线 + Ours 的平均秩与临界差（CD=3.031），用于统计显著性。",
        "metrics": ["Statistical significance"],
    },
    "fig_ablations": {
        "title": "消融实验",
        "desc": "蒸馏来源（L1/L2/L3/Full）与规则泛化（Exact vs Inclusive）对 AR 的影响。",
        "metrics": ["Ablation AR"],
    },
    "fig_chat_stats": {
        "title": "对话实验统计",
        "desc": "对话实验的消息数 / 动作数 / 自主率（有对话实验数据时生成）。",
        "metrics": ["Chat experiment stats"],
    },
}

STATE_LOCK = threading.Lock()
JOBS = {}          # exp_id -> job dict
STOP_FLAGS = {}    # exp_id -> threading.Event


# ============================================================
# 基础工具
# ============================================================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def log_job(exp_id, line):
    job = JOBS.get(exp_id)
    if job:
        job["log"].append(f"[{now_str()}] {line}")
        if len(job["log"]) > 500:
            job["log"] = job["log"][-500:]


def get_registry():
    return load_json(REGISTRY, [])


def save_registry(reg):
    save_json(REGISTRY, reg)


def get_chat_experiments():
    return load_json(CHAT_FILE, [])


def save_chat_experiments(chats):
    save_json(CHAT_FILE, chats)


def api_key_available():
    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    if len(key) > 10:
        return True
    try:
        sys.path.insert(0, POC)
        from config import LLM_API_KEY
        return len(LLM_API_KEY) > 10
    except Exception:
        return False


# ============================================================
# 数据源：内置 + 上传 + 列映射（含 AI 辅助）
# ============================================================

HEURISTIC_COLUMNS = {
    "temperature": ["temperature", "temp", "t", "temp_c", "tempc", "airtemp", "air_temp", "ta"],
    "humidity": ["humidity", "humid", "hum", "rh", "relative_humidity", "moisture"],
    "light": ["light", "lux", "illuminance", "luminance", "illum", "light_level", "lightlevel"],
    "co2": ["co2", "co2_ppm", "co2ppm", "carbon_dioxide", "carbondioxide"],
    "motion": ["motion", "movement", "pir", "occupancy", "presence", "motion_detected", "motiondetected"],
    "temp_trend": ["temp_trend", "temperature_trend", "dtemp", "temp_slope"],
    "hour": ["hour", "time_of_day", "time", "time_hour"],
    "user_input": ["user_input", "user", "command", "instruction", "text", "utterance", "query"],
}


def guess_field_for_column(col, sample_values):
    """启发式：列名匹配 + 数值范围兜底。"""
    c = str(col).strip().lower()
    for field, names in HEURISTIC_COLUMNS.items():
        for n in names:
            if c == n or c.endswith("_" + n) or n in c:
                return field
    nums = []
    for v in sample_values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            pass
    if nums:
        lo, hi = min(nums), max(nums)
        if 200 <= lo and hi <= 6000:
            return "co2"
        if -30 <= lo and hi <= 60:
            return "temperature"
        if 0 <= lo and hi <= 100:
            return "humidity"
        if set(int(n) for n in nums) <= {0, 1} and len(nums) >= 2:
            return "motion"
        if lo >= 0 and hi <= 100000:
            return "light"
    return "ignore"


def _parse_csv(text):
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(text[:4096], delimiters=",;\t")
    except Exception:
        dialect = csv.excel
    return list(csv.DictReader(io.StringIO(text), dialect=dialect))


def _parse_xlsx(data):
    """极简 XLSX 解析（stdlib zipfile+XML），取第一个 sheet。"""
    z = zipfile.ZipFile(io.BytesIO(data))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        for si in root.findall("m:si", ns):
            shared.append("".join(t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")))
    sheet = [n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", n)][0]
    root = ET.fromstring(z.read(sheet))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = []
    for row in root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
        cells = {}
        for c in row:
            ref = c.get("r", "")
            col = re.match(r"[A-Z]+", ref)
            if not col:
                continue
            colname = col.group(0)
            v = c.find("m:v", ns)
            is_str = c.get("t") == "s"
            val = v.text if v is not None else ""
            if is_str and val != "":
                val = shared[int(val)]
            cells[colname] = val
        rows.append(cells)
    if not rows:
        return []
    maxcol = max(len(r) for r in rows)
    headers = []
    for i in range(maxcol):
        col = chr(ord("A") + i)
        headers.append(str(rows[0].get(col, "")))
    out = []
    for r in rows[1:]:
        d = {}
        for i, h in enumerate(headers):
            col = chr(ord("A") + i)
            d[h] = r.get(col, "")
        out.append(d)
    return out


def parse_upload(filename, data):
    """解析上传文件 → (rows, columns, sample)。返回 dict。"""
    name = filename or "upload"
    ext = os.path.splitext(name)[1].lower()
    try:
        if ext in (".xlsx", ".xlsm"):
            rows = _parse_xlsx(data)
        elif ext == ".json":
            obj = json.loads(data.decode("utf-8-sig"))
            if isinstance(obj, dict) and "sensors" in obj:
                obj = [obj]
            if isinstance(obj, dict):
                # 可能是 {records:[...]} 或列式
                for k in ("records", "data", "rows", "snapshots"):
                    if isinstance(obj.get(k), list):
                        obj = obj[k]
                        break
            if isinstance(obj, dict):
                obj = [dict(zip(obj.keys(), vals)) for vals in zip(*obj.values())]
            rows = obj if isinstance(obj, list) else []
        else:
            text = data.decode("utf-8-sig", errors="replace")
            rows = _parse_csv(text)
    except Exception as e:
        return {"ok": False, "error": f"解析失败: {e}"}
    if not rows:
        return {"ok": False, "error": "文件为空或没有数据行"}
    # 嵌套 sensors 展开
    expanded = []
    for r in rows:
        if isinstance(r, dict) and isinstance(r.get("sensors"), dict):
            merged = dict(r)
            merged.pop("sensors", None)
            merged.update(r["sensors"])
            r = merged
        if isinstance(r, dict):
            expanded.append({str(k): v for k, v in r.items()})
    rows = expanded
    columns = list(rows[0].keys())
    samples = rows[:5]
    return {
        "ok": True,
        "n_rows": len(rows),
        "columns": columns,
        "samples": samples,
        "rows": rows,
    }


def heuristic_mapping(columns, rows):
    mapping = {}
    for col in columns:
        vals = [r.get(col) for r in rows[:30]]
        mapping[col] = guess_field_for_column(col, vals)
    return mapping


def ai_mapping(columns, rows):
    """用 DeepSeek 做列映射（用户要求"用 AI 进行会不会更好"）。
    API Key 缺失时回退启发式。"""
    mapping = heuristic_mapping(columns, rows)
    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    if len(key) <= 10:
        try:
            sys.path.insert(0, POC)
            from config import LLM_API_KEY
            key = LLM_API_KEY
        except Exception:
            key = ""
    if len(key) <= 10:
        return {"mapping": mapping, "ai_used": False, "reason": "no_api_key"}
    sample = {}
    for c in columns[:12]:
        sample[c] = [r.get(c) for r in rows[:4]]
    prompt = (
        "You map sensor dataset columns to standard fields. "
        "Allowed target fields: temperature, humidity, light, co2, motion, "
        "temp_trend, hour, user_input, ignore.\n"
        f"Columns and sample values:\n{json.dumps(sample, ensure_ascii=False)}\n"
        "Reply ONLY with a JSON object mapping each column name to a target field."
    )
    body = json.dumps({
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 600,
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            obj = json.loads(resp.read().decode())
        content = obj["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, re.S)
        ai_map = json.loads(m.group(0)) if m else {}
        for col, f in ai_map.items():
            if col in mapping and f in HEURISTIC_COLUMNS:
                mapping[col] = f
        return {"mapping": mapping, "ai_used": True, "reason": "ok"}
    except Exception as e:
        return {"mapping": mapping, "ai_used": False, "reason": str(e)}


def apply_mapping(rows, mapping):
    """按映射把原始行转成标准 snapshots：[{sensors:{...}, user_input:...}]"""
    snaps = []
    for r in rows:
        sensors = {}
        user_input = ""
        for col, field in mapping.items():
            if field in ("ignore", None):
                continue
            val = r.get(col)
            if val is None or str(val).strip() == "":
                continue
            if field == "user_input":
                user_input = str(val)
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            if field == "motion":
                num = 1.0 if num > 0 else 0.0
            if field == "hour":
                num = float(int(num) % 24)
            sensors[field] = num
        if sensors:
            snaps.append({"sensors": sensors, "user_input": user_input})
    return snaps


def save_upload_dataset(upload_id, rows, mapping, meta):
    d = os.path.join(UPLOADS, upload_id)
    os.makedirs(d, exist_ok=True)
    snaps = apply_mapping(rows, mapping)
    with open(os.path.join(d, "snapshots.jsonl"), "w", encoding="utf-8") as f:
        for s in snaps:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    save_json(os.path.join(d, "meta.json"), {
        "upload_id": upload_id,
        "created_at": now_str(),
        "n_rows": len(snaps),
        "mapping": mapping,
        "original": meta.get("original", ""),
    })
    return len(snaps)


def list_datasets():
    items = [
        {"id": "synthetic", "label": "合成正弦波 (Synthetic)", "builtin": True},
        {"id": "uci", "label": "UCI Occupancy (真实传感器)", "builtin": True},
        {"id": "casas", "label": "CASAS Aruba-1 (真实活动序列)", "builtin": True},
    ]
    if os.path.isdir(UPLOADS):
        for d in sorted(os.listdir(UPLOADS)):
            meta = load_json(os.path.join(UPLOADS, d, "meta.json"))
            if meta:
                items.append({
                    "id": "upload:" + d,
                    "label": f"上传: {meta.get('original', d)} ({meta.get('n_rows', 0)} 行)",
                    "builtin": False,
                    "n_rows": meta.get("n_rows", 0),
                })
    return items


def load_snapshots(dataset_id, seed=42, count=0):
    """返回 [{sensors:{...}, user_input:str}]"""
    sensor_fields = {"temperature", "humidity", "light", "co2", "motion",
                     "temp_trend", "hour"}
    if dataset_id == "synthetic":
        snaps = []
        rnd = random.Random(seed)
        n = count or 500
        for i in range(n):
            hour = (i * 7) % 24
            base = 22 + 4 * math.sin(2 * math.pi * (hour - 8) / 24)
            snaps.append({
                "sensors": {
                    "temperature": round(base + rnd.uniform(-0.8, 0.8), 2),
                    "humidity": round(50 + 12 * math.sin(2 * math.pi * hour / 24) + rnd.uniform(-4, 4), 2),
                    "light": round(max(0, 420 + 380 * math.sin(2 * math.pi * (hour - 7) / 24) + rnd.uniform(-30, 30)), 2),
                    "motion": 1 if (8 <= hour <= 22 and rnd.random() < 0.55) else 0,
                    "co2": round(520 + 180 * math.sin(2 * math.pi * (hour - 9) / 24) + rnd.uniform(-40, 40), 2),
                    "hour": hour,
                    "temp_trend": round(rnd.uniform(-0.05, 0.05), 3),
                },
                "user_input": "",
            })
        return snaps
    if dataset_id == "uci":
        path = os.path.join(POC, "data", "uci", "snapshots_enriched.json")
        data = load_json(path, [])
        return [{"sensors": {k: v for k, v in s.items() if k in sensor_fields},
                 "user_input": s.get("user_input", "")} for s in data]
    if dataset_id == "casas":
        path = os.path.join(POC, "data", "casas", "aruba_snapshots.json")
        data = load_json(path, [])
        act_map = {
            "Sleeping": "准备睡觉了", "Meal_Preparation": "要做饭了", "Eating": "吃饭了",
            "Relax": "想放松一下", "Housekeeping": "该打扫了", "Wash_Dishes": "该洗碗了",
            "Work": "开始工作", "Enter_Home": "我回来了", "Leave_Home": "出门了",
        }
        out = []
        for s in data:
            act = s.get("activity", "")
            ui = act_map.get(act, "")
            out.append({"sensors": dict(s.get("sensors", {})), "user_input": ui})
        return out
    if dataset_id.startswith("upload:"):
        uid = dataset_id.split(":", 1)[1]
        path = os.path.join(UPLOADS, uid, "snapshots.jsonl")
        out = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        return out
    return []


def sample_evenly(snaps, days, cap_per_day=20):
    """MCU/自定义实验的行数控制：均匀抽样，总行数 ≤ days×cap_per_day，
    避免大数据源在 MCU 上跑十几个小时。"""
    n = len(snaps)
    cap = max(days, days * cap_per_day)
    if n <= cap:
        return snaps
    idx = [int(i * (n - 1) / (cap - 1)) for i in range(cap)] if cap > 1 else [0]
    return [snaps[i] for i in idx]


# ============================================================
# WebSocket 客户端（纯标准库）
# ============================================================

def ws_connect(ip, port, timeout=10):
    s = socket.create_connection((ip, port), timeout=timeout)
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET / HTTP/1.1\r\nHost: {ip}:{port}\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )
    s.sendall(req.encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        chunk = s.recv(4096)
        if not chunk:
            raise ConnectionError("WS handshake failed")
        resp += chunk
    if b" 101 " not in resp.split(b"\r\n", 1)[0]:
        raise ConnectionError(resp.decode("utf-8", "replace")[:200])
    return s


def ws_send_text(s, text):
    payload = text.encode("utf-8")
    mask = os.urandom(4)
    header = bytearray([0x81])
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


def _recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


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


def ws_wait_for(s, wanted_type, timeout=60):
    """读帧直到出现 wanted_type，返回其 data。"""
    s.settimeout(timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        op, payload = ws_recv_frame(s)
        if op != 1:
            continue
        try:
            m = json.loads(payload.decode("utf-8", errors="replace"))
        except Exception:
            continue
        if m.get("type") == wanted_type:
            return m.get("data")
    return None


# ============================================================
# 指标计算（PC trace / MCU trace 统一口径）
# ============================================================

def percentile(vals, p):
    if not vals:
        return 0.0
    sv = sorted(vals)
    k = (len(sv) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return round(sv[lo], 2)
    return round(sv[lo] + (sv[hi] - sv[lo]) * (k - lo), 2)


def metrics_from_traces(traces, days):
    """按交互数均分天，输出 metrics.jsonl 兼容行。"""
    n = len(traces)
    per_day = max(1, math.ceil(n / max(1, days)))
    rows = []
    for d in range(days):
        chunk = traces[d * per_day:(d + 1) * per_day]
        if not chunk:
            rows.append({
                "day": d + 1, "autonomy_rate": 0.0, "cloud_calls": 0,
                "local_calls": 0, "total": 0, "avg_local_lat_ms": 0.0,
                "avg_cloud_lat_ms": 0.0, "active_rules": 0, "total_rules": 0,
                "new_rules_today": 0,
            })
            continue
        local = [t for t in chunk if t.get("exec_mode") == "local"]
        cloud = [t for t in chunk if t.get("exec_mode") == "cloud"]
        ll = [t.get("latency_ms", 0) for t in local]
        cl = [t.get("latency_ms", 0) for t in cloud]
        ar = round(len(local) / len(chunk) * 100, 1)
        rows.append({
            "day": d + 1,
            "autonomy_rate": ar,
            "cloud_calls": len(cloud),
            "local_calls": len(local),
            "total": len(chunk),
            "avg_local_lat_ms": round(sum(ll) / max(1, len(ll)), 1),
            "avg_cloud_lat_ms": round(sum(cl) / max(1, len(cl)), 1),
            "active_rules": 0,
            "total_rules": 0,
            "new_rules_today": 0,
        })
    return rows


def read_output(exp_id):
    od = os.path.join(OUTPUT, exp_id)
    metrics = []
    path = os.path.join(od, "metrics.jsonl")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            metrics = [json.loads(l) for l in f if l.strip()]
    rules = load_json(os.path.join(od, "rules_snapshot.json"), [])
    traces = []
    tpath = os.path.join(od, "traces.jsonl")
    if os.path.exists(tpath):
        with open(tpath, "r", encoding="utf-8") as f:
            traces = [json.loads(l) for l in f if l.strip()]
    return {"metrics": metrics, "rules": rules, "traces": traces}


def summary_from_output(exp_id):
    d = read_output(exp_id)
    m = d["metrics"]
    last = m[-1] if m else {}
    total = sum(x.get("total", 0) for x in m)
    local = sum(x.get("local_calls", 0) for x in m)
    cloud = sum(x.get("cloud_calls", 0) for x in m)
    return {
        "ar": last.get("autonomy_rate", 0.0),
        "total": total,
        "local": local,
        "cloud": cloud,
        "days": len(m),
        "active_rules": last.get("active_rules", 0),
        "total_rules": len(d["rules"]),
        "traces": len(d["traces"]),
    }


# ============================================================
# 实验运行器：PC 模拟
# ============================================================

def run_pc_experiment(exp):
    exp_id = exp["id"]
    od = os.path.join(OUTPUT, exp_id)
    os.makedirs(od, exist_ok=True)
    seed = exp.get("seed", 42)
    days = exp.get("days", 30)
    kind = exp["type"]
    log_job(exp_id, f"启动 PC 实验 [{kind}] seed={seed} days={days}")
    try:
        if kind == "synthetic":
            cmd = [sys.executable, os.path.join(POC, "experiment.py"),
                   "--real", "--seed", str(seed), "--days", str(days),
                   "--output-dir", od]
        elif kind == "uci":
            cmd = [sys.executable, os.path.join(POC, "experiment_uci.py"),
                   "--seed", str(seed), "--days", str(days), "--output-dir", od]
        elif kind == "casas":
            cmd = [sys.executable, os.path.join(POC, "experiment_strands.py"),
                   "--seed", str(seed), "--days", str(days), "--output-dir", od]
        elif kind == "upload":
            snaps = load_snapshots(exp["dataset"])
            snaps = sample_evenly(snaps, days)
            spath = os.path.join(od, "snapshots.jsonl")
            with open(spath, "w", encoding="utf-8") as f:
                for s in snaps:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            log_job(exp_id, f"自定义数据源已抽样 {len(snaps)} 条（≤ {days} 天 × 20/天）")
            cmd = [sys.executable, os.path.join(POC, "experiment_custom.py"),
                   "--data", spath, "--seed", str(seed), "--days", str(days),
                   "--output-dir", od]
        else:
            raise RuntimeError(f"未知实验类型: {kind}")
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log_job(exp_id, line)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"实验进程退出码 {proc.returncode}")
        job = JOBS.get(exp_id)
        if job:
            job["progress"] = 100
        finish_experiment(exp_id, "done")
    except Exception as e:
        log_job(exp_id, f"错误: {e}")
        finish_experiment(exp_id, "error", str(e))


def finish_experiment(exp_id, status, error=None):
    reg = get_registry()
    for e in reg:
        if e["id"] == exp_id:
            e["status"] = status
            e["error"] = error
            e["finished_at"] = now_str()
            if status == "done":
                e["summary"] = summary_from_output(exp_id)
                e["progress"] = 100
            save_registry(reg)
            break
    with STATE_LOCK:
        STOP_FLAGS.pop(exp_id, None)
    job = JOBS.get(exp_id)
    if job:
        job["status"] = status


# ============================================================
# 实验运行器：MCU 真机（分块喂送）
# ============================================================

def run_mcu_experiment(exp):
    exp_id = exp["id"]
    ip = exp.get("board_ip", "192.168.2.6")
    port = int(exp.get("board_port", 18789))
    pace = float(exp.get("pace_s", 2.5))
    days = exp.get("days", 30)
    snaps = load_snapshots(exp["dataset"])
    snaps = sample_evenly(snaps, days)
    if not snaps:
        finish_experiment(exp_id, "error", "数据源为空")
        return
    od = os.path.join(OUTPUT, exp_id)
    os.makedirs(od, exist_ok=True)
    log_job(exp_id, f"连接板子 {ip}:{port}，共 {len(snaps)} 条快照，每 {pace}s 喂一条")
    try:
        s = ws_connect(ip, port)
        ws_send_text(s, json.dumps({"type": "trace_clear"}))
        time.sleep(0.5)
        n = len(snaps)
        try:
            for i, snap in enumerate(snaps):
                if STOP_FLAGS.get(exp_id) and STOP_FLAGS[exp_id].is_set():
                    log_job(exp_id, f"收到停止，已喂 {i}/{n} 条")
                    break
                ws_send_text(s, json.dumps(
                    {"type": "sensor", "sensors": snap.get("sensors", {})},
                    ensure_ascii=False))
                job = JOBS.get(exp_id)
                if job:
                    job["progress"] = round((i + 1) / n * 90, 1)
                time.sleep(pace)
            # 轮询等待板子处理完（每条真实 LLM 决策约 5-6s，不能喂完立即拉）
            fed = n if not (STOP_FLAGS.get(exp_id) and STOP_FLAGS[exp_id].is_set()) else i
            traces = []
            stable = 0
            last_count = -1
            deadline = time.time() + max(90, fed * 6 + 30)
            while time.time() < deadline:
                try:
                    ws_send_text(s, json.dumps({"type": "trace_read"}))
                    t = ws_wait_for(s, "trace_data", timeout=15) or []
                    if isinstance(t, list):
                        traces = t
                except Exception:
                    pass
                job = JOBS.get(exp_id)
                if job:
                    job["progress"] = round(90 + min(10, len(traces) / max(1, fed) * 10), 1)
                if len(traces) >= fed:
                    break
                if len(traces) == last_count:
                    stable += 1
                    if stable >= 3:
                        log_job(exp_id, f"板子处理进度稳定：{len(traces)}/{fed} 条，结束等待")
                        break
                else:
                    stable = 0
                    last_count = len(traces)
                time.sleep(10)
            if len(traces) < fed:
                log_job(exp_id, f"等待超时：仅收集到 {len(traces)}/{fed} 条 trace")
            with open(os.path.join(od, "traces.jsonl"), "w", encoding="utf-8") as f:
                for t in traces:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            # 拉取规则快照
            ws_send_text(s, json.dumps({"type": "message", "content": "rules"}))
            rules = ws_wait_for(s, "rules", timeout=10) or []
            save_json(os.path.join(od, "rules_snapshot.json"), rules)
            # 计算指标
            metrics = metrics_from_traces(traces, days)
            with open(os.path.join(od, "metrics.jsonl"), "w", encoding="utf-8") as f:
                for m in metrics:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            log_job(exp_id, f"完成：{len(traces)} 条 trace，{len(rules)} 条规则")
            finish_experiment(exp_id, "done")
        finally:
            try:
                s.close()
            except Exception:
                pass
    except Exception as e:
        log_job(exp_id, f"MCU 实验错误: {e}")
        finish_experiment(exp_id, "error", str(e))


def start_job(exp):
    exp_id = exp["id"]
    STOP_FLAGS[exp_id] = threading.Event()
    job = {"status": "running", "progress": 0, "log": [f"[{now_str()}] 实验已创建"]}
    JOBS[exp_id] = job
    target = run_pc_experiment if exp.get("target") == "pc" else run_mcu_experiment
    th = threading.Thread(target=target, args=(exp,), daemon=True)
    job["thread"] = th
    th.start()


def create_experiment(payload):
    with STATE_LOCK:
        exp_id = "exp_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(random.randint(100, 999))
    ds = payload.get("dataset", "synthetic")
    exp_type = "upload" if str(ds).startswith("upload:") else str(ds)
    exp = {
        "id": exp_id,
        "name": (payload.get("name") or "未命名实验").strip(),
        "kind": "normal",
        "type": exp_type,
        "dataset": ds,
        "target": payload.get("target", "pc"),
        "seed": int(payload.get("seed", 42)),
        "days": int(payload.get("days", 30)),
        "pace_s": float(payload.get("pace_s", 2.5)),
        "board_ip": payload.get("board_ip", "192.168.2.6"),
        "board_port": int(payload.get("board_port", 18789)),
        "status": "running",
        "created_at": now_str(),
        "progress": 0,
    }
    # 预估时长
    n_snaps = len(sample_evenly(load_snapshots(exp["dataset"], seed=exp["seed"]), exp["days"]))
    if exp["target"] == "mcu":
        exp["estimate_min"] = round(n_snaps * exp["pace_s"] / 60, 1)
    else:
        exp["estimate_min"] = round(exp["days"] * 0.6, 1)
    reg = get_registry()
    reg.insert(0, exp)
    save_registry(reg)
    start_job(exp)
    return exp


# ============================================================
# 对话实验：自动保存 + 统计（口径参考普通实验）
# ============================================================

def parse_meta(meta):
    """从 H5 的 meta 字符串提取 mode/latency/action/rule。"""
    out = {"mode": None, "latency_ms": None, "action": None, "rule": None}
    if not meta:
        return out
    txt = str(meta)
    m = re.search(r"(local|cloud)", txt)
    if m:
        out["mode"] = m.group(1)
    m = re.search(r"(\d+(?:\.\d+)?)\s*ms", txt)
    if m:
        out["latency_ms"] = float(m.group(1))
    m = re.search(r"action\s+([a-z_\.]+)", txt)
    if m:
        out["action"] = m.group(1)
    m = re.search(r"rule\s+(\S+)", txt)
    if m:
        out["rule"] = m.group(1)
    return out


def chat_stats(msgs):
    ai = [m for m in msgs if m.get("role") == "ai"]
    me = [m for m in msgs if m.get("role") == "me"]
    modes = [parse_meta(m.get("meta", "")) for m in ai]
    local = sum(1 for x in modes if x["mode"] == "local")
    cloud = sum(1 for x in modes if x["mode"] == "cloud")
    actions = [x["action"] for x in modes if x["action"]]
    lats = [x["latency_ms"] for x in modes if x["latency_ms"]]
    devs = {"led": 0, "fan": 0, "curtain": 0}
    for a in actions:
        d = a.split(".")[0]
        if d in devs:
            devs[d] += 1
    decided = local + cloud
    return {
        "messages": len(msgs),
        "user_messages": len(me),
        "ai_messages": len(ai),
        "local": local,
        "cloud": cloud,
        "autonomy_rate": round(local / max(1, decided) * 100, 1),
        "action_count": len(actions),
        "actions_by_device": devs,
        "avg_latency_ms": round(sum(lats) / max(1, len(lats)), 1),
        "p50_latency_ms": percentile(lats, 50),
        "p95_latency_ms": percentile(lats, 95),
    }


def upsert_chat_experiment(payload):
    chats = get_chat_experiments()
    cid = payload.get("id") or ("chat_" + str(int(time.time() * 1000)))
    entry = {
        "id": cid,
        "title": (payload.get("title") or "新对话").strip()[:40],
        "msgs": payload.get("msgs") or [],
        "updated_at": now_str(),
    }
    entry["stats"] = chat_stats(entry["msgs"])
    found = False
    for i, c in enumerate(chats):
        if c.get("id") == cid:
            chats[i] = entry
            found = True
            break
    if not found:
        entry["created_at"] = now_str()
        chats.insert(0, entry)
    save_chat_experiments(chats)
    return entry


# ============================================================
# HTTP 服务
# ============================================================

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(200, {"ok": True})

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path == "/api/health":
                self._send(200, {
                    "ok": True, "version": "7.0", "port": PORT,
                    "has_api_key": api_key_available(),
                    "python": sys.executable,
                })
            elif path == "/api/datasets":
                self._send(200, {"ok": True, "datasets": list_datasets()})
            elif path == "/api/experiments":
                reg = get_registry()
                out = []
                for e in reg:
                    item = dict(e)
                    item.pop("thread", None)
                    job = JOBS.get(e["id"])
                    if job:
                        item["log"] = job["log"][-30:]
                        item["progress"] = job.get("progress", item.get("progress", 0))
                    out.append(item)
                self._send(200, {"ok": True, "experiments": out})
            elif path == "/api/chat-experiments":
                self._send(200, {"ok": True, "experiments": get_chat_experiments()})
            elif path.startswith("/api/experiments/"):
                exp_id = path.split("/")[-1]
                reg = get_registry()
                exp = next((e for e in reg if e["id"] == exp_id), None)
                if not exp:
                    self._send(404, {"ok": False, "error": "experiment not found"})
                    return
                data = read_output(exp_id)
                self._send(200, {"ok": True, "experiment": exp, "data": data})
            elif path == "/api/figures":
                files = []
                if os.path.isdir(FIGURES_DIR):
                    for f in sorted(os.listdir(FIGURES_DIR)):
                        base = os.path.splitext(f)[0]
                        if f.endswith((".png", ".pdf", ".svg")):
                            meta = FIGURE_META.get(base, {})
                            files.append({
                                "name": f,
                                "base": base,
                                "title": meta.get("title", base),
                                "desc": meta.get("desc", ""),
                                "metrics": meta.get("metrics", []),
                                "size": os.path.getsize(os.path.join(FIGURES_DIR, f)),
                                "url": "/figures/" + f,
                            })
                self._send(200, {"ok": True, "figures": files})
            elif path.startswith("/figures/"):
                name = os.path.basename(path[len("/figures/"):])
                fpath = os.path.join(FIGURES_DIR, name)
                if not name or not os.path.isfile(fpath):
                    self._send(404, {"ok": False, "error": "figure not found"})
                    return
                ext = os.path.splitext(name)[1].lower()
                ctype = {
                    ".png": "image/png",
                    ".pdf": "application/pdf",
                    ".svg": "image/svg+xml",
                }.get(ext, "application/octet-stream")
                with open(fpath, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send(404, {"ok": False, "error": "not found"})
        except Exception as e:
            self._send(500, {"ok": False, "error": str(e)})

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            body = self._read_body()
            if path == "/api/datasets/upload":
                filename = body.get("filename", "upload.csv")
                b64 = body.get("content_base64", "")
                try:
                    data = base64.b64decode(b64)
                except Exception:
                    data = b64.encode("utf-8")
                parsed = parse_upload(filename, data)
                if not parsed.get("ok"):
                    self._send(400, parsed)
                    return
                upload_id = "up_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(random.randint(100, 999))
                with STATE_LOCK:
                    pending = {
                        "upload_id": upload_id,
                        "filename": filename,
                        "rows": parsed["rows"],
                        "n_rows": parsed["n_rows"],
                        "columns": parsed["columns"],
                        "samples": parsed["samples"],
                        "created_at": now_str(),
                    }
                    # 临时挂到内存 30 分钟；正式保存发生在 /api/datasets/map
                    PENDING_UPLOADS[upload_id] = pending
                mapping = heuristic_mapping(parsed["columns"], parsed["rows"])
                self._send(200, {
                    "ok": True,
                    "upload_id": upload_id,
                    "n_rows": parsed["n_rows"],
                    "columns": parsed["columns"],
                    "samples": parsed["samples"],
                    "mapping": mapping,
                })
            elif path == "/api/datasets/ai-map":
                upload_id = body.get("upload_id", "")
                pending = PENDING_UPLOADS.get(upload_id)
                if not pending:
                    self._send(404, {"ok": False, "error": "upload not found"})
                    return
                r = ai_mapping(pending["columns"], pending["rows"])
                self._send(200, {"ok": True, **r})
            elif path == "/api/datasets/map":
                upload_id = body.get("upload_id", "")
                pending = PENDING_UPLOADS.get(upload_id)
                if not pending:
                    self._send(404, {"ok": False, "error": "upload not found"})
                    return
                mapping = body.get("mapping") or pending.get("mapping") or {}
                n = save_upload_dataset(upload_id, pending["rows"], mapping, {"original": pending["filename"]})
                PENDING_UPLOADS.pop(upload_id, None)
                self._send(200, {"ok": True, "dataset_id": "upload:" + upload_id, "n_rows": n})
            elif path == "/api/experiments":
                exp = create_experiment(body)
                self._send(200, {"ok": True, "experiment": {k: v for k, v in exp.items() if k != "thread"}})
            elif path == "/api/experiments/stop":
                exp_id = body.get("id", "")
                ev = STOP_FLAGS.get(exp_id)
                if ev:
                    ev.set()
                self._send(200, {"ok": True})
            elif path == "/api/chat-experiments":
                entry = upsert_chat_experiment(body)
                self._send(200, {"ok": True, "experiment": entry})
            elif path == "/api/figures/export":
                gen = os.path.join(FIGURES_DIR, "gen_all.py")
                if not os.path.exists(gen):
                    self._send(400, {"ok": False, "error": "figures/gen_all.py 不存在"})
                    return
                proc = subprocess.Popen(
                    [sys.executable, gen], cwd=ROOT,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace")
                out, _ = proc.communicate(timeout=300)
                files = []
                if os.path.isdir(FIGURES_DIR):
                    files = [f for f in sorted(os.listdir(FIGURES_DIR)) if f.endswith((".png", ".pdf"))]
                self._send(200, {"ok": proc.returncode == 0, "log": out[-2000:], "figures": files})
            else:
                self._send(404, {"ok": False, "error": "not found"})
        except Exception as e:
            self._send(500, {"ok": False, "error": str(e)})


PENDING_UPLOADS = {}


def main():
    global PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--selftest", action="store_true",
                    help="启动后自动跑一遍核心 API 冒烟测试再退出")
    args = ap.parse_args()
    PORT = args.port
    os.makedirs(OUTPUT, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"DistillToMCU 实验服务已启动: http://{args.host}:{args.port}")
    print(f"API Key: {'可用 (真实 LLM)' if api_key_available() else '缺失 (将回退 mock，请设 DEEPSEEK_API_KEY)'}")
    if args.selftest:
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        time.sleep(0.5)
        import urllib.request as ur
        base = f"http://{args.host}:{args.port}"
        results = []
        def call(method, path, body=None):
            data = json.dumps(body).encode() if body is not None else None
            req = ur.Request(base + path, data=data, method=method,
                             headers={"Content-Type": "application/json"})
            with ur.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode())
        try:
            results.append(("health", call("GET", "/api/health")))
            results.append(("datasets", call("GET", "/api/datasets")))
            results.append(("chat upsert", call("POST", "/api/chat-experiments", {
                "id": "selftest_chat", "title": "自检对话",
                "msgs": [
                    {"role": "me", "text": "开灯", "meta": ""},
                    {"role": "ai", "text": "好的", "meta": "cloud · 2500 ms · action led.on"},
                    {"role": "ai", "text": "已关", "meta": "local · rule r1 · 5 ms · action led.off"},
                ],
            })))
            for name, r in results:
                ok = r.get("ok", r.get("experiments") is not None)
                print(f"[selftest] {name}: {'OK' if ok else 'FAIL'} {json.dumps(r, ensure_ascii=False)[:160]}")
        finally:
            srv.shutdown()
        return
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
