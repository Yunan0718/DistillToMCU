"""
DistillToMCU Phase 0a — 基线实现
===============================
所有基线统一接口：Baseline.handle(interaction, current_time) → result

可复现性：每个基线接收相同的 random seed，确保对比公平。
"""

import random
import json
import math
from collections import defaultdict

# 场景化手写规则（基线 B3）：默认环境规则；工业/空气场景用领域规则。
STEEL_USER_RULES = [
    {"conds": [("usage_kwh", "gt", 50)], "act": {"device": "fan", "command": "on", "params": {"speed": 2}}},
    {"conds": [("usage_kwh", "gt", 80)], "act": {"device": "fan", "command": "on", "params": {"speed": 3}}},
    {"conds": [("usage_kwh", "lt", 10)], "act": {"device": "fan", "command": "off", "params": {}}},
    {"conds": [("power_factor", "lt", 70)], "act": {"device": "led", "command": "on", "params": {"brightness": 70}}},
    {"conds": [("power_factor", "lt", 60)], "act": {"device": "led", "command": "on", "params": {"brightness": 90}}},
    {"conds": [("co2", "gt", 0.03)], "act": {"device": "curtain", "command": "on", "params": {"position": 30}}},
    {"conds": [("co2", "gt", 0.05)], "act": {"device": "curtain", "command": "on", "params": {"position": 10}}},
    {"conds": [("load_level", "eq", 3), ("usage_kwh", "gt", 60)], "act": {"device": "curtain", "command": "off", "params": {}}},
    {"conds": [("lagging_power", "gt", 40)], "act": {"device": "led", "command": "on", "params": {"brightness": 60}}},
    {"conds": [("usage_kwh", "lt", 5)], "act": {"device": "led", "command": "off", "params": {}}},
]

AIRQUALITY_USER_RULES = [
    {"conds": [("co", "gt", 5)], "act": {"device": "fan", "command": "on", "params": {"speed": 2}}},
    {"conds": [("co", "gt", 8)], "act": {"device": "fan", "command": "on", "params": {"speed": 3}}},
    {"conds": [("nox", "gt", 400)], "act": {"device": "fan", "command": "on", "params": {"speed": 2}}},
    {"conds": [("no2", "gt", 150)], "act": {"device": "led", "command": "on", "params": {"brightness": 70}}},
    {"conds": [("no2", "gt", 200)], "act": {"device": "led", "command": "on", "params": {"brightness": 90}}},
    {"conds": [("temperature", "gt", 30)], "act": {"device": "curtain", "command": "on", "params": {"position": 70}}},
    {"conds": [("temperature", "lt", 10)], "act": {"device": "curtain", "command": "off", "params": {}}},
    {"conds": [("co", "lt", 1), ("nox", "lt", 50)], "act": {"device": "fan", "command": "off", "params": {}}},
    {"conds": [("humidity", "gt", 70)], "act": {"device": "fan", "command": "on", "params": {"speed": 1}}},
    {"conds": [("temperature", "gt", 25), ("no2", "gt", 100)], "act": {"device": "curtain", "command": "on", "params": {"position": 50}}},
]

USER_RULES_BY_LABEL = {
    "steel": STEEL_USER_RULES,
    "airquality": AIRQUALITY_USER_RULES,
}


def extract_cloud_action(trace: dict) -> dict | None:
    """从一条 trace 提取 LLM 云端决策动作 (device, command, params)。"""
    llm_resp = trace.get("llm_response", {}) or {}
    tool_calls = llm_resp.get("tool_calls") or []
    if not tool_calls:
        return None
    func = tool_calls[0].get("function", {})
    name = func.get("name", "")
    if "_control" not in name:
        return None
    try:
        args = json.loads(func.get("arguments", "{}"))
    except (json.JSONDecodeError, TypeError):
        args = {}
    return {
        "device": name.replace("_control", ""),
        "command": args.get("command", "on"),
        "params": {k: v for k, v in args.items() if k != "command"},
    }


def trace_day_labels(n_traces: int, day_bounds=None) -> list[int]:
    """给按时间排序的 trace 分配 day 编号（1-based）。

    day_bounds: 每个 day 的累计 trace 数量（metrics.jsonl 的 total 字段）。
    缺失时退化为按索引均匀分 30 天。
    """
    labels = [0] * n_traces
    if day_bounds and max(day_bounds) > 0:
        idx = 0
        for day, end in enumerate(day_bounds, start=1):
            while idx < n_traces and idx < end:
                labels[idx] = day
                idx += 1
        for i in range(idx, n_traces):
            labels[i] = len(day_bounds)
    else:
        per = max(1, n_traces // 30)
        for i in range(n_traces):
            labels[i] = min(30, i // per + 1)
    return labels


class ExactCacheBaseline:
    """
    B2: Exact Cache — 精确传感器快照匹配。
    只有当前传感器状态的精确副本在历史中出现过时，才命中。
    这证明"没有泛化"时 AR 有多低——是规则泛化价值的对标基线。

    注：浮点传感器值按 0.1 精度量化后做精确匹配（比真正精确匹配更宽松），
    因此该基线是 Exact Cache 的"宽松上界"。
    """

    def __init__(self, seed=42):
        random.seed(seed)
        self.cache = {}  # frozenset(sensor_items) -> action
        self.metrics = {
            "total": 0, "local": 0, "cloud": 0,
            "local_lat_sum": 0, "cloud_lat_sum": 0,
        }

    def handle(self, interaction, current_time=None):
        self.metrics["total"] += 1
        sensors = interaction["sensors"]

        # 精确匹配：每个传感器的值和类型必须完全一致
        # 使用 frozenset 保证 O(1) 查找
        key = frozenset(
            (k, round(v, 1) if isinstance(v, float) else v)
            for k, v in sorted(sensors.items()) if v is not None
        )

        if key in self.cache:
            # Cache hit → 本地执行
            lat = random.randint(3, 15)
            self.metrics["local"] += 1
            self.metrics["local_lat_sum"] += lat
            return {
                "mode": "local",
                "rule_id": f"cache_{hash(key)}",
                "latency_ms": lat,
                "action": self.cache[key],
            }
        else:
            # Cache miss → 走云端（实际中会调用 LLM，这里记录即可）
            lat = random.randint(800, 2500)
            self.metrics["cloud"] += 1
            self.metrics["cloud_lat_sum"] += lat

            # 从 interaction 中提取 action（模拟"LLM 返回了结果"）
            # 在实际实验中，这个 action 来自真实 LLM 调用
            action = interaction.get("_cloud_action")

            # 记录到缓存
            if action:
                self.cache[key] = action

            return {
                "mode": "cloud",
                "rule_id": None,
                "latency_ms": lat,
                "action": action,
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
            "cache_size": len(self.cache),
        }


class LLMOneShotBaseline:
    """
    B4: LLM One-shot Rules — WireClaw 风格。
    第一天让 LLM 生成 10 条规则，之后冻结不再更新。
    模拟"LLM 一次性写规则，MCU 固定执行"的模式。

    ⚠️ 诚实性（v6 修复）：
      - 只有当 llm_client 提供且可用时，规则才由 LLM 生成（display_name 标注 "LLM-generated"）。
      - 无 LLM / 无 API Key 时使用内置手写规则，display_name 标注 "handcrafted fallback"，
        绝不冒充 LLM 生成。
    """

    def __init__(self, llm_client=None, seed=42):
        random.seed(seed)
        self.llm = llm_client
        self.rules = []  # 一次性生成的固定规则
        self._rules_generated = False
        self.generated_by_llm = False
        self.metrics = {
            "total": 0, "local": 0, "cloud": 0,
            "local_lat_sum": 0, "cloud_lat_sum": 0,
        }

    def _generate_rules(self):
        """让 LLM 一次性生成规则；LLM 不可用时使用手写规则并诚实标注。"""
        llm_ok = False
        if self.llm is not None and hasattr(self.llm, "call_llm_with_backend"):
            try:
                available = self.llm.get_available_llms() \
                    if hasattr(self.llm, "get_available_llms") else []
                llm_ok = len(available) > 0
            except Exception:
                llm_ok = False

        if llm_ok:
            prompt = (
                "Generate 10 smart-home automation rules as a JSON array. "
                "Each rule: {\"conditions\":[{\"sensor\":\"temperature\",\"op\":\"gt\",\"value\":30}],"
                "\"action\":{\"device\":\"fan\",\"command\":\"on\",\"params\":{\"speed\":2}}}. "
                "Sensors: temperature, humidity, light, motion. Devices: led, fan, curtain."
            )
            try:
                resp = self.llm.call_llm_with_backend(
                    [{"role": "user", "content": prompt}],
                    backend="deepseek-v4-flash", temperature=0.0,
                    max_tokens=4096)
                content = (resp.get("content") or "").strip()
                start, end = content.find("["), content.rfind("]")
                if start >= 0 and end > start:
                    parsed = json.loads(content[start:end + 1])
                    if isinstance(parsed, list) and parsed:
                        self.rules = parsed[:10]
                        self.generated_by_llm = True
            except Exception:
                self.rules = []
                self.generated_by_llm = False

        if not self.rules:
            # 无 LLM 时用预定义规则
            self.rules = [
                {"conditions": [{"sensor": "temperature", "op": "gt", "value": 30}],
                 "action": {"device": "fan", "command": "on", "params": {"speed": 2}}},
                {"conditions": [{"sensor": "temperature", "op": "gt", "value": 33}],
                 "action": {"device": "fan", "command": "on", "params": {"speed": 3}}},
                {"conditions": [{"sensor": "light", "op": "lt", "value": 50}],
                 "action": {"device": "led", "command": "on", "params": {"brightness": 60}}},
                {"conditions": [{"sensor": "light", "op": "lt", "value": 20}],
                 "action": {"device": "led", "command": "on", "params": {"brightness": 80}}},
                {"conditions": [{"sensor": "motion", "op": "eq", "value": 1},
                                {"sensor": "light", "op": "lt", "value": 100}],
                 "action": {"device": "led", "command": "on", "params": {"brightness": 50}}},
                {"conditions": [{"sensor": "temperature", "op": "lt", "value": 18}],
                 "action": {"device": "fan", "command": "off", "params": {}}},
                {"conditions": [{"sensor": "light", "op": "gt", "value": 800}],
                 "action": {"device": "curtain", "command": "on", "params": {"position": 50}}},
            ]
            self.generated_by_llm = False
        self._rules_generated = True

    @property
    def display_name(self) -> str:
        return "LLM One-shot (LLM-gen)" if self.generated_by_llm \
            else "LLM One-shot (handcrafted fallback)"

    def _match(self, sensors):
        """检查是否有规则匹配当前传感器状态"""
        for rule in self.rules:
            # v7 防护：空条件规则匹配一切，不公平。跳过。
            if not rule["conditions"]:
                continue
            match = True
            for cond in rule["conditions"]:
                s_val = sensors.get(cond.get("sensor"))
                if s_val is None:
                    match = False
                    break
                op = cond.get("op", "eq")
                threshold = cond.get("value")
                if op == "gt" and not (s_val > threshold):
                    match = False; break
                elif op == "lt" and not (s_val < threshold):
                    match = False; break
                elif op == "gte" and not (s_val >= threshold):
                    match = False; break
                elif op == "lte" and not (s_val <= threshold):
                    match = False; break
                elif op == "eq" and s_val != threshold:
                    match = False; break
            if match:
                return rule["action"]
        return None

    def handle(self, interaction, current_time=None):
        if not self._rules_generated:
            self._generate_rules()

        self.metrics["total"] += 1
        sensors = interaction["sensors"]

        action = self._match(sensors)

        if action:
            lat = random.randint(3, 15)
            self.metrics["local"] += 1
            self.metrics["local_lat_sum"] += lat
            return {"mode": "local", "rule_id": "oneshot", "latency_ms": lat, "action": action}
        else:
            lat = random.randint(800, 2500)
            self.metrics["cloud"] += 1
            self.metrics["cloud_lat_sum"] += lat
            return {"mode": "cloud", "rule_id": None, "latency_ms": lat}

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
            "num_rules": len(self.rules),
        }


# ===== Sentence-Transformers 检测 (离线模式) =====
_SENTENCE_TRANSFORMERS_AVAILABLE = False
_ST_MODEL = None
try:
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    # 包未安装 → 降级为 4 维传感器向量，诚实标注名称
    pass


def _get_st_model():
    """懒加载 MiniLM（避免每次 import baselines 都加载 90MB 模型）。"""
    global _ST_MODEL, _SENTENCE_TRANSFORMERS_AVAILABLE
    if _ST_MODEL is None and _SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            from sentence_transformers import SentenceTransformer
            _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2',
                                            local_files_only=True)
        except Exception:
            _SENTENCE_TRANSFORMERS_AVAILABLE = False
    return _ST_MODEL


class SemanticCacheBaseline:
    """
    Semantic Cache (supplementary) — embedding → cosine similarity 缓存。不在论文主表。

    当 sentence-transformers 可用时：
      使用 all-MiniLM-L6-v2 (384-dim) 将传感器状态转为文本嵌入。
      例: "temperature:31.2 humidity:55 light:45 motion:1" → 384维向量

    不可用时（降级）：
      使用归一化传感器向量 [temp/50, hum/100, light/1000, motion]，
      此时基线名自动变为 "Sensor-Vector Sim Cache"。
    """

    def __init__(self, threshold: float = 0.9, seed: int = 42):
        random.seed(seed)
        self.threshold = threshold
        self.cache_keys = []
        self.cache_actions = []
        self.metrics = {
            "total": 0, "local": 0, "cloud": 0,
            "local_lat_sum": 0, "cloud_lat_sum": 0,
        }
        self._using_real_embeddings = _SENTENCE_TRANSFORMERS_AVAILABLE

    @property
    def display_name(self) -> str:
        return "Semantic Cache (MiniLM)" if self._using_real_embeddings \
            else "Sensor-Vector Sim Cache"

    def _get_embedding(self, sensors: dict) -> list[float]:
        model = _get_st_model()
        if _SENTENCE_TRANSFORMERS_AVAILABLE and model:
            # 真实 MiniLM 嵌入: 传感器→文本→384维向量
            text = " ".join(
                f"{k}:{round(v, 1) if isinstance(v, float) else v}"
                for k, v in sorted(sensors.items())
                if v is not None
            )
            return model.encode(text).tolist()
        # 降级: 4维传感器归一化向量
        vec = []
        for k in ["temperature", "humidity", "light", "motion"]:
            v = sensors.get(k, 0) or 0
            if k == "temperature":
                vec.append(v / 50.0)
            elif k == "humidity":
                vec.append(v / 100.0)
            elif k == "light":
                vec.append(v / 1000.0)
            else:
                vec.append(float(v))
        return vec

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def handle(self, interaction, current_time=None):
        self.metrics["total"] += 1
        sensors = interaction["sensors"]
        emb = self._get_embedding(sensors)

        # 找最相似的缓存条目
        best_sim = -1
        best_idx = -1
        for i, cached_emb in enumerate(self.cache_keys):
            sim = self._cosine_similarity(emb, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self.threshold and best_idx >= 0:
            lat = random.randint(3, 8)
            self.metrics["local"] += 1
            self.metrics["local_lat_sum"] += lat
            return {"mode": "local", "rule_id": f"semcache_{best_idx}",
                    "latency_ms": lat, "action": self.cache_actions[best_idx]}

        # Miss → cloud
        lat = random.randint(800, 2500)
        self.metrics["cloud"] += 1
        self.metrics["cloud_lat_sum"] += lat

        action = interaction.get("_cloud_action")
        if action:
            self.cache_keys.append(emb)
            self.cache_actions.append(action)
        return {"mode": "cloud", "rule_id": None, "latency_ms": lat}

    def get_summary(self):
        m = self.metrics; t = max(1, m["total"])
        lc = max(1, m["local"]); cc = max(1, m["cloud"])
        return {
            "autonomy_rate": round(m["local"] / t * 100, 1),
            "cloud_call_reduction": round((1 - m["cloud"] / t) * 100, 1),
            "avg_local_latency_ms": round(m["local_lat_sum"] / lc, 1),
            "avg_cloud_latency_ms": round(m["cloud_lat_sum"] / cc, 1),
            "total": t, "local": m["local"], "cloud": m["cloud"],
            "cache_size": len(self.cache_keys),
        }


class SensorVectorCacheBaseline:
    """
    B3b: Sensor-Vector Similarity Cache — MCU-feasible 版本。

    与 Semantic Cache (MiniLM) 的关键区别：
      - Semantic Cache 需要 90MB all-MiniLM-L6-v2 模型，ESP32-S3 跑不了
      - SensorVector Cache 只需 4 维归一化传感器向量 × 4B = 16B/缓存条目
      - 余弦相似度 = dot product / (|a| * |b|)，MCU 上 O(4) 乘加运算

    论文定位：和 Semantic Cache 并列展示——高端方案（MiniLM）vs MCU可行方案（向量）。
    """

    def __init__(self, threshold: float = 0.9, seed: int = 42):
        random.seed(seed)
        self.threshold = threshold
        self.cache_keys = []
        self.cache_actions = []
        self.metrics = {
            "total": 0, "local": 0, "cloud": 0,
            "local_lat_sum": 0, "cloud_lat_sum": 0,
        }

    @property
    def display_name(self) -> str:
        return "SensorVector Cache (MCU-feasible)"

    @staticmethod
    def _sensor_vector(sensors: dict) -> list[float]:
        return [
            sensors.get("temperature", 25) / 50.0,
            sensors.get("humidity", 55) / 100.0,
            sensors.get("light", 500) / 1000.0,
            float(sensors.get("motion", 0)),
        ]

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    def handle(self, interaction, current_time=None):
        self.metrics["total"] += 1
        sensors = interaction["sensors"]
        vec = self._sensor_vector(sensors)

        best_sim, best_idx = -1.0, -1
        for i, cv in enumerate(self.cache_keys):
            sim = self._cosine(vec, cv)
            if sim > best_sim:
                best_sim, best_idx = sim, i

        if best_sim >= self.threshold and best_idx >= 0:
            lat = random.randint(3, 8)
            self.metrics["local"] += 1
            self.metrics["local_lat_sum"] += lat
            return {"mode": "local", "rule_id": f"svcache_{best_idx}",
                    "latency_ms": lat, "action": self.cache_actions[best_idx]}

        lat = random.randint(800, 2500)
        self.metrics["cloud"] += 1
        self.metrics["cloud_lat_sum"] += lat
        action = interaction.get("_cloud_action")
        if action:
            self.cache_keys.append(vec)
            self.cache_actions.append(action)
        return {"mode": "cloud", "rule_id": None, "latency_ms": lat}

    def get_summary(self):
        m = self.metrics; t = max(1, m["total"])
        lc = max(1, m["local"]); cc = max(1, m["cloud"])
        return {
            "autonomy_rate": round(m["local"] / t * 100, 1),
            "cloud_call_reduction": round((1 - m["cloud"] / t) * 100, 1),
            "avg_local_latency_ms": round(m["local_lat_sum"] / lc, 1),
            "avg_cloud_latency_ms": round(m["cloud_lat_sum"] / cc, 1),
            "total": t, "local": m["local"], "cloud": m["cloud"],
            "cache_size": len(self.cache_keys),
        }


class UserDefinedRulesBaseline:
    """
    B3: User-defined Rules — 用户手写的 10 条规则，固定不变。
    模拟"人类专家规则"的上限。
    """

    def __init__(self, seed: int = 42, rules: list | None = None):
        random.seed(seed)
        self.rules = rules if rules is not None else [
            {"conds": [("temperature", "gt", 30)], "act": {"device": "fan", "command": "on", "params": {"speed": 2}}},
            {"conds": [("temperature", "gt", 33)], "act": {"device": "fan", "command": "on", "params": {"speed": 3}}},
            {"conds": [("temperature", "lt", 18)], "act": {"device": "fan", "command": "off", "params": {}}},
            {"conds": [("light", "lt", 40)], "act": {"device": "led", "command": "on", "params": {"brightness": 70}}},
            {"conds": [("light", "lt", 80), ("motion", "eq", 1)], "act": {"device": "led", "command": "on", "params": {"brightness": 50}}},
            {"conds": [("light", "gt", 800)], "act": {"device": "curtain", "command": "on", "params": {"position": 70}}},
            {"conds": [("motion", "eq", 0), ("light", "gt", 100)], "act": {"device": "led", "command": "off", "params": {}}},
            {"conds": [("temperature", "gt", 28), ("humidity", "gt", 70)], "act": {"device": "fan", "command": "on", "params": {"speed": 3}}},
            {"conds": [("motion", "eq", 1)], "act": {"device": "led", "command": "on", "params": {"brightness": 40}}},
            {"conds": [("temperature", "lt", 22), ("temperature", "gt", 20)], "act": {"device": "fan", "command": "off", "params": {}}},
        ]
        self.metrics = {"total": 0, "local": 0, "cloud": 0, "local_lat_sum": 0, "cloud_lat_sum": 0}

    def _match(self, sensors):
        for rule in self.rules:
            ok = True
            for sname, op, val in rule["conds"]:
                sv = sensors.get(sname)
                if sv is None: ok = False; break
                if op == "gt" and not (sv > val): ok = False; break
                if op == "lt" and not (sv < val): ok = False; break
                if op == "eq" and sv != val: ok = False; break
            if ok: return rule["act"]
        return None

    def handle(self, interaction, current_time=None):
        self.metrics["total"] += 1
        act = self._match(interaction["sensors"])
        if act:
            lat = random.randint(3, 15)
            self.metrics["local"] += 1
            self.metrics["local_lat_sum"] += lat
            return {"mode": "local", "rule_id": "userdef", "latency_ms": lat, "action": act}
        lat = random.randint(800, 2500)
        self.metrics["cloud"] += 1
        self.metrics["cloud_lat_sum"] += lat
        return {"mode": "cloud", "rule_id": None, "latency_ms": lat}

    def get_summary(self):
        m = self.metrics; t = max(1, m["total"])
        return {
            "autonomy_rate": round(m["local"] / t * 100, 1),
            "cloud_call_reduction": round((1 - m["cloud"] / t) * 100, 1),
            "avg_local_latency_ms": round(m["local_lat_sum"] / max(1, m["local"]), 1),
            "avg_cloud_latency_ms": round(m["cloud_lat_sum"] / max(1, m["cloud"]), 1),
            "total": t, "local": m["local"], "cloud": m["cloud"],
            "num_rules": len(self.rules),
        }


class DecisionTreeBaseline:
    """
    B5: Decision Tree Imitation — 用 CART 从 LLM 行为中克隆决策边界。
    sklearn.tree.DecisionTreeClassifier，训练后提取规则。
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._tree = None
        self._trained = False
        self._trained_with_sklearn = False
        self._feature_names = ["temperature", "humidity", "light", "motion"]
        self.metrics = {"total": 0, "local": 0, "cloud": 0, "local_lat_sum": 0, "cloud_lat_sum": 0}
        self._X = []
        self._y = []

    def train(self, traces: list):
        """从 trace 数据训练决策树"""
        try:
            import numpy as np
            from sklearn.tree import DecisionTreeClassifier
        except ImportError:
            # 无 sklearn：不训练，AR=0，诚实标注（不冒充决策树）
            self._trained = True
            self._trained_with_sklearn = False
            return

        # 推断特征字段：默认环境字段不在 trace 里时，用 trace 实际数值字段
        keys = set()
        for t in traces:
            keys.update((t.get("sensors", {}) or {}).keys())
        non_numeric = {"time_of_day", "light_category"}
        if not set(self._feature_names).issubset(keys):
            inferred = sorted(k for k in keys if k not in non_numeric)
            if inferred:
                self._feature_names = inferred

        X, y = [], []
        for t in traces:
            sensors = t.get("sensors", {})
            tc = (t.get("llm_response", {}) or {}).get("tool_calls") or []
            if tc:
                func = tc[0].get("function", {})
                name = func.get("name", "").replace("_control", "")
                feat = [sensors.get(f, 0) or 0 for f in self._feature_names]
                X.append(feat)
                y.append(name)

        if len(X) < 10:
            self._trained = True
            return

        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42)
        clf.fit(np.array(X), np.array(y))
        self._tree = clf
        self._trained = True
        self._trained_with_sklearn = True

    @property
    def display_name(self) -> str:
        return "Decision Tree (sklearn)" if self._trained_with_sklearn \
            else "Decision Tree (unavailable, AR=0)"

    def _predict(self, sensors: dict) -> str | None:
        if not self._tree:
            return None
        try:
            import numpy as np
            feat = np.array([[sensors.get(f, 0) or 0 for f in self._feature_names]])
            # v7 修复：加入 no-action 能力。如果最大预测概率 < 0.5，
            # 返回 None（走向云端），避免决策树对所有输入都输出设备名。
            proba = self._tree.predict_proba(feat)[0]
            best_idx = int(proba.argmax())
            if proba[best_idx] < 0.5:
                return None
            return self._tree.classes_[best_idx]
        except Exception:
            return None

    def handle(self, interaction, current_time=None):
        self.metrics["total"] += 1
        if self._trained:
            predicted = self._predict(interaction["sensors"])
            if predicted:
                lat = random.randint(5, 15)
                self.metrics["local"] += 1
                self.metrics["local_lat_sum"] += lat
                return {"mode": "local", "rule_id": "dt", "latency_ms": lat,
                        "action": {"device": predicted, "command": "on", "params": {}}}
        lat = random.randint(800, 2500)
        self.metrics["cloud"] += 1
        self.metrics["cloud_lat_sum"] += lat
        return {"mode": "cloud", "rule_id": None, "latency_ms": lat}

    def get_summary(self):
        m = self.metrics; t = max(1, m["total"])
        return {
            "autonomy_rate": round(m["local"] / t * 100, 1),
            "cloud_call_reduction": round((1 - m["cloud"] / t) * 100, 1),
            "avg_local_latency_ms": round(m["local_lat_sum"] / max(1, m["local"]), 1),
            "avg_cloud_latency_ms": round(m["cloud_lat_sum"] / max(1, m["cloud"]), 1),
            "total": t, "local": m["local"], "cloud": m["cloud"],
        }


class OnlineDailyRefitDecisionTreeBaseline(DecisionTreeBaseline):
    """B5b: Online Decision Tree（每日重训，公平的在线对标基线）。

    协议与 Ours 完全一致：
      - 第 1 天无任何模型 → 全部走云端；
      - 每天开始前用"截至前一天"的所有云端决策重训 CART；
      - 评估期内继续用已观测到的过去决策增量学习（只用过去，不偷看未来）。

    与批量 DT（一次拿全量 trace 训练后回放）形成对照：同样一颗树，
    换成在线信息条件后 AR/AGREE 会发生什么变化。
    """

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._history = []          # (feature_vec, device_label)
        self._last_fit_day = -1
        self._min_samples = 10

    @property
    def online_learning(self) -> bool:
        return True

    def _feat_label(self, interaction, act) -> tuple:
        sensors = interaction.get("sensors", {})
        feat = [sensors.get(f, 0) or 0 for f in self._feature_names]
        return (feat, act["device"])

    def train(self, traces: list):
        """预热期：仅收集样本，不拟合；首个评估日再开始按天重训。"""
        self._history = []
        for t in traces:
            act = t.get("_cloud_action")
            if act:
                self._history.append(self._feat_label(t, act))

    def _fit(self):
        if len(self._history) < self._min_samples:
            self._tree = None
            self._trained = False
            return
        try:
            import numpy as np
            from sklearn.tree import DecisionTreeClassifier
        except ImportError:
            self._trained = False
            return
        X = np.array([h[0] for h in self._history], dtype=float)
        y = [h[1] for h in self._history]
        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3,
                                     random_state=42)
        clf.fit(X, np.array(y))
        self._tree = clf
        self._trained = True
        self._trained_with_sklearn = True

    def handle(self, interaction, current_time=None):
        self.metrics["total"] += 1
        day = interaction.get("_day")
        if day is not None and day > self._last_fit_day:
            self._fit()
            self._last_fit_day = day

        predicted = self._predict(interaction["sensors"]) if self._trained else None

        # 在线语义：先做决策，再观察当前决策结果并加入历史（不偷看未来）
        act = interaction.get("_cloud_action")
        if act:
            self._history.append(self._feat_label(interaction, act))

        if predicted:
            lat = random.randint(5, 15)
            self.metrics["local"] += 1
            self.metrics["local_lat_sum"] += lat
            return {"mode": "local", "rule_id": "dt_online", "latency_ms": lat,
                    "action": {"device": predicted, "command": "on", "params": {}}}
        lat = random.randint(800, 2500)
        self.metrics["cloud"] += 1
        self.metrics["cloud_lat_sum"] += lat
        return {"mode": "cloud", "rule_id": None, "latency_ms": lat}

    @property
    def display_name(self) -> str:
        return "Decision Tree (online daily-refit)" if self._trained_with_sklearn \
            else "Decision Tree (online, sklearn unavailable)"


class ESPHomeStateMachineBaseline:
    """
    ESPHome State Machine (supplementary) — 传统阈值规则。不在主表。
    基于固定阈值的 if-else 逻辑。
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.metrics = {"total": 0, "local": 0, "cloud": 0, "local_lat_sum": 0, "cloud_lat_sum": 0}

    def _evaluate(self, sensors: dict) -> dict | None:
        temp = sensors.get("temperature", 25)
        light = sensors.get("light", 500)
        motion = sensors.get("motion", 0)

        # ESPHome 常见逻辑
        if temp > 32:
            return {"device": "fan", "command": "on", "params": {"speed": 3}}
        elif temp > 28:
            return {"device": "fan", "command": "on", "params": {"speed": 1}}
        elif temp < 18:
            return {"device": "fan", "command": "off", "params": {}}

        if light < 30:
            return {"device": "led", "command": "on", "params": {"brightness": 80}}
        elif light < 60 and motion == 1:
            return {"device": "led", "command": "on", "params": {"brightness": 50}}

        if light > 800:
            return {"device": "curtain", "command": "on", "params": {"position": 50}}

        return None

    def handle(self, interaction, current_time=None):
        self.metrics["total"] += 1
        act = self._evaluate(interaction["sensors"])
        if act:
            lat = random.randint(3, 15)
            self.metrics["local"] += 1
            self.metrics["local_lat_sum"] += lat
            return {"mode": "local", "rule_id": "esphome", "latency_ms": lat, "action": act}
        lat = random.randint(800, 2500)
        self.metrics["cloud"] += 1
        self.metrics["cloud_lat_sum"] += lat
        return {"mode": "cloud", "rule_id": None, "latency_ms": lat}

    def get_summary(self):
        m = self.metrics; t = max(1, m["total"])
        return {
            "autonomy_rate": round(m["local"] / t * 100, 1),
            "cloud_call_reduction": round((1 - m["cloud"] / t) * 100, 1),
            "avg_local_latency_ms": round(m["local_lat_sum"] / max(1, m["local"]), 1),
            "avg_cloud_latency_ms": round(m["cloud_lat_sum"] / max(1, m["cloud"]), 1),
            "total": t, "local": m["local"], "cloud": m["cloud"],
        }


class ESPClawStyleBaseline:
    """
    B6: ESP-Claw-style — 模拟 Lua 动态规则生成。
    规则在运行时根据模式匹配结果动态创建和修改。
    Phase 0b: 模拟 ESP-Claw 的 Lua 规则引擎行为（简化版）。
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.patterns = defaultdict(list)
        self.rules = []
        self._rule_counter = 0
        self.metrics = {"total": 0, "local": 0, "cloud": 0, "local_lat_sum": 0, "cloud_lat_sum": 0}

    @property
    def online_learning(self) -> bool:
        return True

    def _learn_from_cloud(self, sensors, action):
        """从云端决策中学习新模式 — 多传感器关联"""
        if not action:
            return
        device = action["device"]
        keys = [k for k in sensors.keys() if sensors.get(k) is not None]
        non_numeric = {"time_of_day", "light_category"}
        CONT_SENSORS = [k for k in keys if k not in non_numeric
                        and k not in ("motion", "load_level")]
        DISC_SENSORS = [k for k in keys if k in ("motion", "load_level")]

        # 为所有连续传感器记录值
        for sname in CONT_SENSORS + DISC_SENSORS:
            sval = sensors.get(sname)
            if sval is not None:
                self.patterns[(sname, device)].append(sval)

        # 每个传感器积累 ≥3 样本后，为它创建区间规则
        conds = []
        for sname in CONT_SENSORS:
            vals = self.patterns.get((sname, device), [])
            if len(vals) >= 3:
                lo = min(vals) - 1
                hi = max(vals) + 1
                conds.append((sname, "gte", lo))
                conds.append((sname, "lte", hi))
        # 离散传感器取众数
        for sname in DISC_SENSORS:
            vals = self.patterns.get((sname, device), [])
            if len(vals) >= 3:
                from collections import Counter
                mode_val = Counter(vals).most_common(1)[0][0]
                conds.append((sname, "eq", mode_val))

        if conds:
            self._rule_counter += 1
            self.rules.append({
                "id": f"espclaw_{self._rule_counter}",
                "conds": conds,
                "act": action,
            })

    def _match(self, sensors):
        for rule in self.rules:
            ok = True
            for sname, op, val in rule["conds"]:
                sv = sensors.get(sname)
                if sv is None: ok = False; break
                if op == "gte" and not (sv >= val): ok = False; break
                if op == "lte" and not (sv <= val): ok = False; break
            if ok: return rule["act"]
        return None

    def handle(self, interaction, current_time=None):
        self.metrics["total"] += 1
        act = self._match(interaction["sensors"])
        if act:
            lat = random.randint(3, 15)
            self.metrics["local"] += 1
            self.metrics["local_lat_sum"] += lat
            return {"mode": "local", "rule_id": "espclaw", "latency_ms": lat, "action": act}

        # Miss → learn from cloud action
        cloud_act = interaction.get("_cloud_action")
        if cloud_act:
            self._learn_from_cloud(interaction["sensors"], cloud_act)

        lat = random.randint(800, 2500)
        self.metrics["cloud"] += 1
        self.metrics["cloud_lat_sum"] += lat
        return {"mode": "cloud", "rule_id": None, "latency_ms": lat}

    def get_summary(self):
        m = self.metrics; t = max(1, m["total"])
        return {
            "autonomy_rate": round(m["local"] / t * 100, 1),
            "cloud_call_reduction": round((1 - m["cloud"] / t) * 100, 1),
            "avg_local_latency_ms": round(m["local_lat_sum"] / max(1, m["local"]), 1),
            "avg_cloud_latency_ms": round(m["cloud_lat_sum"] / max(1, m["cloud"]), 1),
            "total": t, "local": m["local"], "cloud": m["cloud"],
            "learned_rules": len(self.rules),
        }


class PureCloudBaseline:
    """
    B1: Pure Cloud — 每次交互都调云端 LLM，不做任何本地执行。
    这是理论上限——AR=0%，但准确率最高（LLM 直接决策）。
    """

    def __init__(self, seed=42):
        random.seed(seed)
        self.metrics = {
            "total": 0, "local": 0, "cloud": 0,
            "local_lat_sum": 0, "cloud_lat_sum": 0,
        }

    def handle(self, interaction, current_time=None):
        self.metrics["total"] += 1
        lat = random.randint(800, 2500)
        self.metrics["cloud"] += 1
        self.metrics["cloud_lat_sum"] += lat
        return {"mode": "cloud", "rule_id": None, "latency_ms": lat}

    def get_summary(self):
        m = self.metrics
        t = max(1, m["total"])
        cc = max(1, m["cloud"])
        return {
            "autonomy_rate": 0.0,  # Always 0 — pure cloud
            "cloud_call_reduction": 0.0,
            "avg_local_latency_ms": 0,
            "avg_cloud_latency_ms": round(m["cloud_lat_sum"] / cc, 1),
            "total": t, "local": 0, "cloud": m["cloud"],
        }


# ============================================================
# 对比实验运行器
# ============================================================

def run_baseline_comparison(traces, baselines: dict, seed=42,
                            train_ratio: float = 0.7, day_bounds=None):
    """
    用同一批 traces 对多个基线进行对比测试。

    Args:
        traces: 交互 trace 列表（包含 sensors + LLM 的 tool_call 决策）
        baselines: {"name": Baseline()}
        train_ratio: 时间切分比例。1.0 = in-sample（训练集=评估集）；
                     0.7 = 前 70% 天预热、只在后 30% 天评估（真 held-out）。
        day_bounds: metrics.jsonl 的每日累计 trace 数（用于恢复 day 边界）。

    Returns:
        {name: {"final_ar": float, "daily_metrics": [...]}}
    """
    random.seed(seed)

    # v10.6: 真 held-out。前 train_ratio 天仅作预热/训练，评估只发生在
    # 后 (1-train_ratio) 天；此前"train 前 70%、evaluate 全部"混入了训练数据。
    day_labels = trace_day_labels(len(traces), day_bounds)
    enriched_traces = []
    for t, day in zip(traces, day_labels):
        enriched = dict(t)
        # 只有 cloud trace 携带真实的教师决策；local trace 的 tool_calls 是
        # 规则镜像动作，不是 ground truth。
        enriched["_cloud_action"] = (
            extract_cloud_action(t)
            if t.get("execution", {}).get("mode") == "cloud" else None
        )
        enriched["_day"] = day
        enriched_traces.append(enriched)

    if train_ratio >= 1.0:
        warm_span: list = []
        eval_span = enriched_traces
    else:
        n_days = max(day_labels)
        split_day = int(n_days * train_ratio) + 1  # 第一个评估日（1-based）
        warm_span = [e for e in enriched_traces if e["_day"] < split_day]
        eval_span = [e for e in enriched_traces if e["_day"] >= split_day]

    for name, baseline in baselines.items():
        if hasattr(baseline, 'train'):
            baseline.train(warm_span)
        if getattr(baseline, "online_learning", False):
            # 在线学习基线（Online-DT / ESP-Claw）在预热期默默学习，不计指标
            for e in warm_span:
                baseline.handle({
                    "sensors": e.get("sensors", {}),
                    "_cloud_action": e.get("_cloud_action"),
                    "_day": e.get("_day"),
                })

    results = {}
    for name, baseline in baselines.items():
        if hasattr(baseline, "display_name"):
            results[name] = {"display_name": baseline.display_name}
        else:
            results[name] = {}
        baseline.metrics = {
            "total": 0, "local": 0, "cloud": 0,
            "local_lat_sum": 0, "cloud_lat_sum": 0,
        }
        agree = 0
        n_decision = 0
        for e in eval_span:
            r = baseline.handle(
                {"sensors": e.get("sensors", {}),
                 "_cloud_action": e.get("_cloud_action"),
                 "_day": e.get("_day")},
            )
            # 决策一致性只对 LLM 真实决策（cloud trace）计算
            expected = e.get("_cloud_action")
            if expected is None:
                continue
            n_decision += 1
            action = r.get("action")
            if action is not None and action.get("device") == expected["device"] \
                    and action.get("command") == expected["command"]:
                agree += 1
        results[name]["decision_agreement_pct"] = round(
            agree / max(1, n_decision) * 100, 1) if n_decision else 0.0
        results[name]["decision_agreement_n"] = n_decision
        results[name]["eval_window"] = {
            "first_day": min((e["_day"] for e in eval_span), default=1),
            "last_day": max((e["_day"] for e in eval_span), default=1),
            "n_traces": len(eval_span),
        }
        results[name].update(baseline.get_summary())

    return results


def save_baseline_results(results, path):
    """保存基线对比结果到 JSON 文件（论文主表数据源）。"""
    import json as _json
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(results, f, indent=2, ensure_ascii=False)


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    random.seed(42)

    # 模拟 traces
    traces = []
    for i in range(50):
        temp = random.gauss(31.0, 1.0)
        light = random.gauss(50, 10)
        traces.append({
            "sensors": {
                "temperature": round(temp, 1),
                "light": round(light, 1),
                "motion": random.choice([0, 1]),
            },
            "llm_response": {
                "tool_calls": [{
                    "function": {
                        "name": "fan_control",
                        "arguments": json.dumps({"command": "on", "speed": 2}),
                    }
                }]
            } if temp > 29 else {"tool_calls": []},
        })

    baselines = {
        "Pure Cloud": PureCloudBaseline(seed=42),
        "Exact Cache": ExactCacheBaseline(seed=42),
        "LLM One-shot": LLMOneShotBaseline(llm_client=None, seed=42),
    }

    results = run_baseline_comparison(traces, baselines, seed=42)

    print("Baseline Comparison (50 synthetic interactions):")
    print(f"  {'Baseline':<20s} {'AR%':>6s} {'Local':>6s} {'Cloud':>6s}")
    print(f"  {'-'*40}")
    for name, r in results.items():
        print(f"  {name:<20s} {r['autonomy_rate']:5.1f}% {r['local']:5d}  {r['cloud']:5d}")
