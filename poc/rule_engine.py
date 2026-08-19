"""
DistillToMCU Phase 0 PoC — 规则引擎
====================================
规则的表示、存储、匹配、生命周期管理。
每条规则是一个可执行的知识单元。MCU 上的核心引擎。
"""

import json
import math
import os
import time
from config import (
    RULE_CONFIDENCE_THRESHOLD_LOCAL, RULE_CONFIDENCE_THRESHOLD_ASYNC,
    RULE_WILSON_Z, RULE_DECAY_TAU_BASE, RULE_MAX_ACTIVE,
    RULE_RETIRE_AFTER_DAYS, OUTPUT_DIR, RULES_FILE, ACTUATORS,
)


# ========== Wilson Score Confidence ==========

def wilson_score(positive, total, z=RULE_WILSON_Z):
    """
    Wilson score confidence interval (lower bound).
    证据量少时自动压低置信度，避免小样本高置信的欺骗。
    当 total=0 时返回 0。
    """
    if total == 0:
        return 0.0
    p_hat = positive / total
    z2 = z * z
    n = total
    denominator = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) / n + z2 / (4 * n * n))) / denominator
    return max(0.0, min(1.0, center - margin))


# ========== Time-decaying Freshness ==========

def freshness_score(last_triggered_ts, evidence_count, now_ts=None):
    """
    时间衰减新鲜度：f = e^(-Δt / τ)
    τ = base_decay + evidence_count * 0.1
    Δt = 天数 since last_triggered
    越久没触发 → freshness 越低
    evidence 越多 → 衰减越慢（重要的记忆不容易忘）
    """
    if now_ts is None:
        now_ts = time.time()
    delta_days = (now_ts - last_triggered_ts) / 86400.0
    tau = RULE_DECAY_TAU_BASE + evidence_count * 0.15
    return math.exp(-delta_days / tau)


# ========== Rule Class ==========

class Rule:
    """单条规则"""

    STATE_ORDER = ["candidate", "verified", "active", "degraded", "retired"]

    def __init__(self, rule_id, conditions, action, source, safety_level=1):
        now = time.time()
        self.id = rule_id
        self.conditions = conditions          # [{sensor, op, value}, ...]
        self.action = action                  # {device, command, params}
        self.source = source                  # "L1" | "L2" | "L3"
        self.safety_level = safety_level      # 0-3
        self.state = "candidate"

        # 统计
        self.evidence_count = 0
        self.positive_feedback = 0
        self.negative_feedback = 0
        self.confidence = 0.0
        self.freshness = 1.0

        # 时间戳
        self.created_at = now
        self.last_triggered = now
        self.last_modified = now
        self.state_entered_at = now

        # specificity 计算（条件越多→越精细→越优先）
        self.specificity = self._calc_specificity()

    def _calc_specificity(self):
        """条件越多、越精细 → specificity 越高。用于冲突消解。"""
        depth = 0
        for c in self.conditions:
            depth += 1  # 每个 sensor 条件 = +1
            if isinstance(c.get("value"), (int, float)):
                depth += 1  # 数值比较 = +1
            if "time" in c:
                depth += 1  # 时间约束 = +1
        return depth

    def matches(self, sensors, current_time=None, weekday=None):
        """检查这条规则是否匹配当前传感器状态"""
        # v9: 至少需要 max(2, 条件数/3) 个传感器字段在快照中存在
        # 避免一个 12-condition 规则在 4 字段快照（合成数据）永远不匹配，
        # 也避免一个 5-condition 规则仅靠 1 个字段就匹配
        sensor_conditions = [c for c in self.conditions if c.get("sensor")]
        if sensor_conditions:
            present_count = sum(1 for c in sensor_conditions
                               if c.get("sensor") in sensors
                               and sensors[c["sensor"]] is not None)
            min_required = max(2, len(sensor_conditions) // 3)
            if present_count < min_required:
                return False

        for cond in self.conditions:
            if not self._check_condition(cond, sensors, current_time, weekday):
                return False
        return True

    def _check_condition(self, cond, sensors, current_time, weekday):
        sensor = cond.get("sensor")
        if sensor:
            val = sensors.get(sensor)
            # v6 fix: 缺失字段 = 跳过该条件（与 C rule_engine.c _chk() 一致）
            # 合成数据只有 temp/hum/light/motion，学习出的规则可能有 hour/co2 等条件
            if val is None:
                return True
            op = cond.get("op", "eq")
            threshold = cond.get("value")
            if op == "gt":
                return val > threshold
            elif op == "lt":
                return val < threshold
            elif op == "gte":
                return val >= threshold
            elif op == "lte":
                return val <= threshold
            elif op == "eq":
                return val == threshold
        if "time" in cond and current_time is not None:
            t = cond["time"]
            op = t.get("op", "between")
            if op == "between":
                start_h, start_m = map(int, t["start"].split(":"))
                end_h, end_m = map(int, t["end"].split(":"))
                cur_min = current_time.hour * 60 + current_time.minute
                start_min = start_h * 60 + start_m
                end_min = end_h * 60 + end_m
                return start_min <= cur_min <= end_min
        return True  # 没有匹配条件 → 视为通过

    def to_dict(self):
        return {
            "id": self.id,
            "conditions": self.conditions,
            "action": self.action,
            "source": self.source,
            "safety_level": self.safety_level,
            "state": self.state,
            "confidence": round(self.confidence, 4),
            "freshness": round(self.freshness, 4),
            "evidence_count": self.evidence_count,
            "positive_feedback": self.positive_feedback,
            "negative_feedback": self.negative_feedback,
            "specificity": self.specificity,
            "created_at": self.created_at,
            "last_triggered": self.last_triggered,
            "last_modified": self.last_modified,
        }

    def __repr__(self):
        cond_str = " AND ".join(
            f"{c.get('sensor','')} {c.get('op','')} {c.get('value','')}"
            for c in self.conditions
        )
        return f"Rule[{self.id}]({self.state}) {cond_str} → {self.action['device']}.{self.action['command']}"


# ========== Rule Engine ==========

class RuleEngine:
    """
    规则引擎：存储、匹配、生命周期管理的完整封装。
    模拟 MCU 上的 L4+L5 层。
    """

    def __init__(self, selector: str = "deterministic", selector_seed: int = 42):
        self.rules = {}           # id → Rule
        self._inverted_index = {}  # sensor_name → [rule_ids]
        self._id_counter = 0
        self.selector = selector  # "deterministic" | "pms"
        self.selector_seed = selector_seed
        self._pms_selector = None

    # ---- CRUD ----

    def add_rule(self, conditions, action, source, safety_level=1, state="candidate"):
        self._id_counter += 1
        rid = f"rule_{self._id_counter:04d}"
        rule = Rule(rid, conditions, action, source, safety_level)
        rule.state = state
        self.rules[rid] = rule
        self._update_index(rid, rule)
        return rule

    def _update_index(self, rid, rule):
        for c in rule.conditions:
            sensor = c.get("sensor")
            if sensor:
                self._inverted_index.setdefault(sensor, [])
                if rid not in self._inverted_index[sensor]:
                    self._inverted_index[sensor].append(rid)

    def get(self, rule_id):
        return self.rules.get(rule_id)

    # ---- 规则匹配 ----

    def match(self, sensors, current_time=None, weekday=None):
        """
        根据传感器状态匹配所有规则。
        返回激活的规则列表，按优先级排序。
        优先级链：safety_level（L0优先） > specificity（精细优先）
                  > confidence（高置信优先） > freshness（新鲜优先）
        """
        # 先收集可能匹配的 candidate rules
        candidate_ids = set()
        active_sensors = [s for s, v in sensors.items() if v is not None]
        for s in active_sensors:
            if s in self._inverted_index:
                candidate_ids.update(self._inverted_index[s])

        matches = []
        for rid in candidate_ids:
            rule = self.rules.get(rid)
            if rule is None:
                continue
            if rule.state not in ("candidate", "verified", "active"):
                continue
            if rule.matches(sensors, current_time, weekday):
                # safety_level=3 的设备永久禁止本地自动执行
                if rule.safety_level >= 3:
                    continue
                matches.append(rule)

        # 排序：优先级链
        matches.sort(key=lambda r: (
            r.safety_level,       # L0 > L1 > L2 (数字小的优先)
            -r.specificity,       # 越精细越优先
            -r.confidence,        # 置信度高的优先
            -r.freshness,         # 新鲜的优先
        ))
        return matches

    # ---- 冲突消解（Hysteresis + Mutex） ----

    def resolve_conflict(self, matches, actuator_states=None):
        """
        从多个匹配中选一条执行。
        actuator_states: {device: {"state": "on"/"off"/..., "last_actuated": ts}}
        Hysteresis：执行器已 ON → 维持 ON 的规则优先。
        """
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]

        # v10: 可选 PMS 选择器——非平稳/探索场景下替代确定性排序
        if self.selector == "pms":
            if self._pms_selector is None:
                from bandit_selector import PMSRuleSelector
                self._pms_selector = PMSRuleSelector(self, seed=self.selector_seed)
            return self._pms_selector.select_from_matches(matches)

        if actuator_states:
            # 检查是否有规则维持当前状态
            best = matches[0]
            for rule in matches:
                dev = rule.action["device"]
                state = actuator_states.get(dev, {})
                # 如果规则的动作和当前状态一致 → 加分（Hysteresis）
                rule._hysteresis_bonus = 0
                expected_state = "on" if rule.action["command"] in ("on", "set") else "off"
                if state.get("state") == expected_state:
                    rule._hysteresis_bonus = 1  # 维持当前状态 → 优先

            # 重新排序：加 Hysteresis bonus
            matches.sort(key=lambda r: (
                r.safety_level,
                -r.specificity,
                -(r.confidence + getattr(r, "_hysteresis_bonus", 0) * 0.1),
                -r.freshness,
            ))

        best = matches[0]
        # v7 修复：如果有两条规则置信度非常接近（< 0.15），
        # 不 fallback 到云端——选 evidence 更多的（数据支撑更可靠）。
        # 旧行为：return None（导致 Full Pipeline 的 AR 崩溃）。
        second = matches[1] if len(matches) > 1 else None
        if second and abs(best.confidence - second.confidence) < 0.15:
            # 选 evidence_count 高的；相等则选 specificity 高的
            if second.evidence_count > best.evidence_count:
                best = second
            elif (second.evidence_count == best.evidence_count
                  and second.specificity > best.specificity):
                best = second

        return best

    # ---- 生命周期更新 ----

    def update_on_execution(self, rule_id, feedback="accepted"):
        """规则被匹配并执行后更新统计"""
        rule = self.rules.get(rule_id)
        if rule is None:
            return

        now = time.time()
        rule.evidence_count += 1
        if feedback == "accepted":
            rule.positive_feedback += 1
        elif feedback == "corrected":
            rule.negative_feedback += 1

        rule.last_triggered = now
        rule.last_modified = now

        # 更新置信度
        rule.confidence = wilson_score(rule.positive_feedback, rule.evidence_count)

        # 更新新鲜度
        rule.freshness = freshness_score(rule.last_triggered, rule.evidence_count, now)

        # 状态迁移
        self._migrate_state(rule)

    def _migrate_state(self, rule):
        """驱动五状态迁移"""
        now = time.time()
        days_since_created = (now - rule.created_at) / 86400.0

        if rule.state == "candidate":
            if rule.evidence_count >= 3 and rule.confidence >= 0.70:
                rule.state = "verified"
                rule.state_entered_at = now
            elif days_since_created > 7 and rule.evidence_count == 0:
                rule.state = "retired"
                rule.state_entered_at = now

        elif rule.state == "verified":
            if rule.confidence >= 0.85 and rule.negative_feedback == 0:
                rule.state = "active"
                rule.state_entered_at = now
            elif rule.negative_feedback > 0 and rule.confidence < 0.60:
                rule.state = "degraded"
                rule.state_entered_at = now
            elif rule.freshness < 0.2:
                rule.state = "degraded"
                rule.state_entered_at = now

        elif rule.state == "active":
            if rule.negative_feedback > 0:
                rule.state = "degraded"
                rule.state_entered_at = now
            elif rule.freshness < 0.3 and rule.evidence_count < 20:
                rule.state = "degraded"
                rule.state_entered_at = now

        elif rule.state == "degraded":
            if rule.confidence >= 0.85 and rule.negative_feedback == 0:
                rule.state = "active"
                rule.state_entered_at = now
            elif (now - rule.state_entered_at) / 86400.0 > RULE_RETIRE_AFTER_DAYS:
                rule.state = "retired"
                rule.state_entered_at = now

    def update_all_freshness(self):
        """周期性更新所有规则 freshNess（仿真每天调用一次）"""
        now = time.time()
        for rule in self.rules.values():
            if rule.state in ("retired",):
                continue
            rule.freshness = freshness_score(rule.last_triggered, rule.evidence_count, now)
            # 检查是否需要 degrade 由于 freshness 过低
            if rule.state == "active" and rule.freshness < 0.2:
                rule.state = "degraded"
                rule.state_entered_at = now

    # ---- 垃圾回收 ----

    def gc(self):
        """清理 retired + 限制活跃规则数"""
        now = time.time()
        # 清理旧 retired
        to_remove = []
        for rid, rule in self.rules.items():
            if rule.state == "retired" and (now - rule.state_entered_at) > 86400 * 30:
                to_remove.append(rid)
        for rid in to_remove:
            del self.rules[rid]

        # 限制活跃规则数
        active = [r for r in self.rules.values() if r.state in ("active", "verified")]
        if len(active) > RULE_MAX_ACTIVE:
            active.sort(key=lambda r: (r.confidence, r.freshness))
            for rule in active[:len(active) - RULE_MAX_ACTIVE]:
                rule.state = "degraded"
                rule.state_entered_at = now

    # ---- 统计 ----

    def stats(self):
        states = {}
        for r in self.rules.values():
            states[r.state] = states.get(r.state, 0) + 1
        return {
            "total": len(self.rules),
            "by_state": states,
            "active_count": states.get("active", 0),
            "verified_count": states.get("verified", 0),
        }

    def save_snapshot(self, path=None):
        if path is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            path = os.path.join(OUTPUT_DIR, RULES_FILE)
        snapshot = [r.to_dict() for r in self.rules.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        return path

    def load_snapshot(self, path=None):
        if path is None:
            path = os.path.join(OUTPUT_DIR, RULES_FILE)
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            rules_data = json.load(f)
        for d in rules_data:
            rule = Rule(
                d["id"], d["conditions"], d["action"],
                d.get("source", "L1"), d.get("safety_level", 1)
            )
            rule.state = d.get("state", "candidate")
            rule.evidence_count = d.get("evidence_count", 0)
            rule.positive_feedback = d.get("positive_feedback", 0)
            rule.negative_feedback = d.get("negative_feedback", 0)
            rule.confidence = d.get("confidence", 0.0)
            rule.freshness = d.get("freshness", 1.0)
            rule.specificity = d.get("specificity", rule._calc_specificity())
            self.rules[rule.id] = rule
            self._update_index(rule.id, rule)
            self._id_counter = max(self._id_counter, int(rule.id.split("_")[1]))
