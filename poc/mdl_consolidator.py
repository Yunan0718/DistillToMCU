"""
DistillToMCU v8 — Sensor-Aware MDL Rule Consolidation
=====================================================
C1 核心贡献：用 Minimum Description Length (MDL) 原则控制规则粒度和数量。

对标 RIMRULE (Gao, Baidya et al., ACL 2026):
  RIMRULE: LLM失败trace → MDL符号规则 → 注入回LLM prompt → LLM继续运行
  我们:     LLM成功trace → MDL传感器区间规则 → 部署到MCU → MCU独立执行

关键区别:
  - RIMRULE 的 MDL 作用在符号空间（NL tokens），我们作用在物理传感器空间
  - 我们的编码方案考虑传感器物理范围、精度、离散化粒度
  - 输出的规则部署到 MCU 而非注入回 LLM

MDL 原理 (Grunwald 2007):
  Total Cost = L(RuleBase) + L(Data | RuleBase)
  选择最小化总成本的规则库。

算法:
  1. 计算每对规则的合并收益 ΔCost = Cost(merged) - (Cost(R1) + Cost(R2))
  2. 如果 ΔCost < 0（合并减少总成本），合并
  3. 贪心迭代直到收敛

Usage:
    from mdl_consolidator import MDLConsolidator, consolidate_rules

    cons = MDLConsolidator()
    consolidated = cons.consolidate(rules)
"""

import math
from collections import defaultdict


class MDLConsolidator:
    """
    Sensor-Aware MDL 规则合并器。

    编码方案:
      L(Rule) = 规则ID + 每个sensor条件的bits + action的bits
      L(Data|Rules) = 每条trace的覆盖成本

    传感器物理范围用于决定编码精度:
      temperature: -20~50°C, 精度 0.5°C → 140 个离散值 → ~7 bits
      humidity:    0~100%, 精度 1% → 100 个离散值 → ~7 bits
      light:       0~2000 lux, 精度 10 lux → 200 个离散值 → ~8 bits
    """

    # 传感器物理范围和精度 (用于 MDL 编码)
    SENSOR_CODING = {
        "temperature": {"min": -20.0, "max": 50.0, "precision": 0.5},
        "humidity":    {"min":   0.0, "max": 100.0, "precision": 1.0},
        "light":       {"min":   0.0, "max": 2000.0, "precision": 10.0},
        "co2":         {"min":   0.0, "max": 3000.0, "precision": 50.0},
    }

    # 默认值
    DEFAULT_CODING = {"min": 0.0, "max": 1000.0, "precision": 10.0}

    # v10: 云端兜底编码成本（bits）。未命中规则的 trace 需要调用云端 LLM，
    # 其成本远高于本地编码规则 id 的成本。该常数只影响合并的相对偏好。
    CLOUD_COST_BITS = 15.0

    def __init__(self, cost_threshold: float = 0.0):
        """
        Args:
            cost_threshold: 合并的成本阈值。负数表示仅当合并明确降低总成本时才合并。
                            0 = 只要不增加成本就合并。正数 = 即使稍微增加成本也合并。
        """
        self.cost_threshold = cost_threshold

    def consolidate(self, rules: list[dict],
                    traces: list[dict] | None = None) -> list[dict]:
        """
        对规则列表进行 MDL 贪心合并。

        Args:
            rules: [{conditions, action, source, ...}, ...]

        Returns:
            合并后的规则列表
        """
        if len(rules) <= 1:
            return rules

        current = [dict(r) for r in rules]  # 深拷贝
        changed = True

        while changed:
            changed = False
            n = len(current)

            # 计算所有对的合并成本
            best_delta = float("inf")
            best_pair = (-1, -1)
            best_merged = None

            for i in range(n):
                for j in range(i + 1, n):
                    merged = self._try_merge(current[i], current[j])
                    if merged is None:
                        continue
                    delta = self._total_cost([merged], traces) \
                        - self._total_cost([current[i], current[j]], traces)
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (i, j)
                        best_merged = merged

            # 如果最佳合并确实降低成本，执行
            if best_delta <= self.cost_threshold and best_merged is not None:
                i, j = best_pair
                # 删除 j（索引较大），替换 i
                if j > i:
                    del current[j]
                    current[i] = best_merged
                else:
                    del current[i]
                    current[j] = best_merged
                changed = True

        return current

    def _try_merge(self, r1: dict, r2: dict) -> dict | None:
        """
        尝试合并两条规则。如果不可合并，返回 None。

        可合并条件:
          - 相同 device + command（同一 action）
          - 条件中的 sensor 集合可重叠
        """
        a1 = r1.get("action", {})
        a2 = r2.get("action", {})
        if a1.get("device") != a2.get("device"):
            return None
        if a1.get("command") != a2.get("command"):
            return None

        conds1 = self._normalize_conditions(r1.get("conditions", []))
        conds2 = self._normalize_conditions(r2.get("conditions", []))

        # 合并条件
        merged_conds = self._merge_condition_sets(conds1, conds2)

        if merged_conds is None:
            return None

        # 合并后的规则继承更高证据
        ev1 = r1.get("evidence_count", r1.get("sample_count", 0))
        ev2 = r2.get("evidence_count", r2.get("sample_count", 0))
        total_ev = max(ev1, ev2) + min(ev1, ev2) * 0.5  # discount overlapping evidence

        sources = set(r1.get("source", "").split("+") +
                      r2.get("source", "").split("+"))

        return {
            "conditions": merged_conds,
            "action": dict(a1),
            "source": "+".join(sorted(s for s in sources if s)),
            "initial_evidence": int(total_ev),
        }

    def _normalize_conditions(self, conditions: list[dict]) -> list[tuple[str, str, float]]:
        """
        将条件列表标准化为 [(sensor, op, value), ...] 的列表。
        保留所有条件（同sensor的gte和lte是两个独立条件）。
        """
        result = []
        for c in conditions:
            s = c.get("sensor", "")
            op = c.get("op", "eq")
            v = c.get("value", 0)
            result.append((s, op, v))
        return result

    def _merge_condition_sets(self,
                               c1: list[tuple[str, str, float]],
                               c2: list[tuple[str, str, float]]) -> list[dict] | None:
        """
        合并两个条件集合。对同一 sensor:
          - gte → 取更小的 lower (更宽)
          - lte → 取更大的 upper (更宽)
          - eq  → 一致则保留，不一致则返回 None（不可合并）
        """
        # Group by sensor
        by_sensor = defaultdict(lambda: {"eq": set(), "gte": [], "lte": []})
        for s, op, v in c1 + c2:
            if op == "eq":
                by_sensor[s]["eq"].add(v)
            elif op == "gte":
                by_sensor[s]["gte"].append(v)
            elif op == "lte":
                by_sensor[s]["lte"].append(v)

        merged = []
        for s, ops in by_sensor.items():
            # eq: 必须一致
            if len(ops["eq"]) > 1:
                return None
            if ops["eq"]:
                merged.append({"sensor": s, "op": "eq", "value": ops["eq"].pop()})

            # gte: 取 min (更宽的下界)
            if ops["gte"]:
                merged.append({"sensor": s, "op": "gte", "value": min(ops["gte"])})

            # lte: 取 max (更宽的上界)
            if ops["lte"]:
                merged.append({"sensor": s, "op": "lte", "value": max(ops["lte"])})

        return merged

    # ---- MDL Cost Functions ----

    def _total_cost(self, rules: list[dict],
                    traces: list[dict] | None = None) -> float:
        """Total MDL cost = L(Rules) + L(Data|Rules)"""
        return self._model_cost(rules) + self._data_cost(rules, traces)

    def _model_cost(self, rules: list[dict]) -> float:
        """
        L(Rules): 规则库的编码成本 (bits)。

        每条规则:
          - 规则 ID: log2(K) bits (K = 规则总数)
          - device: log2(3) ≈ 1.6 bits (led/fan/curtain)
          - command: log2(3) ≈ 1.6 bits (on/off/set)
          - 每个 sensor 条件: 编码区间宽度
        """
        K = max(1, len(rules))
        cost = 0.0

        for r in rules:
            cost += math.log2(K)  # rule ID
            cost += 1.6  # device
            cost += 1.6  # command

            conditions = r.get("conditions", [])
            for c in conditions:
                s = c.get("sensor", "")
                coding = self.SENSOR_CODING.get(s, self.DEFAULT_CODING)
                phys_range = coding["max"] - coding["min"]
                precision = coding["precision"]
                n_bins = phys_range / precision

                op = c.get("op", "eq")
                if op in ("gte", "lte"):
                    # 编码一个边界值: log2(n_bins) bits
                    cost += math.log2(max(1, n_bins))
                elif op == "eq":
                    # 离散值: log2(n_bins) bits
                    cost += math.log2(max(1, n_bins))

        return cost

    def _data_cost(self, rules: list[dict],
                   traces: list[dict] | None = None) -> float:
        """
        L(Data|Rules): 数据在规则库下的编码成本。

        简化版本: 每条规则覆盖的数据量作为代理。
        规则越精准 → 覆盖越少 → 数据成本越高 → 倾向于合并。

        实际系统中，这是通过 trace replay 来估计的。
        """
        if not traces:
            # 无数据时退化为规则数量正则（保持旧行为，避免报错）
            K = max(1, len(rules))
            return -K * math.log2(K) * 0.1

        # v10: 真实数据编码成本 —— 对每条 trace 模拟规则匹配：
        #   命中规则 → log2(K) bits（记录 rule id）
        #   未命中   → CLOUD_COST_BITS（云端 LLM 兜底）
        # 合并使规则更宽 → 覆盖更多 trace（数据成本↓），但模型成本↑，
        # MDL 在这两者间找到最优规则库。
        K = max(1, len(rules))
        total = 0.0
        matched_cost = math.log2(K)

        for t in traces:
            sensors = t.get("sensors", t) if isinstance(t, dict) else t
            hit = False
            for r in rules:
                if self._matches_rule(r, sensors):
                    hit = True
                    break
            total += matched_cost if hit else self.CLOUD_COST_BITS

        return total

    @staticmethod
    def _matches_rule(rule: dict, sensors: dict) -> bool:
        """Simplified condition matching; missing sensor field = pass (matches
        the firmware _chk() semantics)."""
        conditions = rule.get("conditions", [])
        if not conditions:
            return True
        for c in conditions:
            s = c.get("sensor")
            op = c.get("op")
            v = c.get("value", c.get("lower"))
            sv = sensors.get(s) if isinstance(sensors, dict) else None
            if sv is None:
                continue
            if op == "gt" and not (sv > v):
                return False
            elif op == "lt" and not (sv < v):
                return False
            elif op == "gte" and not (sv >= v):
                return False
            elif op == "lte" and not (sv <= v):
                return False
            elif op == "eq" and sv != v:
                return False
            elif op not in ("gt", "lt", "gte", "lte", "eq"):
                return False
        return True


def consolidate_rules(rules: list[dict],
                      cost_threshold: float = 0.0) -> list[dict]:
    """
    便捷函数：对规则列表执行 MDL 合并。
    """
    cons = MDLConsolidator(cost_threshold=cost_threshold)
    return cons.consolidate(rules)


# ========== Self-Test ==========

if __name__ == "__main__":
    print("=" * 60)
    print("  MDL Consolidator — Self-Test")
    print("=" * 60)

    cons = MDLConsolidator(cost_threshold=0.0)

    # Test 1: 可合并 —— 相同action，重叠传感器
    print("\n[Test 1] Mergeable rules with overlapping sensor conditions")
    rules1 = [
        {"conditions": [
            {"sensor": "temperature", "op": "gte", "value": 29.0},
            {"sensor": "temperature", "op": "lte", "value": 33.0},
        ], "action": {"device": "fan", "command": "on"}, "source": "L1",
         "evidence_count": 5},
        {"conditions": [
            {"sensor": "temperature", "op": "gte", "value": 28.5},
            {"sensor": "temperature", "op": "lte", "value": 32.5},
            {"sensor": "light", "op": "lte", "value": 100.0},
        ], "action": {"device": "fan", "command": "on"}, "source": "L3",
         "evidence_count": 3},
    ]
    result = cons.consolidate(rules1)
    print(f"  Input: {len(rules1)} rules → Output: {len(result)} rules")
    assert len(result) == 1, f"Should merge into 1 rule, got {len(result)}"
    merged = result[0]
    # 验证合并后的条件：温度下界=min(29.0,28.5)=28.5, 上界=max(33.0,32.5)=33.0
    conds_by_sensor_op = {(c["sensor"], c["op"]): c["value"] for c in merged["conditions"]}
    assert conds_by_sensor_op.get(("temperature", "gte")) == 28.5, \
        f"gte should be 28.5 (wider), got {conds_by_sensor_op}"
    assert conds_by_sensor_op.get(("temperature", "lte")) == 33.0, \
        f"lte should be 33.0 (wider), got {conds_by_sensor_op}"
    # light lte from r2 should be preserved
    assert ("light", "lte") in conds_by_sensor_op
    print(f"  Merged source: {merged['source']}")
    print(f"  Merged evidence: {merged.get('initial_evidence')}")
    print(f"  Merged conditions: {merged['conditions']}")
    print("  [PASS]")

    # Test 2: 不可合并 —— 不同 action
    print("\n[Test 2] Unmergeable — different actions")
    rules2 = [
        {"conditions": [{"sensor": "temperature", "op": "gte", "value": 30}],
         "action": {"device": "fan", "command": "on"}, "source": "L1"},
        {"conditions": [{"sensor": "light", "op": "lt", "value": 50}],
         "action": {"device": "led", "command": "on"}, "source": "L1"},
    ]
    result2 = cons.consolidate(rules2)
    print(f"  Input: {len(rules2)} rules → Output: {len(result2)} rules")
    assert len(result2) == 2, "Different actions should not merge"
    print("  [PASS]")

    # Test 3: 不可合并 —— 离散条件冲突
    print("\n[Test 3] Unmergeable — conflicting discrete conditions")
    rules3 = [
        {"conditions": [{"sensor": "motion", "op": "eq", "value": 1}],
         "action": {"device": "led", "command": "on"}, "source": "L1"},
        {"conditions": [{"sensor": "motion", "op": "eq", "value": 0}],
         "action": {"device": "led", "command": "on"}, "source": "L1"},
    ]
    result3 = cons.consolidate(rules3)
    print(f"  Input: {len(rules3)} rules → Output: {len(result3)} rules")
    assert len(result3) == 2, "Conflicting eq should not merge"
    print("  [PASS]")

    # Test 4: MDL 成本下降验证
    print("\n[Test 4] MDL cost reduction verification")
    r1 = {"conditions": [
        {"sensor": "temperature", "op": "gte", "value": 28}, {"sensor": "temperature", "op": "lte", "value": 34}
    ], "action": {"device": "fan", "command": "on"}, "source": "L1", "evidence_count": 5}
    r2 = {"conditions": [
        {"sensor": "temperature", "op": "gte", "value": 29}, {"sensor": "temperature", "op": "lte", "value": 33}
    ], "action": {"device": "fan", "command": "on"}, "source": "L3", "evidence_count": 3}

    cost_before = cons._total_cost([r1, r2])
    merged = cons._try_merge(r1, r2)
    cost_after = cons._total_cost([merged]) if merged else float("inf")
    print(f"  Cost before merge: {cost_before:.2f} bits")
    print(f"  Cost after merge:  {cost_after:.2f} bits")
    print(f"  Delta: {cost_after - cost_before:+.2f} bits")
    assert cost_after < cost_before, "MDL should prefer merging overlapping rules"
    print("  [PASS]")

    # Test 5: 空输入
    print("\n[Test 5] Empty input")
    assert cons.consolidate([]) == []
    assert cons.consolidate([rules1[0]]) == [rules1[0]]  # 单规则不变
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("  ALL MDL CONSOLIDATOR TESTS PASSED [OK]")
    print("=" * 60)
