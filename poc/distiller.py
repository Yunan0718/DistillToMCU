"""
DistillToMCU Phase 0a — 三源规则蒸馏器 (v2: 包容区间学习)
=========================================================
从交互轨迹中提取候选规则。
L1: 从 LLM tool_call 行为轨迹中自动学习连续传感器的包容区间 → 泛化为规则
L2: 从用户指令历史统计偏好参数
L3: 从传感器-动作时间关联检测环境触发模式
"""

import time
from collections import defaultdict
from config import ACTUATORS, SAFETY_L1, SAFETY_L2, SAFETY_L3, MAJORITY_PARAM_THRESHOLD
from rule_engine import wilson_score
from rule_generalizer import RuleGeneralizer, group_traces_by_action


def get_device_safety(device_name):
    """获取设备的安全等级，默认为 L1"""
    if device_name in ACTUATORS:
        return ACTUATORS[device_name]["safety"]
    return SAFETY_L1


class Distiller:
    """
    三源规则蒸馏器 (v2)。
    L1 使用包容区间学习替代硬编码阈值：
      - 按 (device, command) 分组所有 LLM 行为
      - RuleGeneralizer.learn_intervals() 自动学习每个传感器的触发区间
      - 不是 "temp > 28"（硬编码），而是 "temp ∈ [29.5, 32.5]"（从数据中学）
    """

    def __init__(self, rule_engine, llm_client):
        self.engine = rule_engine
        self.llm = llm_client
        self.generalizer = RuleGeneralizer(min_samples=3)
        self._last_distill_ts = 0

    def distill(self, traces):
        """
        从 trace 列表中提取候选规则。
        返回 (new_rule_count, total_candidates)
        """
        new_count = 0

        # 只处理有 tool_call 的 cloud 模式 trace
        cloud_traces = [t for t in traces
                        if t["execution"]["mode"] == "cloud"
                        and t.get("llm_response", {}).get("tool_calls")]

        candidates = []

        # L1: 从 tool_call 关联 sensor
        candidates += self._distill_l1(cloud_traces)

        # L2: 从历史统计偏好
        candidates += self._distill_l2(cloud_traces)

        # L3: 从传感器-动作时间关联
        candidates += self._distill_l3(cloud_traces)

        # 去重（相同 conditions + action 的规则合并）
        candidates = self._deduplicate(candidates)

        # v10.2: one-holdout harmful-condition pruning
        # 对每个条件做留一法：移除后新增覆盖的 trace 中，LLM 决策与该 action
        # 一致的数量 > 其他设备动作的数量 → 该条件有害（如 UCI 的 humidity
        # 下界排除真实 fan 决策），移除。迭代直到稳定或条件数 <= 2。
        candidates = [self._prune_conditions(c, cloud_traces) for c in candidates]

        # 过滤空条件规则（L2 偏好规则若无传感器条件会匹配所有情况 = 噪音）
        candidates = [c for c in candidates if c.get("conditions")]

        # LLM sanity check + 写入规则引擎
        for candidate in candidates:
            should_add, existing_rule = self._should_add(candidate)
            if should_add:
                rule_text = self._format_rule(candidate)
                # Sanity check：有 LLM 则走 LLM，否则默认通过
                if self.llm and hasattr(self.llm, 'sanity_check_rule'):
                    check = self.llm.sanity_check_rule(rule_text)
                    reasonable = check.get("reasonable", True)
                else:
                    reasonable = True  # mock 模式默认通过
                if reasonable:
                    rule = self.engine.add_rule(
                        conditions=candidate["conditions"],
                        action=candidate["action"],
                        source=candidate["source"],
                        safety_level=get_device_safety(candidate["action"]["device"]),
                    )
                    # 设置初始证据（蒸馏来源的 trace 本身就是证据）
                    if candidate.get("initial_evidence", 0) > 0:
                        rule.evidence_count = candidate["initial_evidence"]
                        rule.positive_feedback = candidate["initial_evidence"]
                        rule.confidence = wilson_score(rule.positive_feedback, rule.evidence_count)
                        # 有足够证据的可直接 verified
                        if rule.evidence_count >= 3 and rule.confidence >= 0.7:
                            rule.state = "verified"
                            rule.state_entered_at = time.time()
                    new_count += 1
            elif existing_rule is not None:
                # 规则已存在：用当前重新学习的证据覆盖（v6 fix）
                # 之前用 += inc 累加——但 candidate.initial_evidence 是每天从全量
                # traces 重算的累计值，不是增量。每天 += 累计值 → O(n²)膨胀。
                # 修复：直接 SET 为当前累计值（同新规则的逻辑一致）。
                total_ev = candidate.get("initial_evidence", 0)
                if total_ev > 0:
                    # v7: 跨 source 时取更高证据（哪个 source 样本多信哪个）
                    if total_ev > existing_rule.evidence_count:
                        existing_rule.evidence_count = total_ev
                        existing_rule.positive_feedback = total_ev
                        existing_rule.confidence = wilson_score(
                            existing_rule.positive_feedback,
                            existing_rule.evidence_count,
                        )

                    # v7: 跨 source 合并条件——取更宽的区间
                    if candidate["source"] != existing_rule.source:
                        existing_rule.source = f"{existing_rule.source}+{candidate['source']}"
                        existing_rule.conditions = self._merge_conditions(
                            existing_rule.conditions,
                            candidate.get("conditions", []),
                        )

                    existing_rule.last_modified = time.time()
                    # 证据充足 → 升级状态
                    if (existing_rule.state == "candidate"
                            and existing_rule.evidence_count >= 3
                            and existing_rule.confidence >= 0.7):
                        existing_rule.state = "verified"
                        existing_rule.state_entered_at = time.time()

        # v8: MDL rule consolidation (offline analysis, skipped during experiment)
        # v10: 传入 cloud traces，MDL 使用真实 L(Data|Rules)（覆盖 vs 云端成本）
        if candidates and not hasattr(self, '_skip_mdl'):
            from mdl_consolidator import MDLConsolidator
            mdl = MDLConsolidator(cost_threshold=0.0)
            candidates = mdl.consolidate(candidates, traces=cloud_traces)

        self._last_distill_ts = time.time()
        return new_count, len(candidates)

    # ---- L1: 包容区间学习 (v2: 替代硬编码阈值) ----

    def _distill_l1(self, cloud_traces):
        """
        从 LLM 行为轨迹中自动学习连续传感器的包容区间。

        v2 改动：
          旧方法：逐条 trace 硬编码 threshold（temp > 28, light < 50）
          新方法：先按 (device, command) 分组 → 用 RuleGeneralizer 学习区间

        这样"温度 > 28"变成了"温度 ∈ [29.5, 32.5]"——阈值来自数据，
        范围自动适配 LLM 的行为分布，可以泛化到区间内未见过的值。
        """
        candidates = []

        # Step 1: 按 action 分组传感器快照
        # 用 group_traces_by_action 提取 {(device, command): [sensors]}
        grouped = group_traces_by_action(cloud_traces)

        # Step 2: 对每组 action，用包容区间学习算法
        # v10: 传入全局背景分布，过滤无判别力条件（防止虚假相关）
        background = [t.get("sensors", {}) for t in cloud_traces]
        learned = self.generalizer.learn_from_trace_group(grouped, background)

        import json as _json

        for (device, command), conditions in learned.items():
            if not conditions:
                continue

            # 从 group 中收集参数偏好（多数投票）
            params_list = []
            for trace in cloud_traces:
                for tc in (trace.get("llm_response", {}).get("tool_calls") or []):
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    if name != f"{device}_control":
                        continue
                    args_str = func.get("arguments", "{}")
                    try:
                        args = _json.loads(args_str) if isinstance(args_str, str) else args_str
                    except _json.JSONDecodeError:
                        continue
                    if args.get("command") == command:
                        params_list.append(
                            {k: v for k, v in args.items() if k != "command"}
                        )

            # L2-style 参数偏好统计嵌入 L1
            preferred_params = self._majority_params(params_list)

            # Step 3: 构建候选规则
            # initial_evidence = 学习此规则时用的样本数
            evidence = max(c.get("sample_count", 0) for c in conditions)

            # 用 conditions（已经包含 learned lower/upper/confidence）
            # 转换为规则引擎可用的 format
            rule_conditions = self._format_learned_conditions(conditions)

            if rule_conditions:
                candidates.append({
                    "conditions": rule_conditions,
                    "action": {
                        "device": device,
                        "command": command,
                        "params": preferred_params,
                    },
                    "source": "L1",
                    "initial_evidence": evidence,
                })

        return candidates

    def _format_learned_conditions(self, learned_conditions):
        """
        将 RuleGeneralizer 输出的条件转换为规则引擎格式。

        RuleGeneralizer 输出:
          {sensor, op, lower, upper, confidence, sample_count, ...}

        规则引擎需要:
          {sensor, op, value} 或 {sensor, op, lower, upper}
        """
        formatted = []
        for c in learned_conditions:
            if c.get("op") == "between":
                formatted.append({
                    "sensor": c["sensor"],
                    "op": "gte",
                    "value": c["lower"],
                })
                formatted.append({
                    "sensor": c["sensor"],
                    "op": "lte",
                    "value": c["upper"],
                })
            elif c.get("op") == "eq":
                formatted.append({
                    "sensor": c["sensor"],
                    "op": "eq",
                    "value": c["value"],
                })
            # 其他 op 直接传递
            else:
                formatted.append({
                    "sensor": c["sensor"],
                    "op": c.get("op", "eq"),
                    "value": c.get("value", c.get("lower", 0)),
                })
        # v9: 去重 — L1+L3 合并时可能产生重复的 {sensor, op, value}
        deduped = []
        seen = set()
        for c in formatted:
            key = (c.get("sensor"), c.get("op"), c.get("value"))
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        return deduped

    @staticmethod
    def _majority_params(params_list):
        """
        从参数列表中统计多数投票的参数值。
        只保留频率 > 70% 的参数值。
        """
        if len(params_list) < 2:
            return params_list[0] if params_list else {}

        from collections import Counter
        param_values = defaultdict(list)
        for p in params_list:
            for k, v in p.items():
                param_values[k].append(v)

        result = {}
        for k, vals in param_values.items():
            if len(vals) < 2:
                continue
            counter = Counter(vals)
            most_val, most_count = counter.most_common(1)[0]
            if most_count / len(vals) > MAJORITY_PARAM_THRESHOLD:
                result[k] = most_val
        return result

    @staticmethod
    def _conds_match(conditions: list, sensors: dict) -> bool:
        """Check whether a condition list matches a sensor snapshot."""
        for c in conditions:
            sv = sensors.get(c.get("sensor"))
            if sv is None:
                continue
            op = c.get("op")
            v = c.get("value")
            if op == "gte" and not (sv >= v):
                return False
            elif op == "lte" and not (sv <= v):
                return False
            elif op == "eq" and sv != v:
                return False
        return True

    def _prune_conditions(self, candidate: dict,
                          cloud_traces: list) -> dict:
        """
        v10.2: One-holdout harmful-condition pruning.

        对每个条件做留一法：移除 c 后新增覆盖的 trace 中，LLM 决策与该
        action 一致的样本数 good、与其他设备动作一致的样本数 wrong。
        若 good >= 2 且 good > wrong → c 有害（它排除了真实 LLM 决策），移除。
        迭代直到稳定或条件数 <= 2。

        Motivation: 区间学习会把与动作弱相关/共线的传感器（如 UCI 中与
        CO2 共线的 humidity）纳入条件，其下/上界会排除真实决策。剪枝
        用训练 trace 上的留一法识别并移除这类有害条件。
        """
        conds = list(candidate.get("conditions", []))
        dev = candidate.get("action", {}).get("device")
        if len(conds) <= 2 or not cloud_traces:
            return candidate

        changed = True
        while changed and len(conds) > 2:
            changed = False
            removed = None
            for c in conds:
                reduced = [x for x in conds if x != c]
                good = wrong = 0
                for t in cloud_traces:
                    s = t.get("sensors", {})
                    if (self._conds_match(reduced, s)
                            and not self._conds_match(conds, s)):
                        acts = set()
                        for tc in (t.get("llm_response", {}).get("tool_calls")
                                   or []):
                            fn = tc.get("function", {}).get("name", "")
                            if fn.endswith("_control"):
                                acts.add(fn.replace("_control", ""))
                        if dev in acts:
                            good += 1
                        elif acts:
                            wrong += 1
                if good >= 2 and good > wrong:
                    removed = c
                    break
            if removed is not None:
                conds.remove(removed)
                changed = True

        candidate["conditions"] = conds
        return candidate

    # ---- L2: 用户偏好统计 (v7 修复：产出带传感器条件的规则) ----

    def _distill_l2(self, cloud_traces):
        """
        统计相同 device+command 下的 params 分布。
        如果某个 param 有 >70% 的一致选择 → 偏好规则。

        v7 修复：不再产出空条件规则。改为：先学传感器区间（复用 L1 的
        group_traces_by_action），再附加参数偏好。这样 L2 规则既有传感器
        触发条件又有参数偏好，不会因空条件被过滤。
        """
        candidates = []
        import json as _json

        # 按 (device, command) 分组——同时收集 params 和 sensors
        grouped_sensors = defaultdict(list)   # (device, command) → [sensors]
        grouped_params = defaultdict(list)    # (device, command) → [params]

        for trace in cloud_traces:
            sensors = trace.get("sensors", {})
            for tc in (trace.get("llm_response", {}).get("tool_calls") or []):
                func = tc.get("function", {})
                name = func.get("name", "")
                if "_control" not in name:
                    continue
                device = name.replace("_control", "")
                try:
                    args = _json.loads(func.get("arguments", "{}"))
                except _json.JSONDecodeError:
                    continue
                command = args.get("command", "on")
                params = {k: v for k, v in args.items() if k != "command"}
                grouped_sensors[(device, command)].append({"sensors": sensors})
                grouped_params[(device, command)].append(params)

        for (device, command), param_list in grouped_params.items():
            if len(param_list) < 3:
                continue

            # 统计参数偏好（与旧版相同）
            param_values = defaultdict(list)
            for p in param_list:
                for k, v in p.items():
                    param_values[k].append(v)

            pref_params = {}
            for k, vals in param_values.items():
                if len(vals) < 3:
                    continue
                from collections import Counter
                counter = Counter(vals)
                most_common_val, most_common_count = counter.most_common(1)[0]
                if most_common_count / len(vals) > MAJORITY_PARAM_THRESHOLD:
                    pref_params[k] = most_common_val

            if not pref_params:
                continue

            # v7: 用传感器快照学习区间条件（复用 RuleGeneralizer，min_samples=3）
            # v10: 传入全局背景分布，过滤无判别力条件
            sensor_snapshots = grouped_sensors.get((device, command), [])
            if len(sensor_snapshots) >= 3:
                background = [t.get("sensors", {}) for t in cloud_traces]
                conditions = self.generalizer.learn_intervals(
                    sensor_snapshots, background)
                rule_conditions = self._format_learned_conditions(conditions)
            else:
                rule_conditions = []  # 样本不够时不加传感器条件

            # 即使没有传感器条件，也保留偏好规则（有条件的更可靠）
            candidates.append({
                "conditions": rule_conditions,
                "action": {"device": device, "command": command, "params": pref_params},
                "source": "L2",
                "initial_evidence": len(param_list),
            })
        return candidates

    # ---- L3: 传感器-动作时间关联 ----

    def _distill_l3(self, cloud_traces):
        """
        检测传感器模式 → LLM 动作 之间的高频关联。

        v2: 同样使用包容区间学习，但阈值更高（min_samples=5），
        关注的是统计上显著的传感器-动作共现模式。
        与 L1 的区分：L3 不区分 device/command，而是检测
        "某个传感器模式在多种 action 中高频出现"的跨动作模式。
        """
        candidates = []

        # 按 device 分组（不按 command，发现跨命令的模式）
        device_traces = defaultdict(list)
        for trace in cloud_traces:
            sensors = trace.get("sensors", {})
            for tc in (trace.get("llm_response", {}).get("tool_calls") or []):
                func = tc.get("function", {})
                name = func.get("name", "")
                if "_control" not in name:
                    continue
                device = name.replace("_control", "")
                if device in ACTUATORS:
                    device_traces[device].append({"sensors": sensors})

        for device, snapshots in device_traces.items():
            if len(snapshots) < 5:
                continue

            # 用更高阈值（5）的 RuleGeneralizer
            # v10: 传入全局背景分布，过滤无判别力条件
            l3_generalizer = RuleGeneralizer(min_samples=5)
            background = [t.get("sensors", {}) for t in cloud_traces]
            conditions = l3_generalizer.learn_intervals(snapshots, background)

            if not conditions:
                continue

            rule_conditions = self._format_learned_conditions(conditions)
            evidence = max(c.get("sample_count", 0) for c in conditions)

            if rule_conditions:
                candidates.append({
                    "conditions": rule_conditions,
                    "action": {"device": device, "command": "on", "params": {}},
                    "source": "L3",
                    "initial_evidence": min(evidence, 20),
                })

        return candidates

    # ---- 去重（L1 相同规则合并证据）----

    def _deduplicate(self, candidates):
        merged = {}
        for c in candidates:
            key = (
                frozenset(
                    (cc["sensor"], cc["op"], str(cc.get("value", "")))
                    for cc in c["conditions"]
                ),
                c["action"]["device"],
                c["action"]["command"],
                c["source"],
            )
            if key in merged:
                # v10: evidence 是"该规则对应的 LLM 行为样本数"，
                # 完全相同的候选来自同一批样本，相加会重复计数
                # （旧版曾导致 evidence_count=2024 > trace总数 515）。
                # 修复：取 max（相同样本不重复累计）。
                merged[key]["initial_evidence"] = max(
                    merged[key].get("initial_evidence", 0),
                    c.get("initial_evidence", 0),
                )
            else:
                merged[key] = c.copy()
        return list(merged.values())

    def _should_add(self, candidate):
        """检查规则是否该添加。返回 (should_add, existing_rule_or_none)

        如果已有相同 (device, command, source) 且状态活跃的规则 → 返回已存在的规则对象，
        以便调用方更新其证据计数。这修复了证据积累停滞的 bug。

        v7 修复：如果已有相同 (device, command) 但不同 source 的规则 →
        返回那条规则，让调用方合并条件（取更宽的区间 + 更高证据），
        避免 Full Pipeline 中 L1 和 L3 产出重叠规则导致冲突 → AR 崩溃。
        """
        # 第一步：精确匹配 (device, command, source)
        for rule in self.engine.rules.values():
            if (rule.action["device"] == candidate["action"]["device"]
                    and rule.action["command"] == candidate["action"]["command"]
                    and rule.source == candidate["source"]
                    and rule.state in ("candidate", "verified", "active")):
                return False, rule

        # v7: 第二步：跨 source 匹配 (device, command) → 合并而非新建
        for rule in self.engine.rules.values():
            if (rule.action["device"] == candidate["action"]["device"]
                    and rule.action["command"] == candidate["action"]["command"]
                    and rule.state in ("candidate", "verified", "active")):
                # 不同 source 的同 action 规则 → 合并条件
                return False, rule

        return True, None

    @staticmethod
    def _merge_conditions(existing: list, new: list) -> list:
        """
        v9: 合并两个来源的条件列表。
        对同一 sensor: gte 取 min (更宽下界), lte 取 max (更宽上界),
        eq 一致则保留。每个 sensor 最多保留 gte+lte+eq 三条。
        """
        if not new:
            return existing
        if not existing:
            return new

        # Collect gte/lte/eq values per sensor from both sources
        by_sensor = {}  # sensor → {"gte": [vals], "lte": [vals], "eq": set()}
        for c in existing + new:
            s = c.get("sensor")
            if not s:
                continue
            if s not in by_sensor:
                by_sensor[s] = {"gte": [], "lte": [], "eq": set()}
            op = c.get("op", "eq")
            if op == "gte":
                by_sensor[s]["gte"].append(c.get("value", 0))
            elif op == "lte":
                by_sensor[s]["lte"].append(c.get("value", 0))
            elif op == "eq":
                by_sensor[s]["eq"].add(c.get("value"))

        merged = []
        for s, vals in by_sensor.items():
            if vals["gte"]:
                merged.append({"sensor": s, "op": "gte", "value": min(vals["gte"])})
            if vals["lte"]:
                merged.append({"sensor": s, "op": "lte", "value": max(vals["lte"])})
            if len(vals["eq"]) == 1:
                merged.append({"sensor": s, "op": "eq", "value": vals["eq"].pop()})
            # eq 有冲突 → 不加条件

        return merged

    def _format_rule(self, candidate):
        """格式化规则为可读文本，用于 LLM sanity check"""
        cond_str = " AND ".join(
            f"{c.get('sensor','')} {c.get('op','')} {c.get('value','')}"
            for c in candidate["conditions"]
        ) or "always"
        act = candidate["action"]
        params_str = ", ".join(f"{k}={v}" for k, v in act.get("params", {}).items()) or "default"
        return (f"IF {cond_str} THEN {act['device']}.{act['command']}({params_str}) "
                f"[source: {candidate['source']}]")
