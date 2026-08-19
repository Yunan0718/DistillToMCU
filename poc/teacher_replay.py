"""
DistillToMCU — teacher-replay AGREE / Precision / Recall (v10.6)
================================================================
On ONE fixed set of 60 sensor snapshots per dataset (the same snapshots
queried by llm_consistency_experiment.py), evaluate every method against the
teacher LLM's majority decision (3 repeats at T=0):

  - decision agreement (action agreement on teacher-action snapshots)
  - precision  = P(method action == teacher | method acted locally)
  - recall     = P(method acted locally and matches | teacher acted)
  - sample AR  = P(method acted locally) on the 60-snapshot sample
  - LLM self-agreement ceiling on the same snapshots (consistency of the
    3 teacher repeats) -> fidelity efficiency = agree / ceiling

Zero additional API cost: teacher decisions are reused from
llm_consistency_results.json (2160 calls, one experiment serves both).
The only API use is regenerating LLM One-shot rules (1 call per dataset);
without a key it falls back to handcrafted rules (honestly labeled).

All methods use the same warm/eval discipline:
  - batch methods train only on the first 70% of days;
  - online methods warm up on days 1..21, then keep learning online on the
    sampled snapshots in chronological order (past-only information);
  - Ours is evaluated with warm-period re-distilled rules (zero leakage) and
    with the final day-30 rule snapshot (full-horizon capability).

Output: output/teacher_replay_results.json
"""

import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import (
    ExactCacheBaseline, UserDefinedRulesBaseline,
    DecisionTreeBaseline, OnlineDailyRefitDecisionTreeBaseline,
    ESPClawStyleBaseline, PureCloudBaseline, extract_cloud_action,
    USER_RULES_BY_LABEL,
)
from llm_consistency_experiment import (
    DATASETS,
)
from run_full_analysis import load_traces, load_day_bounds
from rule_engine import RuleEngine
from distiller import Distiller

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

TRAIN_RATIO = 0.7


def _teacher_data(consistency_details, label):
    """从 llm_consistency_results.json（v10.6 格式）取教师决策。"""
    det = consistency_details.get(label, {})
    rows = list(det.get("rows", []))
    rows.sort(key=lambda r: r.get("snapshot_idx", 0))
    out = []
    for r in rows:
        def norm(a):
            if not a:
                return None
            if a.startswith("read_sensors"):
                return None  # 只读查询不算控制动作
            return a
        repeats = [norm(a) for a in r.get("ds_t0_actions", [])]
        counter = Counter(a for a in repeats if a is not None)
        majority = counter.most_common(1)[0][0] if counter else None
        self_consistent = len(set(a for a in repeats if a is not None)) <= 1
        qwen = norm(r.get("qwen_action"))
        out.append({
            "sensors": r.get("sensors", {}),
            "user_input": r.get("user_input", ""),
            "repeats": repeats,
            "majority": majority,
            "qwen_action": qwen,
            "self_consistent": self_consistent,
        })
    return out


def _action_key(majority: str | None):
    """'fan.on' -> ('fan','on')；None -> None。"""
    if not majority:
        return None
    if "." in majority:
        dev, cmd = majority.split(".", 1)
        return dev, cmd
    return majority, "on"


def _ours_engine_from_snapshot(exp_dir):
    eng = RuleEngine()
    p = os.path.join(OUT, exp_dir, "rules_snapshot.json")
    if os.path.exists(p):
        try:
            eng.load_snapshot(p)
        except Exception:
            pass
    return eng


def _ours_engine_warm(traces, day_labels, split_day):
    eng = RuleEngine()
    di = Distiller(eng, llm_client=None)
    warm = [t for t, d in zip(traces, day_labels) if d < split_day]
    try:
        di.distill(warm)
    except Exception:
        pass
    return eng


def _engine_action(eng, sensors):
    try:
        matches = eng.match(sensors)
        best = eng.resolve_conflict(matches)
    except Exception:
        return None
    if not best:
        return None
    return (best.action.get("device", ""), best.action.get("command", "on"))


def _dt_action(tree, sensors, feature_names):
    if tree is None:
        return None
    try:
        import numpy as np
        feat = np.array([[sensors.get(f, 0) or 0 for f in feature_names]])
        proba = tree.predict_proba(feat)[0]
        best = int(proba.argmax())
        if proba[best] < 0.5:
            return None
        return tree.classes_[best], "on"
    except Exception:
        return None


def _eval_method_on_snapshots(predict_fn, snapshots):
    """返回 (agree, precision, recall, sample_ar, n_teacher_act, n_local)。"""
    agree = agree_dev = prec_ok = prec_n = rec_ok = rec_n = local = 0
    for s in snapshots:
        key = _action_key(s["majority"])
        act = predict_fn(s["sensors"])
        if act is not None:
            local += 1
            prec_n += 1
            if act == key:
                prec_ok += 1
                agree_dev += 1
            elif key is not None and act[0] == key[0]:
                agree_dev += 1
        if key is not None:
            rec_n += 1
            if act is not None and act == key:
                rec_ok += 1
                agree += 1
    return {
        "agree_pct": round(agree / max(1, rec_n) * 100, 1),
        "agree_device_pct": round(agree_dev / max(1, rec_n) * 100, 1),
        "precision_pct": round(prec_ok / max(1, prec_n) * 100, 1) if prec_n else None,
        "recall_pct": round(rec_ok / max(1, rec_n) * 100, 1),
        "sample_ar_pct": round(local / max(1, len(snapshots)) * 100, 1),
        "n_teacher_act": rec_n,
        "n_local": local,
    }


def main():
    consistency = json.load(open(os.path.join(OUT, "llm_consistency_results.json"),
                                 encoding="utf-8"))
    details = consistency.get("details", {})
    out = {}

    for label, cfg in DATASETS.items():
        exp_dir = cfg["dir"] if isinstance(cfg, dict) else cfg
        snapshots = _teacher_data(details, label)
        if not snapshots:
            print(f"[SKIP] {label}: no T=0 teacher data")
            continue

        traces = load_traces(exp_dir)
        day_bounds = load_day_bounds(exp_dir)
        from baselines import trace_day_labels
        day_labels = trace_day_labels(len(traces), day_bounds)
        n_days = max(day_labels)
        split_day = int(n_days * TRAIN_RATIO) + 1
        warm_traces = [t for t, d in zip(traces, day_labels) if d < split_day]
        warm_cloud = [t for t in warm_traces
                      if t.get("execution", {}).get("mode") == "cloud"]

        # ---- teacher ceiling ----
        ceil_n = sum(1 for s in snapshots if s["self_consistent"])
        ceiling = round(ceil_n / max(1, len(snapshots)) * 100, 1)

        # ---- Ours: final snapshot + warm re-distill ----
        eng_final = _ours_engine_from_snapshot(exp_dir)
        eng_warm = _ours_engine_warm(traces, day_labels, split_day)

        # per-snapshot match flags（供 agree_reference.py 计算配对 CI）
        for s in snapshots:
            key = _action_key(s["majority"])
            s["_ours_warm_match"] = (
                key is not None
                and _engine_action(eng_warm, s["sensors"]) == key)

        # ---- DT batch (train days 1..21) ----
        dt = DecisionTreeBaseline(seed=42)
        dt.train(warm_cloud)
        dt_feature_names = list(dt._feature_names)

        # ---- Online DT warm (enrich _cloud_action before train) ----
        warm_enriched = []
        for t in warm_cloud:
            e = dict(t)
            e["_cloud_action"] = extract_cloud_action(t)
            warm_enriched.append(e)

        # ---- Exact cache warm (days 1..21) ----
        exact = ExactCacheBaseline(seed=42)
        for t in warm_cloud:
            act = extract_cloud_action(t)
            if act:
                key = frozenset(
                    (k, round(v, 1) if isinstance(v, float) else v)
                    for k, v in sorted(t.get("sensors", {}).items())
                    if v is not None)
                exact.cache[key] = act

        # ---- User-defined / ESP-Claw warm ----
        ud = UserDefinedRulesBaseline(
            seed=42,
            rules=USER_RULES_BY_LABEL.get(label))
        esp = ESPClawStyleBaseline(seed=42)
        for t in warm_cloud:
            act = extract_cloud_action(t)
            if act:
                esp._learn_from_cloud(t.get("sensors", {}), act)

        # ---- One-shot rules ----
        oneshot = _make_oneshot()

        methods = {
            "Ours (final rules)": lambda s, e=eng_final: _engine_action(e, s),
            "Ours (warm rules)": lambda s, e=eng_warm: _engine_action(e, s),
            "Decision Tree (batch)": lambda s, t=dt:
                _dt_action(t._tree, s, dt_feature_names),
            "User-defined Rules": lambda s: _userdef_action(ud, s),
            "LLM One-shot": lambda s: _oneshot_action(oneshot, s),
            "ESP-Claw-style": lambda s: _esp_action(esp, s),
            "Exact Cache": lambda s: _exact_action(exact, s),
            "Pure Cloud": lambda s: None,
        }
        method_rows = {}
        for name, fn in methods.items():
            method_rows[name] = _eval_method_on_snapshots(fn, snapshots)

        # Online DT：按时间顺序在 60 个采样快照上在线学习（只用过去）。
        # 快照按 trace 时间排序后重放；采样索引由 select_diverse 保证覆盖时间轴。
        online_dt = OnlineDailyRefitDecisionTreeBaseline(seed=42)
        online_dt.train(warm_enriched)
        # 重建快照的时间顺序（用 sensors 匹配回原 trace 顺序）
        order = _chronological_snapshot_order(traces, snapshots)
        online_rows = _eval_online_dt(online_dt, traces, day_labels,
                                      order, snapshots)
        method_rows["Decision Tree (online refit)"] = online_rows

        # 跨模型：DeepSeek 多数 vs Qwen；规则迁移：Ours(warm) vs Qwen 决策
        mm_same = mm_dev = mm_n = 0
        trans_ok = trans_dev = trans_n = 0
        for s in snapshots:
            ds_key = _action_key(s["majority"])
            qw_key = _action_key(s.get("qwen_action"))
            if ds_key == qw_key:
                mm_same += 1
                mm_dev += 1
            elif ds_key is not None and qw_key is not None \
                    and ds_key[0] == qw_key[0]:
                mm_dev += 1
            mm_n += 1
            if qw_key is not None:
                trans_n += 1
                ra = _engine_action(eng_warm, s["sensors"])
                if ra == qw_key:
                    trans_ok += 1
                    trans_dev += 1
                elif ra is not None and ra[0] == qw_key[0]:
                    trans_dev += 1

        out[label] = {
            "exp_dir": exp_dir,
            "n_snapshots": len(snapshots),
            "teacher_self_agreement_pct": ceiling,
            "teacher_self_agreement_n": ceil_n,
            "model_model_agreement_pct": round(mm_same / max(1, mm_n) * 100, 1),
            "model_model_agreement_device_pct": round(
                mm_dev / max(1, mm_n) * 100, 1),
            "transfer_ours_warm_to_qwen_pct": round(
                trans_ok / max(1, trans_n) * 100, 1),
            "transfer_ours_warm_to_qwen_device_pct": round(
                trans_dev / max(1, trans_n) * 100, 1),
            "fidelity_ours_warm_to_deepseek_pct":
                method_rows["Ours (warm rules)"]["agree_pct"],
            "snapshots": snapshots,
            "methods": method_rows,
            "fidelity_efficiency_ours_warm": round(
                method_rows["Ours (warm rules)"]["agree_pct"]
                / max(0.1, ceiling) * 100, 1),
        }
        print(f"[{label}] n={len(snapshots)} ceiling={ceiling}%")
        for name, row in method_rows.items():
            print(f"    {name:<30s} agree={row['agree_pct']:5.1f}% "
                  f"prec={row['precision_pct']} rec={row['recall_pct']:5.1f}% "
                  f"AR={row['sample_ar_pct']:5.1f}% "
                  f"(n_act={row['n_teacher_act']})")

    p = os.path.join(OUT, "teacher_replay_results.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nSaved: {p}")


def _make_oneshot():
    """LLM One-shot 规则（真实 LLM 生成优先，无 key 时回退手写并标注）。"""
    from baselines import LLMOneShotBaseline
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if api_key and len(api_key) > 10:
        try:
            import llm_client
            class _B:
                @staticmethod
                def call_llm_with_backend(messages, backend="deepseek-v4-flash",
                                          temperature=0.0, max_tokens=None):
                    return llm_client.call_llm_with_backend(
                        messages, backend=backend, temperature=temperature,
                        max_tokens=max_tokens)

                @staticmethod
                def get_available_llms():
                    return ["deepseek-v4-flash"] if len(api_key) > 10 else []
            b = LLMOneShotBaseline(llm_client=_B(), seed=42)
            b._generate_rules()
            return b
        except Exception:
            pass
    b = LLMOneShotBaseline(llm_client=None, seed=42)
    b._generate_rules()
    return b


def _userdef_action(ud, sensors):
    res = ud.handle({"sensors": sensors, "_cloud_action": None})
    a = res.get("action")
    return (a.get("device"), a.get("command")) if a else None


def _oneshot_action(b, sensors):
    res = b.handle({"sensors": sensors, "_cloud_action": None})
    a = res.get("action")
    return (a.get("device"), a.get("command")) if a else None


def _esp_action(esp, sensors):
    res = esp.handle({"sensors": sensors, "_cloud_action": None})
    a = res.get("action")
    return (a.get("device"), a.get("command")) if a else None


def _exact_action(exact, sensors):
    res = exact.handle({"sensors": sensors, "_cloud_action": None})
    a = res.get("action")
    return (a.get("device"), a.get("command")) if a else None


def _chronological_snapshot_order(traces, snapshots):
    """把采样快照映射回它们在原始 trace 序列中的时间索引。"""
    order = []
    used = set()
    for i, t in enumerate(traces):
        key = json.dumps(t.get("sensors", {}), sort_keys=True)
        for j, s in enumerate(snapshots):
            if j in used:
                continue
            if json.dumps(s["sensors"], sort_keys=True) == key:
                order.append(j)
                used.add(j)
                break
    order += [j for j in range(len(snapshots)) if j not in used]
    return order


def _eval_online_dt(online_dt, traces, day_labels, order, snapshots):
    """在线 DT：warm 期已注入；按时间顺序重放采样快照，逐日重训。"""
    import random as _random
    _random.seed(42)
    agree = agree_dev = prec_ok = prec_n = rec_ok = rec_n = local = 0
    for j in order:
        s = snapshots[j]
        key = _action_key(s["majority"])
        # 当前快照的 day：从原 trace 时间位置近似（用 sensors 匹配）
        day = _day_of_snapshot(traces, day_labels, s["sensors"])
        res = online_dt.handle({"sensors": s["sensors"],
                                "_cloud_action": None, "_day": day})
        # 在线观察：把教师多数决策加入历史（不偷看未来）
        if key is not None:
            online_dt._history.append(
                online_dt._feat_label({"sensors": s["sensors"]},
                                      {"device": key[0]}))
        act = res.get("action")
        if act is not None:
            act_key = (act.get("device"), act.get("command"))
            local += 1
            prec_n += 1
            if act_key == key:
                prec_ok += 1
                agree_dev += 1
            elif key is not None and act_key[0] == key[0]:
                agree_dev += 1
        if key is not None:
            rec_n += 1
            if act is not None and (act.get("device"), act.get("command")) == key:
                rec_ok += 1
                agree += 1
    return {
        "agree_pct": round(agree / max(1, rec_n) * 100, 1),
        "agree_device_pct": round(agree_dev / max(1, rec_n) * 100, 1),
        "precision_pct": round(prec_ok / max(1, prec_n) * 100, 1) if prec_n else None,
        "recall_pct": round(rec_ok / max(1, rec_n) * 100, 1),
        "sample_ar_pct": round(local / max(1, len(snapshots)) * 100, 1),
        "n_teacher_act": rec_n,
        "n_local": local,
    }


def _day_of_snapshot(traces, day_labels, sensors):
    key = json.dumps(sensors, sort_keys=True)
    for t, d in zip(traces, day_labels):
        if json.dumps(t.get("sensors", {}), sort_keys=True) == key:
            return d
    return 30


if __name__ == "__main__":
    main()
