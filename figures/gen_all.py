#!/usr/bin/env python3
"""
DistillToMCU — 论文级图表生成 (v7.0)
=====================================
按 IEEE/Elsevier 期刊规范生成数据图表：
  - Times 衬线字体、8-10pt、300 DPI、PDF 矢量 + PNG 栅格
  - Okabe-Ito 色盲安全配色，突出"我们的方法"(DistillToMCU)
  - 误差棒/置信带、直接标签、无顶/右边框

数据源：poc/output/ 下 6 组实验 + baseline_results + statistics_results +
        ablation_results + chat_experiments（存在时）。
输出：figures/fig_*.pdf + fig_*.png
"""

import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "poc", "output")
FIG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "poc"))
import statistics_tests as st  # 项目自带 Friedman/Nemenyi/Bootstrap（无 scipy 依赖）

# ---------------- 期刊风格 ----------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
})

OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
         "#D55E00", "#CC79A7", "#999999"]
OURS = "#D55E00"  # 朱红，色盲安全且突出
FIG_SINGLE, FIG_FULL = (3.35, 2.6), (6.9, 2.9)

EXPERIMENTS = [
    # v10.7: representative run of each 4x dataset (repeat seed 42)
    ("run4b_seed42_seed42", "seed42", "Seed 42 (Syn)"),
    ("run4b_seed123_seed42", "seed123", "Seed 123 (Syn)"),
    ("run4b_seed999_seed42", "seed999", "Seed 999 (Syn)"),
    ("run4b_seed777_seed42", "seed777", "Seed 777 (Syn)"),
    ("run4b_strands_seed42", "strands", "STRANDS Aruba-1"),
    ("run4b_uci_seed42", "uci_v3", "UCI V3 (Real)"),
    ("run4b_sml2010_seed42", "sml2010", "SML2010 (Real)"),
    ("run4b_steel_seed42", "steel", "Steel Ind. (Real)"),
    ("run4b_airquality_seed42", "airquality", "Air Quality (Real)"),
]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(exp_dir):
    p = os.path.join(OUT, exp_dir, "metrics.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_traces(exp_dir):
    p = os.path.join(OUT, exp_dir, "traces.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def trace_mode(t):
    """兼容 PC trace（execution.mode）与 MCU trace（exec_mode）。"""
    m = t.get("exec_mode")
    if not m:
        m = (t.get("execution") or {}).get("mode")
    return m


def trace_latency(t):
    """PC trace 在 llm_response.latency_ms；MCU trace 在顶层 latency_ms。"""
    lat = (t.get("llm_response") or {}).get("latency_ms")
    if lat is None:
        lat = t.get("latency_ms")
    return float(lat) if lat is not None else None


def trace_has_action(t):
    """动作判定：顶层 action 或 llm_response.tool_calls 非空。"""
    if t.get("action"):
        return True
    return bool((t.get("llm_response") or {}).get("tool_calls"))


T95_N3 = 4.303  # t(0.975, df=2)，3 个种子的 95% CI 用 t 分布而非正态 1.96
T95_N4 = 3.182  # t(0.975, df=3)，4 个种子的 95% CI（v10: 加入 seed777）


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)
    print(f"[fig] {name}.pdf / .png")


def pct(arr, p):
    a = sorted(arr)
    if not a:
        return 0.0
    k = (len(a) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return a[lo]
    return a[lo] + (a[hi] - a[lo]) * (k - lo)


# ============================================================
# 图 1: 自主率增长曲线（3 种子均值±95%CI + 真实数据集）
# ============================================================
def fig_ar_learning():
    seeds = ["seed42", "seed123", "seed999", "seed777"]
    curves = [load_metrics(d) for d in
              ["seed42", "run_seed123", "run_seed999", "seed777"]]
    curves = [c for c in curves if c]
    if len(curves) < 2:
        return
    n = max(len(c) for c in curves)
    days = np.arange(1, n + 1)
    mat = np.full((len(curves), n), np.nan)
    for i, c in enumerate(curves):
        for m in c:
            if 1 <= m["day"] <= n:
                mat[i, m["day"] - 1] = m["autonomy_rate"]
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0, ddof=1)
    se = std / np.sqrt(len(curves))
    t_crit = T95_N4 if len(curves) == 4 else T95_N3
    fig, axes = plt.subplots(1, 2, figsize=FIG_FULL)
    ax = axes[0]
    for i, (d, c) in enumerate(zip(seeds, curves)):
        x = [m["day"] for m in c]
        y = [m["autonomy_rate"] for m in c]
        ax.plot(x, y, color=OKABE[i], marker="o", markevery=max(1, len(x) // 8),
                label=d.replace("seed", "Seed "), lw=1.3)
    ax.plot(days, mean, color=OURS, lw=2.2, label="Mean (ours)")
    ax.fill_between(days, mean - t_crit * se, mean + t_crit * se,
                    color=OURS, alpha=0.15, label=f"95% CI (t, n={len(curves)})")
    # v10: oracle ceiling (full-info capacity upper bound)
    oracle = load_json(os.path.join(OUT, "oracle_replay_all.json")) or {}
    for i, key in enumerate(["seed42", "seed123", "seed999", "seed777"]):
        ov = oracle.get(key)
        if ov:
            ax.axhline(ov["ar_full_pct"], color=OKABE[i], ls=":", lw=1.2, alpha=0.7)
    oracle_means = [oracle[k]["ar_full_pct"] for k in
                    ["seed42", "seed123", "seed999", "seed777"]
                    if oracle.get(k)]
    if oracle_means:
        ax.axhline(float(np.mean(oracle_means)), color=OURS, ls="--", lw=1.5,
                   label="Oracle ceiling (mean)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Autonomy Rate (%)")
    ax.set_ylim(-2, 105)
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.set_title(f"(a) Synthetic, {len(curves)} seeds")
    ax = axes[1]
    for i, (d, c) in enumerate([("run4b_strands_seed42", "STRANDS Aruba-1"),
                                ("run4b_uci_seed42", "UCI V3 (Real)"),
                                ("run4b_sml2010_seed42", "SML2010 (Real)"),
                                ("run4b_steel_seed42", "Steel Ind. (Real)"),
                                ("run4b_airquality_seed42", "Air Quality")]):
        mm = load_metrics(d)
        if not mm:
            continue
        x = [m["day"] for m in mm]
        y = [m["autonomy_rate"] for m in mm]
        ax.plot(x, y, color=OKABE[i], marker="s", markevery=max(1, len(x) // 8),
                label=c, lw=1.3)
    for i, key in enumerate(["strands", "uci_v3"]):
        ov = oracle.get(key)
        if ov:
            ax.axhline(ov["ar_full_pct"], color=OKABE[i], ls=":", lw=1.2, alpha=0.7)
            ax.text(0.5, ov["ar_full_pct"] + 2, f"oracle {ov['ar_full_pct']:.0f}%",
                    fontsize=6.5, color=OKABE[i])
    ax.set_xlabel("Day")
    ax.set_ylabel("Autonomy Rate (%)")
    ax.set_ylim(-2, 105)
    ax.legend(loc="lower right")
    ax.set_title("(b) Real / activity datasets")
    save(fig, "fig_ar_learning")


# ============================================================
# 图 1b (v10): Online vs Oracle — learning speed vs capacity ceiling
# ============================================================
def fig_oracle_vs_online():
    oracle = load_json(os.path.join(OUT, "oracle_replay_all.json")) or {}
    br = load_json(os.path.join(OUT, "baseline_results_4x.json")) or {}
    labels = ["Seed 42", "Seed 123", "Seed 999", "Seed 777",
              "STRANDS", "UCI V3", "SML2010", "Steel", "Air"]
    keys = ["seed42", "seed123", "seed999", "seed777", "strands", "uci_v3",
            "sml2010", "steel", "airquality"]
    br_keys = ["synthetic_seed42_s42", "synthetic_seed123_s42",
               "synthetic_seed999_s42", "synthetic_seed777_s42",
               "strands_s42", "uci_v3_s42", "sml2010_s42", "steel_s42",
               "airquality_s42"]
    online, oracle_ar, oracle_act = [], [], []
    for k, bk in zip(keys, br_keys):
        online.append(br.get(bk, {}).get("system_autonomy_rate", 0))
        ov = oracle.get(k, {})
        oracle_ar.append(ov.get("ar_full_pct", 0))
        oracle_act.append(ov.get("ar_actionable_pct", 0))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    x = np.arange(len(labels))
    w = 0.34
    ax.bar(x - w / 2, online, w, label="Online (30-day growth)",
           color=OKABE[1])
    ax.bar(x + w / 2, oracle_ar, w, label="Oracle (full-info ceiling)",
           color=OURS, alpha=0.85)
    for i, v in enumerate(oracle_act):
        ax.text(x[i] + w / 2, oracle_ar[i] + 2, f"{v:.0f}%",
                ha="center", fontsize=6.5, color=OURS)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=20, ha="right")
    ax.set_ylabel("Autonomy Rate (%)")
    ax.set_ylim(0, 112)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Online learning vs. rule capacity ceiling")
    save(fig, "fig_oracle_vs_online")


# ============================================================
# 图 2/3: 基线对比（AR 与 Cloud Call Reduction，均值±SD）
# ============================================================
def _system_ar_per_dataset():
    br = load_json(os.path.join(OUT, "baseline_results_4x.json")) or {}
    out = {}
    for k in ["synthetic_seed42", "synthetic_seed123", "synthetic_seed999",
              "strands_aruba1", "uci_v2", "uci_v3"]:
        out[k] = br.get(k, {}).get("system_autonomy_rate", 0.0)
    return out


def fig_baselines():
    stats = load_json(os.path.join(OUT, "statistics_4x_baselines.json")) or {}
    datasets = stats.get("datasets", [])
    names = stats.get("baselines", [])
    ar_m = stats.get("ar_matrix_percent", [])
    ccr_m = stats.get("ccr_matrix_percent", [])
    if not ar_m or not names:
        return
    labels = [n.replace("B1 ", "").replace("B2 ", "").replace("B3 ", "")
               .replace("B4 ", "").replace("B5 ", "").replace("B6 ", "")
               .replace("B7 ", "").replace("B8 ", "").replace("B5b ", "")
               for n in names]
    labels[-1] = "DistillToMCU (ours)"
    fig, axes = plt.subplots(1, 2, figsize=FIG_FULL)
    for ax, matrix, ylab in [(axes[0], ar_m, "Autonomy Rate (%)"),
                             (axes[1], ccr_m, "Cloud Call Reduction (%)")]:
        means, errs = [], []
        cols = []
        for j in range(len(names)):
            vals = [row[j] for row in matrix if row]
            means.append(np.mean(vals))
            errs.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
            cols.append(OURS if j == len(names) - 1 else OKABE[j % len(OKABE)])
        ypos = np.arange(len(labels))
        ax.barh(ypos, means, xerr=errs, color=cols, height=0.62,
                edgecolor="white", linewidth=0.5, capsize=2.5)
        for y, m, e in zip(ypos, means, errs):
            ax.text(m + e + 3.5, y, f"{m:.0f}", va="center", fontsize=7)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel(ylab)
        ax.set_xlim(0, 136)
    axes[0].set_title("(a) Autonomy Rate (held-out window, 36 blocks)")
    axes[1].set_title("(b) Cloud Call Reduction (held-out window)")
    fig.text(0.5, -0.03,
             "All methods evaluated on days 22-30 (train/warm-up on days 1-21); "
             "Ours AR = live online decisions, Ours AGREE = warm-period re-distilled "
             "rules (zero leakage). See statistics_4x_baselines.json note.",
             ha="center", fontsize=6.3, color="#555555")
    fig.tight_layout()
    save(fig, "fig_baselines")


# ============================================================
# 图 4: 本地 vs 云端延迟分布（合并 6 组实验 trace）
# ============================================================
def fig_latency():
    local, cloud = [], []
    per_ds = {}
    for d, key, label in EXPERIMENTS:
        ll, cl = [], []
        for t in load_traces(d):
            lat = trace_latency(t)
            if lat is None:
                continue
            mode = trace_mode(t)
            if mode == "local":
                ll.append(float(lat))
            elif mode == "cloud":
                cl.append(float(lat))
        local += ll
        cloud += cl
        per_ds[key] = (ll, cl)
    if not local and not cloud:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIG_FULL)
    ax = axes[0]
    data = [cloud]
    bp = ax.boxplot(data, tick_labels=["Cloud LLM"], widths=0.45,
                    patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], [OKABE[2], OKABE[5]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_ylabel("Latency (ms)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 30000)
    ax.set_title("(a) Cloud round-trip latency (real traces)")
    # v10.5g: on-device measured match latency (rule_engine_match incl.
    # cJSON walk; replaces the simulated local-execution latency)
    mcu3 = load_json(os.path.join(OUT, "mcu_latency_3x_summary.json")) or {}
    mcu = load_json(os.path.join(OUT, "mcu_metrics.json")) or {}
    match_lat = mcu3.get("pooled", {}) or {}
    p50_us = match_lat.get("p50_range_us") or [None]
    if p50_us[0] is None:
        p50_us = [ (mcu.get("match_latency_us") or {}).get("p50_us") ]
    if p50_us[0]:
        ax.axhline(p50_us[0] / 1000.0, color=OURS, ls="--", lw=1.4)
        ax.text(0.97, p50_us[0] / 1000.0 * 2.5,
                f"Match p50 = {p50_us[0]/1000:.2f} ms (measured)",
                ha="right", va="bottom", fontsize=7, color=OURS,
                transform=ax.get_yaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=OURS, lw=0.7, alpha=0.9))
    ax = axes[1]
    labels, p50c, p95c = [], [], []
    for key, label in [("seed42", "Seed42"), ("seed123", "Seed123"),
                       ("seed999", "Seed999"), ("seed777", "Seed777"),
                       ("strands", "STRANDS"), ("uci_v3", "UCI V3"),
                       ("sml2010", "SML2010"), ("steel", "Steel"),
                       ("airquality", "Air")]:
        cl = per_ds.get(key, ([], []))[1]
        if not cl:
            continue
        labels.append(label)
        p50c.append(pct(cl, 50) if cl else 0)
        p95c.append(pct(cl, 95) if cl else 0)
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x, p50c, w, label="Cloud p50 (real)", color=OKABE[5], alpha=0.85)
    if p50_us[0]:
        ax.axhline(p50_us[0] / 1000.0, color=OURS, ls="--", lw=1.4,
                   label=f"Match p50 ({p50_us[0]/1000:.2f} ms, measured)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.6, 20000)
    ax.set_ylabel("p50 latency (ms, log)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=5.6)
    ax.set_title("(b) Per-dataset cloud p50 (log scale)", pad=32)
    fig.text(0.5, -0.04,
             "Provenance: cloud = real API latency (llm_response.latency_ms in released "
             "traces); match latency = measured on-device via esp_timer "
             "(mcu_latency_3x_summary.json, three sessions).",
             ha="center", fontsize=6.5, color="#666666")
    fig.tight_layout()
    save(fig, "fig_latency")


# ============================================================
# 图 5: Precision / Recall（系统在各数据集的规则执行质量）
# ============================================================
def _fig_precision_recall_teacher(tr):
    """真实 teacher-replay Precision/Recall（60 快照/数据集，教师多数决策）。"""
    short = {
        "synthetic_seed42": "42", "synthetic_seed123": "123",
        "synthetic_seed999": "999", "synthetic_seed777": "777",
        "strands_aruba1": "STRANDS", "uci_v3": "UCI",
        "sml2010": "SML2010", "steel": "Steel", "airquality": "Air",
    }
    order = [k for k in short if k in tr]
    if not order:
        return

    ours, dt = "Ours (warm rules)", "Decision Tree (batch)"
    op, orr, dp, dr = [], [], [], []
    for k in order:
        m = tr[k]["methods"]
        op.append(m[ours]["precision_pct"] or 0)
        orr.append(m[ours]["recall_pct"])
        dp.append(m[dt]["precision_pct"] or 0)
        dr.append(m[dt]["recall_pct"])

    fig, axes = plt.subplots(1, 2, figsize=FIG_FULL, gridspec_kw={"wspace": 0.34})
    x = np.arange(len(order))
    w = 0.38
    ax = axes[0]
    ax.bar(x - w / 2, op, w, label="Precision (Ours)", color=OKABE[0])
    ax.bar(x + w / 2, orr, w, label="Recall (Ours)", color=OKABE[3])
    ax.set_xticks(x)
    ax.set_xticklabels([short[k] for k in order], fontsize=7,
                       rotation=30, ha="right")
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=7)
    ax.set_title("(a) Ours — teacher-relative P/R (warm rules)")
    for xi, p, r in zip(x, op, orr):
        ax.text(xi - w / 2, p + 1.5, f"{p:.0f}", ha="center", fontsize=6.5)
        ax.text(xi + w / 2, r + 1.5, f"{r:.0f}", ha="center", fontsize=6.5)

    ax = axes[1]
    names, pm, rm = [], [], []
    for name in ["Ours (warm rules)", "Decision Tree (batch)",
                 "Decision Tree (online refit)", "ESP-Claw-style",
                 "LLM One-shot", "User-defined Rules", "Exact Cache"]:
        ps = [tr[k]["methods"][name]["precision_pct"] for k in order]
        rs = [tr[k]["methods"][name]["recall_pct"] for k in order]
        ps = [p for p in ps if p is not None]
        rs = [r for r in rs if r is not None]
        if name == "Decision Tree (batch)":
            names.append("DT (batch)")
        elif name == "Decision Tree (online refit)":
            names.append("DT (online)")
        else:
            names.append(name)
        pm.append(round(float(np.mean(ps)), 1))
        rm.append(round(float(np.mean(rs)), 1))
    xx = np.arange(len(names))
    ax.bar(xx - w / 2, pm, w, label="Precision (mean)", color=OKABE[0])
    ax.bar(xx + w / 2, rm, w, label="Recall (mean)", color=OKABE[3])
    ax.set_xticks(xx)
    ax.set_xticklabels(names, fontsize=6.5, rotation=18, ha="right")
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=7)
    ax.set_title("(b) Method comparison (mean over 9 datasets)")
    for xi, p, r in zip(xx, pm, rm):
        ax.text(xi - w / 2, p + 1.5, f"{p:.0f}", ha="center", fontsize=6)
        ax.text(xi + w / 2, r + 1.5, f"{r:.0f}", ha="center", fontsize=6)

    fig.text(0.5, -0.04,
             "Provenance: 60 sensor snapshots per dataset; ground truth = majority of "
             "3 teacher (DeepSeek T=0) repeats; precision = P(method action == teacher | "
             "method acted); recall = P(method acted and matched | teacher acted). "
             "Replaces the Phase-0 placeholder (no user feedback).",
             ha="center", fontsize=6.5, color="#555555")
    save(fig, "fig_precision_recall")


def fig_precision_recall():
    tr = load_json(os.path.join(OUT, "teacher_replay_results.json")) or {}
    if tr:
        _fig_precision_recall_teacher(tr)
        return
    labels, prec, rec = [], [], []
    for d, key, label in EXPERIMENTS:
        traces = load_traces(d)
        if not traces:
            continue
        local = [t for t in traces if trace_mode(t) == "local"]
        accepted = [t for t in local if (t.get("feedback") or {}).get("type") == "accepted"]
        cloud_with_action = [t for t in traces
                             if trace_mode(t) == "cloud" and trace_has_action(t)]
        total_action = len(local) + len(cloud_with_action)
        labels.append(label)
        prec.append(len(accepted) / max(1, len(local)) * 100)
        rec.append(len(local) / max(1, total_action) * 100)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    labels_short = [l.replace(" (Syn)", "").replace(" (Speech)", "")
                    .replace(" (Real)", "").replace("STRANDS Aruba-1", "STRANDS")
                    for l in labels]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, prec, w, label="Precision", color=OKABE[0])
    ax.bar(x + w / 2, rec, w, label="Recall", color=OKABE[3])
    ymax = max(max(prec), max(rec)) * 1.18
    ax.set_ylim(0, ymax)
    for xi, p, r in zip(x, prec, rec):
        ax.text(xi - w / 2, p + ymax * 0.02, f"{p:.0f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, r + ymax * 0.02, f"{r:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=7.5)
    ax.set_ylabel("Percent (%)")
    ax.legend()
    ax.set_title("Precision / Recall — Phase 0 placeholder")
    fig.text(0.5, -0.04,
             "[!] NOT yet reportable: override feedback path is wired in firmware "
             "(v10.5f: 5s veto window after local execution, GPIO14 = corrected) but "
             "no user veto data has been collected yet; report after an interactive "
             "user study.",
             ha="center", fontsize=6.5, color="#B00020")
    save(fig, "fig_precision_recall")


# ============================================================
# 图 6: 规则库规模随天增长
# ============================================================
def fig_rules_size():
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    plotted = False
    for i, (d, key, label) in enumerate(EXPERIMENTS):
        mm = load_metrics(d)
        if not mm:
            continue
        x = [m["day"] for m in mm]
        y = [m.get("total_rules", 0) for m in mm]
        ax.plot(x, y, color=OKABE[i % len(OKABE)], marker="o",
                markevery=max(1, len(x) // 8), label=label, lw=1.2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Day")
    ax.set_ylabel("Total distilled rules")
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    save(fig, "fig_rules_size")


# ============================================================
# 图 7: Friedman + Nemenyi 临界差 (CD) 图
# ============================================================

# ============================================================
# ? 1x (v10.5): 4x ?????????n=4 ?? ? Bootstrap CI?
# ============================================================
def fig_4x_ar_learning():
    d4 = load_json(os.path.join(FIG, "data_4x.json")) or {}
    oracle = load_json(os.path.join(OUT, "oracle_replay_all.json")) or {}
    synth = ["seed42", "seed123", "seed999", "seed777"]
    real = ["strands", "uci_v3", "sml2010", "steel", "airquality"]
    real_labels = {"strands": "STRANDS", "uci_v3": "UCI V3",
                   "sml2010": "SML2010", "steel": "Steel Ind.",
                   "airquality": "Air Quality"}
    fig, axes = plt.subplots(1, 2, figsize=FIG_FULL)
    ax = axes[0]
    for i, k in enumerate(synth):
        g = d4.get(k, {}).get("growth")
        if not g:
            continue
        x = np.arange(1, len(g["mean"]) + 1)
        ax.plot(x, g["mean"], color=OKABE[i], lw=1.4,
                label=f"Synthetic {k[4:]} (n={d4.get(k, {}).get('n_runs', 4)})")
        ax.fill_between(x, g["ci_lo"], g["ci_hi"], color=OKABE[i], alpha=0.12)
    if oracle:
        ov = [oracle[k]["ar_full_pct"] for k in synth if oracle.get(k)]
        if ov:
            ax.axhline(float(np.mean(ov)), color=OURS, ls="--", lw=1.5,
                       label="Oracle ceiling (mean)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Autonomy Rate (%)")
    ax.set_ylim(-2, 108)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("(a) Synthetic 4x repeats")
    ax = axes[1]
    for i, k in enumerate(real):
        g = d4.get(k, {}).get("growth")
        if not g:
            continue
        x = np.arange(1, len(g["mean"]) + 1)
        ax.plot(x, g["mean"], color=OKABE[i], lw=1.5,
                label=real_labels[k])
        ax.fill_between(x, g["ci_lo"], g["ci_hi"], color=OKABE[i], alpha=0.12)
    for i, k in enumerate(real):
        ov = oracle.get(k)
        if ov:
            ax.axhline(ov["ar_full_pct"], color=OKABE[i], ls=":", lw=1.2, alpha=0.7)
    ax.set_xlabel("Day")
    ax.set_ylim(-2, 108)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#888888", ls=":", lw=1.2,
                          label="Oracle ceiling"))
    ax.legend(handles=handles, fontsize=7)
    ax.set_title("(b) Real datasets (30-day full horizon)")
    fig.text(0.5, -0.04,
             "Full-horizon curves cover days 1-30 including warm-up; held-out "
             "window (days 22-30) means are in Table 1 and differ (e.g., UCI "
             "48.2% held-out vs 32.4% full-horizon).",
             ha="center", fontsize=6.5, color="#666666")
    save(fig, "fig_ar_learning")


# ============================================================
# ? 8 (v10.5): Cross-LLM ??????DeepSeek vs Qwen?
# ============================================================
def fig_cross_llm():
    fig = plt.figure(figsize=(6.9, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.82], hspace=0.45, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])
    w = 0.34
    tr = load_json(os.path.join(OUT, "teacher_replay_results.json")) or {}

    # (a) model-model agreement（教师不同模型对同一 prompt 的一致性）
    mm_overall, mm_dev, mm_labels = [], [], []
    for k in ["synthetic_seed42", "synthetic_seed123", "synthetic_seed999",
              "synthetic_seed777", "strands_aruba1", "uci_v3"]:
        d = tr.get(k)
        if not d:
            continue
        mm_labels.append({"synthetic_seed42": "42",
                          "synthetic_seed123": "123",
                          "synthetic_seed999": "999",
                          "synthetic_seed777": "777",
                          "strands_aruba1": "STRANDS",
                          "uci_v3": "UCI"}[k])
        mm_overall.append(d.get("model_model_agreement_pct") or 0)
        mm_dev.append(d.get("model_model_agreement_device_pct") or 0)
    if mm_labels:
        x = np.arange(len(mm_labels))
        ax_a.bar(x - w / 2, mm_overall, w, label="Overall (device+cmd)",
                 color=OKABE[2])
        ax_a.bar(x + w / 2, mm_dev, w, label="Device-level", color=OKABE[5])
        for xi, o, a in zip(x, mm_overall, mm_dev):
            ax_a.text(xi - w / 2, o + 1.5, f"{o:.0f}", ha="center", fontsize=6.8)
            ax_a.text(xi + w / 2, a + 1.5, f"{a:.0f}", ha="center", fontsize=6.8)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(mm_labels, fontsize=8)
        ax_a.set_ylabel("Agreement (%)")
        ax_a.set_ylim(0, 130)
        ax_a.legend(loc="upper center", ncol=2, fontsize=6.6, columnspacing=1.0,
                    handletextpad=0.6)
    ax_a.set_title("(a) Model-model agreement", fontsize=9)

    # (c) 蒸馏规则保真度 / 迁移
    rt = load_json(os.path.join(OUT, "rule_transfer.json")) or {}
    fid, trans, mm, rt_labels = [], [], [], []
    for k in ["synthetic_seed42", "synthetic_seed123", "synthetic_seed999",
              "synthetic_seed777", "strands_aruba1", "uci_v3"]:
        d = rt.get(k)
        if not d:
            continue
        rt_labels.append({"synthetic_seed42": "42",
                          "synthetic_seed123": "123",
                          "synthetic_seed999": "999",
                          "synthetic_seed777": "777",
                          "strands_aruba1": "STRANDS",
                          "uci_v3": "UCI"}[k])
        fid.append(d.get("fidelity_to_deepseek_pct") or 0)
        trans.append(d.get("transfer_to_qwen_pct") or 0)
        mm.append(d.get("model_model_agreement_pct") or 0)
    if rt_labels:
        x3 = np.arange(len(rt_labels))
        ww = 0.25
        ax_c.bar(x3 - ww, fid, ww, label="Fidelity (DS)", color=OKABE[0])
        ax_c.bar(x3, trans, ww, label="Transfer (Qwen)", color=OKABE[1])
        ax_c.bar(x3 + ww, mm, ww, label="Model-model", color=OKABE[2])
        for xi, a, b, c in zip(x3, fid, trans, mm):
            ax_c.text(xi - ww, a + 1.5, f"{a:.0f}", ha="center", fontsize=6.2)
            ax_c.text(xi, b + 1.5, f"{b:.0f}", ha="center", fontsize=6.2)
            ax_c.text(xi + ww, c + 1.5, f"{c:.0f}", ha="center", fontsize=6.2)
        ax_c.set_xticks(x3)
        ax_c.set_xticklabels(rt_labels, fontsize=8)
        ax_c.set_ylabel("Agreement (%)")
        ax_c.set_ylim(0, 130)
        ax_c.legend(loc="upper center", ncol=3, fontsize=6.2, columnspacing=1.0,
                    handletextpad=0.6)
    ax_c.set_title("(c) Distilled-rule fidelity / transfer", fontsize=9)

    # (b) 同一数据换教师后的在线 AR
    cm = load_json(os.path.join(OUT, "cross_model_same_data.json")) or {}
    ar_ds, ar_qw, cm_labels = [], [], []
    for k in ["seed42", "seed777", "uci_v3"]:
        d = cm.get(k)
        if not d:
            continue
        cm_labels.append({"seed42": "42", "seed777": "777", "uci_v3": "UCI"}[k])
        ar_ds.append(d["deepseek_ar"] or 0)
        ar_qw.append(d["qwen_ar"] or 0)
    if cm_labels:
        x2 = np.arange(len(cm_labels))
        ax_b.bar(x2 - w / 2, ar_ds, w, label="DeepSeek", color=OKABE[2])
        ax_b.bar(x2 + w / 2, ar_qw, w, label="Qwen 3.7", color=OKABE[5])
        for xi, a, b in zip(x2, ar_ds, ar_qw):
            ax_b.text(xi - w / 2, a + 1.5, f"{a:.0f}", ha="center", fontsize=8)
            ax_b.text(xi + w / 2, b + 1.5, f"{b:.0f}", ha="center", fontsize=8)
        ax_b.set_xticks(x2)
        ax_b.set_xticklabels(cm_labels, fontsize=9)
        ax_b.set_ylabel("AR (%)")
        ax_b.set_ylim(0, 115)
        ax_b.legend(loc="upper left", ncol=2, fontsize=8)
    ax_b.set_title("(b) Same-data online AR (teacher swap)", fontsize=9)

    fig.text(0.5, 0.005,
             "(a) same-prompt DeepSeek-majority vs Qwen on 60 snapshots/dataset; "
             "(b) online runs on identical sensor sequences (v10.6 generator fix); "
             "(c) same teacher-replay snapshots, original run prompt, engine "
             "resolution (device+cmd; device-level in rule_transfer.json).",
             ha="center", fontsize=6.3, color="#555555")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.09)
    save(fig, "fig_cross_llm")


def fig_nemenyi_cd():
    stats = load_json(os.path.join(OUT, "statistics_4x_baselines.json")) or {}
    datasets = stats.get("datasets", [])
    names = stats.get("baselines", [])
    ar_m = stats.get("ar_matrix_percent", [])
    if not datasets or not ar_m or not names:
        return
    # v10.7: AR 矩阵（36 blocks × 8 方法）已包含 Ours 最后一列，直接检验。
    matrix = [list(row) for row in ar_m if row]
    labels = [n.replace("B1 ", "").replace("B2 ", "").replace("B3 ", "")
               .replace("B4 ", "").replace("B5 ", "").replace("B6 ", "")
               .replace("B7 ", "").replace("B8 ", "").replace("B5b ", "")
               for n in names]
    labels[-1] = "DistillToMCU"
    fried = st.friedman_test(matrix)
    nem = st.nemenyi_posthoc(fried["mean_ranks"], len(matrix), labels)
    cd = nem["critical_difference"]
    pairs = nem["pairs"]
    avg = fried["mean_ranks"]
    order2 = np.argsort(avg)
    fig, ax = plt.subplots(figsize=FIG_FULL)
    y = np.arange(len(labels))
    for i in order2:
        c = OURS if i == len(labels) - 1 else OKABE[i % len(OKABE)]
        ax.plot([avg[i], avg[i]], [y[i] - 0.20, y[i] + 0.20], color=c, lw=2)
        ax.plot(avg[i], y[i], "o", color=c, ms=6)
        ax.text(avg[i] - 0.18, y[i], labels[i], ha="right", va="center",
                fontsize=7.5, color=c,
                fontweight="bold" if i == len(labels) - 1 else "normal")
    # 非显著配对连接线（Nemenyi 标准画法）
    layer = {}
    for p in pairs:
        if p.get("significant"):
            continue
        a, b = p["pair"].split(" vs ")
        ia, ib = labels.index(a), labels.index(b)
        yline = min(y[ia], y[ib]) + 0.30
        layer[yline] = layer.get(yline, 0) + 1
        yline += ((layer[yline] - 1) % 3) * 0.13
        ax.plot([avg[ia], avg[ib]], [yline, yline], color="#444444", lw=2.0, alpha=0.8)
    # CD 标尺（顶部）
    x0 = min(avg) - 0.55
    ax.plot([x0, x0 + cd], [len(labels) + 0.35] * 2, color="black", lw=1.4)
    ax.text(x0 + cd / 2, len(labels) + 0.58, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.plot([x0, x0], [len(labels) + 0.25, len(labels) + 0.45], color="black", lw=1.2)
    ax.plot([x0 + cd, x0 + cd], [len(labels) + 0.25, len(labels) + 0.45], color="black", lw=1.2)
    ax.set_xlim(x0 - 1.3, max(avg) + 0.6)
    ax.set_ylim(-0.9, len(labels) + 0.85)
    ax.set_xlabel("Average rank (lower = better)")
    ax.set_yticks([])
    ax.grid(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(False)
    pval = fried["p_value"]
    ptxt = f"p = {pval:.1e}" if pval > 0 else "p < 1e-6"
    ax.set_title(f"Friedman {ptxt} "
                 f"({len(matrix)} blocks × {len(labels)} methods)")
    fig.text(0.5, -0.04,
             "Friedman + Nemenyi post-hoc (α=0.05); DistillToMCU included in the test. "
             "Bars connect methods without significant rank difference.",
             ha="center", fontsize=6.5, color="#666666")
    save(fig, "fig_nemenyi_cd")


# ============================================================
# 图 8: 消融
# ============================================================
def fig_ablations():
    ab = (load_json(os.path.join(OUT, "ablation_results_4x.json")) or
          load_json(os.path.join(OUT, "seed42", "ablation_results.json")) or {})
    if not ab:
        return
    fig, axes = plt.subplots(2, 2, figsize=FIG_FULL)

    def bars(ax, labels, vals, title, highlight=None, ylab="AR (%)"):
        cols = [OURS if (highlight is not None and i in highlight) else OKABE[i % len(OKABE)]
                for i in range(len(vals))]
        ax.bar(range(len(vals)), vals, color=cols, edgecolor="white", linewidth=0.5)
        for xi, v in zip(range(len(vals)), vals):
            ax.text(xi, v + max(vals) * 0.02, f"{v:.1f}", ha="center", fontsize=6.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=6.2)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(title, fontsize=8.5)

    # (a) 蒸馏来源
    d = ab.get("1_distillation_sources", {})
    order = ["L1 only", "L2 only", "L3 only", "Full (L1+L2+L3)"]
    keys = [k for k in order if k in d]
    vals = [d[k]["ar"] for k in keys]
    bars(axes[0][0], [k.replace(" (L1+L2+L3)", "") for k in keys], vals,
         "(a) Distillation sources", highlight=[len(vals) - 1])
    # (b) 泛化（含 held-out）
    g = ab.get("2_rule_generalization", {})
    gl = ["In-sample exact", "In-sample inclusive", "Held-out exact", "Held-out inclusive"]
    gv = [g.get("exact_match", {}).get("ar_in_sample", 0),
          g.get("inclusive_interval", {}).get("ar_in_sample", 0),
          g.get("heldout_exact_match_ar", 0),
          g.get("heldout_inclusive_interval_ar", 0)]
    bars(axes[0][1], gl, gv, "(b) Rule generalization", highlight=[1, 3])
    # (c) 生命周期
    lc = ab.get("3_rule_lifecycle", {})
    lo = ["None (permanent)", "Simple TTL (7d)", "Time-decay only", "Full lifecycle (ours)"]
    lk = [k for k in lo if k in lc]
    lv = [lc[k]["ar"] for k in lk]
    bars(axes[1][0], [k.replace(" (permanent)", "").replace(" (7d)", "") for k in lk], lv,
         "(c) Rule lifecycle", highlight=[len(lk) - 1])
    # (d) PMS bandit: stationary vs nonstationary regret（lower = better）
    bd = ab.get("4_pms_vs_alternatives", {})
    if bd:
        ax = axes[1][1]
        method_order = ["PMS (ours)", "Exact TS", "ε-Greedy(0.1)", "Greedy"]
        method_short = {"PMS (ours)": "PMS", "Exact TS": "Exact TS",
                        "ε-Greedy(0.1)": "ε-Greedy", "Greedy": "Greedy"}
        groups = [("stationary", "Stationary"), ("nonstationary", "Non-stationary")]
        x = np.arange(len(groups))
        w = 0.18
        colors = [OURS, OKABE[1], OKABE[3], OKABE[2]]
        all_vals = []
        for mi, mkey in enumerate(method_order):
            vals = [bd.get(g, {}).get(mkey, {}).get("regret", 0) for g, _ in groups]
            all_vals += vals
            off = (mi - (len(method_order) - 1) / 2) * w
            ax.bar(x + off, vals, w, label=method_short[mkey],
                   color=colors[mi], edgecolor="white", linewidth=0.4)
            for xi, v in zip(x, vals):
                ax.text(xi + off, v + 14, f"{v:.0f}", ha="center", fontsize=5.4)
        ax.set_xticks(x)
        ax.set_xticklabels([g[1] for g in groups], fontsize=6.5)
        ax.set_ylabel("Cumulative regret", fontsize=8)
        ax.set_ylim(0, max(all_vals) * 1.12)
        ax.legend(fontsize=5.5, ncol=2, loc="upper left")
        ax.set_title("(d) PMS vs alternatives (regret, lower = better)", fontsize=8.5)
    fig.text(0.5, -0.04,
             "Provenance: (a-c) run4b_seed42_seed42, held-out protocol (train first 70% "
             "of days, evaluate last 30%); (d) synthetic bandit simulation "
             "(bandit_selector.py) — CPU cycles are design estimates, not MCU measurements.",
             ha="center", fontsize=6.5, color="#666666")
    fig.tight_layout()
    save(fig, "fig_ablations")


# ============================================================
# 图 9: 系统架构图（论文 Figure 1 用）
# ============================================================
def fig_architecture():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=FIG_FULL)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#E8EDF2", ec="#4A90D9", fs=8):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           fc=fc, ec=ec, lw=1.2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, linespacing=1.35)

    def arrow(x1, y1, x2, y2, dashed=False, label=None, lab_dx=0.0, lab_dy=0.22,
              ha="center"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.4,
                                    linestyle=(0, (4, 2)) if dashed else "solid"))
        if label:
            ax.text((x1 + x2) / 2 + lab_dx, (y1 + y2) / 2 + lab_dy, label,
                    ha=ha, fontsize=6.8, color="#555555")

    box(0.3, 5.0, 2.5, 1.3, "Sensor data sources\nUCI / CASAS /\nsynthetic / upload")
    box(3.9, 5.0, 2.8, 1.3, "PC Distiller\nLLM behavior traces →\nthreshold-interval rules",
        fc="#FFF3E0", ec="#D97706")
    box(7.5, 5.0, 2.6, 1.3, "ESP32-S3 MCU\nrule match → GPIO/LED\n(local execution)",
        fc="#E8F2EE", ec="#059669")
    box(7.5, 1.0, 2.6, 1.3, "Cloud LLM\n(DeepSeek)\ntool-call fallback", fc="#E8F2EE", ec="#059669")
    box(0.3, 1.0, 2.5, 1.3, "Trace store\nJSONL + metrics", fc="#E8EDF2", ec="#4A90D9")
    arrow(2.92, 5.65, 3.78, 5.65, label="rows", lab_dy=0.24)
    arrow(6.82, 5.65, 7.38, 5.65, label="rules", lab_dy=0.24)
    arrow(8.8, 4.88, 8.8, 2.52, label="no match", lab_dx=0.14, ha="left")
    # trace feedback：从 Cloud 框底边先向下引出，再左折入 Trace store（不贴框边）
    ax.plot([8.3, 8.3], [0.90, 0.58], color="#444444", lw=1.4,
            linestyle=(0, (4, 2)))
    arrow(8.3, 0.58, 2.82, 0.58, dashed=True, label="trace feedback", lab_dy=-0.26)
    arrow(1.55, 2.42, 1.55, 4.88, dashed=True, label="re-distill", lab_dx=-0.14,
          ha="right")
    ax.set_title("DistillToMCU: from cloud-LLM behavior to MCU-independent rules")
    fig.text(0.5, -0.04,
             "Solid = online data/control flow; dashed = offline distillation feedback loop.",
             ha="center", fontsize=6.5, color="#666666")
    save(fig, "fig_architecture")


# ============================================================
# 图 10: 规则生命周期状态机
# ============================================================
def fig_rule_lifecycle():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=FIG_FULL)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    states = ["candidate", "verified", "active", "degraded", "retired"]
    xs = [0.4, 3.0, 5.6, 8.2, 10.8]
    bw, bh, by = 1.9, 0.95, 1.8
    for x, s in zip(xs, states):
        fc = "#E8F2EE" if s == "active" else ("#F5F0E8" if s == "retired" else "#E8EDF2")
        p = FancyBboxPatch((x, by), bw, bh, boxstyle="round,pad=0.06",
                           fc=fc, ec="#4A90D9", lw=1.2)
        ax.add_patch(p)
        ax.text(x + bw / 2, by + bh / 2, s, ha="center", va="center", fontsize=9)

    def ar(x1, x2, label):
        ax.annotate("", xy=(x2, by + bh / 2), xytext=(x1 + bw, by + bh / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.3))
        ax.text((x1 + bw + x2) / 2, by + bh + 0.30, label, ha="center",
                va="bottom", fontsize=6.8, color="#333333", linespacing=1.25)

    ar(0.4, 3.0, "evidence ≥3\nconf ≥0.70")
    ar(3.0, 5.6, "conf ≥0.85\nno negative")
    ar(5.6, 8.2, "negative\nfeedback")
    ar(8.2, 10.8, "degraded\n≥14 days")
    # 反馈边：active → degraded（红色虚线，位于下方）
    ax.annotate("", xy=(8.32, by - 0.42), xytext=(6.6, by - 0.42),
                arrowprops=dict(arrowstyle="-|>", color="#CC4444", lw=1.2,
                                linestyle=(0, (3, 2))))
    ax.text(7.45, by - 0.68, "freshness decay / correction", ha="center",
            va="top", fontsize=6.3, color="#CC4444")
    ax.set_title("Rule lifecycle (confidence + freshness driven)")
    fig.text(0.5, -0.04,
             "Thresholds: candidate→verified: Wilson 95% CI lower bound ≥0.70 with ≥3 evidence; "
             "verified→active: ≥0.85; degraded→retired after 14 days.",
             ha="center", fontsize=6.5, color="#666666")
    save(fig, "fig_rule_lifecycle")


# ============================================================
# 图 9: 对话实验统计（存在时）
# ============================================================
def fig_chat_stats():
    chats = load_json(os.path.join(OUT, "chat_experiments.json")) or []
    chats = [c for c in chats if c.get("id") != "selftest_chat"]
    if not chats:
        print("[fig] no chat experiments, skip fig_chat_stats")
        return
    titles = [(c.get("title") or "?")[:12] for c in chats[:8]]
    stats = [c.get("stats") or {} for c in chats[:8]]
    msgs = [s.get("messages", 0) for s in stats]
    acts = [s.get("action_count", 0) for s in stats]
    ars = [s.get("autonomy_rate", 0) for s in stats]
    fig, ax = plt.subplots(figsize=FIG_FULL)
    x = np.arange(len(titles))
    w = 0.28
    ax.bar(x - w, msgs, w, label="Messages", color=OKABE[1])
    ax.bar(x, acts, w, label="Actions", color=OKABE[5])
    ax.bar(x + w, ars, w, label="Autonomy rate (%)", color=OKABE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(titles, rotation=18, fontsize=7)
    ax.legend(ncol=3, fontsize=7.5)
    ax.set_ylabel("Count / %")
    save(fig, "fig_chat_stats")


def main():
    os.makedirs(FIG, exist_ok=True)
    funcs = [fig_architecture, fig_rule_lifecycle, fig_4x_ar_learning, fig_baselines,
             fig_latency, fig_precision_recall, fig_rules_size, fig_nemenyi_cd,
             fig_ablations, fig_oracle_vs_online, fig_cross_llm, fig_chat_stats]
    for fn in funcs:
        try:
            fn()
        except Exception as e:
            print(f"[fig] {fn.__name__} FAILED: {e}")
    print("done.")


if __name__ == "__main__":
    main()
