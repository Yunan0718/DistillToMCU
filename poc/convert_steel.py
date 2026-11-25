"""
DistillToMCU — 钢铁工业能耗数据转换器 (UCI 851)
================================================
把钢铁工业能耗（功率/能耗/CO2/功率因数）转成统一快照格式。
场景：工业能源控制（能耗高→节能、功率因数低→补偿、CO2 高→减排）。

输出: data/steel/snapshots.json — 600 条均匀采样
"""

import csv
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "steel", "Steel_industry_data.csv")
OUT = os.path.join(ROOT, "data", "steel", "snapshots.json")

LOAD_LEVEL = {"Light_Load": 1, "Medium_Load": 2, "Maximum_Load": 3}


def _time_of_day(hour):
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _f(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def build_snapshots():
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    snaps = []
    prev_usage = None
    for r in rows:
        date_s = r.get("date", "")
        try:
            hour = int(date_s.split(" ")[1].split(":")[0])
        except (ValueError, IndexError):
            hour = 12
        usage = _f(r.get("Usage_kWh"))
        trend = round(usage - prev_usage, 2) if (usage is not None
                                                 and prev_usage is not None) else 0.0
        prev_usage = usage
        snaps.append({
            "usage_kwh": usage,
            "lagging_power": _f(r.get("Lagging_Current_Reactive.Power_kVarh")),
            "power_factor": _f(r.get("Lagging_Current_Power_Factor")),
            "co2": _f(r.get("CO2(tCO2)")),
            "load_level": LOAD_LEVEL.get(r.get("Load_Type"), 1),
            "hour": hour,
            "time_of_day": _time_of_day(hour),
            "usage_trend": trend,
        })
    return snaps


def stratified_sample(snaps, n=600):
    if len(snaps) <= n:
        return snaps
    step = len(snaps) / n
    out = []
    for i in range(n):
        lo = int(i * step)
        hi = max(lo + 1, int((i + 1) * step))
        out.append(snaps[(lo + hi) // 2])
    return out


def main():
    snaps = stratified_sample(build_snapshots(), 600)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(snaps, f, ensure_ascii=False)
    print(f"[steel] wrote {OUT}: {len(snaps)} snapshots")


if __name__ == "__main__":
    main()
