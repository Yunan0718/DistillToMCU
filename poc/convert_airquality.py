"""
DistillToMCU — UCI 空气质量数据转换器 (UCI 360)
================================================
把意大利城市空气质量（CO/NOx/NO2/温湿度）转成统一快照格式。
场景：空气质量/通风控制（污染物高→通风、NO2 高→净化、温度高→开窗）。

输出: data/airquality/snapshots.json — 600 条均匀采样
"""

import csv
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "airquality", "AirQualityUCI.csv")
OUT = os.path.join(ROOT, "data", "airquality", "snapshots.json")


def _time_of_day(hour):
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _f(s):
    s = (s or "").strip().replace(",", ".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def build_snapshots():
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append(r)

    snaps = []
    prev_temp = None
    for r in rows:
        time_s = (r.get("Time") or "").split(".")[0]
        try:
            hour = int(time_s.split(":")[0])
        except (ValueError, IndexError):
            hour = 12
        temp = _f(r.get("T"))
        # 缺失值 -200 视为 None
        if temp is not None and temp <= -100:
            temp = None
        trend = round(temp - prev_temp, 2) if (temp is not None
                                               and prev_temp is not None) else 0.0
        prev_temp = temp

        def clean(k):
            v = _f(r.get(k))
            if v is not None and v <= -100:
                return None
            return v

        snaps.append({
            "co": clean("CO(GT)"),
            "nox": clean("NOx(GT)"),
            "no2": clean("NO2(GT)"),
            "temperature": temp,
            "humidity": clean("RH"),
            "hour": hour,
            "time_of_day": _time_of_day(hour),
            "temp_trend": trend,
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
    n_co = sum(1 for s in snaps if s["co"] is not None)
    print(f"[airquality] wrote {OUT}: {len(snaps)} snapshots (co {n_co})")


if __name__ == "__main__":
    main()
