"""
DistillToMCU — SML2010 数据转换器
=================================
把 UCI 274 SML2010（domotic house 室内气候）转成与 UCI Occupancy 同构的
传感器快照格式：temperature / humidity / light / co2 / motion(None)，
并派生 hour / time_of_day / temp_trend / light_category。

输入:
  data/sml2010/NEW-DATA-1.T15.txt  (训练段)
  data/sml2010/NEW-DATA-2.T15.txt  (测试段)

输出:
  data/sml2010/snapshots.json  — [{temperature, humidity, light, co2,
                                   motion, hour, time_of_day, temp_trend,
                                   light_category}, ...]

说明: SML2010 没有运动/占用传感器，motion 固定为 None（诚实空缺，不合成）。
取 Habitacion（卧室）一路传感器作为主空间读数。
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "sml2010")
OUT = os.path.join(DATA, "snapshots.json")


def parse_rows():
    rows = []
    for fn in ["NEW-DATA-1.T15.txt", "NEW-DATA-2.T15.txt"]:
        with open(os.path.join(DATA, fn), encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # header
                parts = line.split()
                if len(parts) < 24:
                    continue
                rows.append(parts)
    return rows


def _to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _time_of_day(hour):
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _light_category(light):
    if light is None:
        return None
    if light < 25:
        return "dark"
    if light <= 60:
        return "normal"
    return "bright"


def build_snapshots():
    rows = parse_rows()
    snaps = []
    prev_temp = None
    for r in rows:
        time_s = r[1]
        try:
            hour = int(time_s.split(":")[0])
        except (ValueError, IndexError):
            hour = 12
        temp = _to_float(r[3])       # Temperature_Habitacion_Sensor
        co2 = _to_float(r[6])        # CO2_Habitacion_Sensor
        hum = _to_float(r[8])        # Humedad_Habitacion_Sensor
        light = _to_float(r[10])     # Lighting_Habitacion_Sensor
        trend = round(temp - prev_temp, 3) if (temp is not None
                                               and prev_temp is not None) else 0.0
        prev_temp = temp
        snaps.append({
            "temperature": temp,
            "humidity": hum,
            "light": light,
            "co2": co2,
            "motion": None,
            "hour": hour,
            "time_of_day": _time_of_day(hour),
            "temp_trend": trend,
            "light_category": _light_category(light),
        })
    return snaps


def stratified_sample(snaps, n=600):
    """按时间轴均匀采样 n 条，覆盖完整时间跨度。"""
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
    snaps = build_snapshots()
    snaps = stratified_sample(snaps, 600)
    os.makedirs(DATA, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(snaps, f, ensure_ascii=False)
    n_temp = sum(1 for s in snaps if s["temperature"] is not None)
    n_light = sum(1 for s in snaps if s["light"] is not None)
    print(f"[sml2010] wrote {OUT}: {len(snaps)} snapshots "
          f"(temp {n_temp}, light {n_light}, motion=0 by design)")


if __name__ == "__main__":
    main()
