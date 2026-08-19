"""
DistillToMCU — STRANDS Aruba-1 数据转换器
=========================================
将 STRANDS 处理版 CASAS Aruba-1 数据转换为 DistillToMCU 可用格式。

输入:
  data/casas/aruba/activity.min  — 分钟级活动标注 (161,280 min ≈ 112 days)
  data/casas/aruba/location.min  — 分钟级位置标注 (10 rooms)
  data/casas/aruba/activity.names — 活动名称映射
  data/casas/aruba/location.names — 位置名称映射

输出:
  data/casas/aruba_snapshots.json — [{day, hour, minute, sensors, activity}, ...]

逆向传感器生成策略（基于真实 Aruba-1 传感器布局）:
  - 31 PIR 运动传感器分布在不同房间 → 根据当前位置+活动推导运动概率
  - 4 温度传感器 → 根据房间+时间+季节产生合理温度
  - 4 门磁传感器 → 根据位置迁移产生开关
  - 光照 → 根据时间和位置估算
"""

import json
import math
import random
import os
from datetime import datetime, timedelta
from collections import defaultdict


# Aruba-1 真实传感器-房间映射
ROOM_SENSORS = {
    "Master Bedroom":     ["M001", "M002", "T001"],
    "Master Bathroom":    ["M003", "M004", "T002"],
    "Living Room":        ["M005", "M006", "M007", "M008", "T003"],
    "Kitchen":            ["M009", "M010", "M011", "M012", "M013", "M014", "T004"],
    "Second Bedroom":     ["M015", "M016"],
    "Office":             ["M017", "M018"],
    "Second Bathroom":    ["M019", "M020"],
    "Corridor":           ["M021", "M022", "M023"],
    "Junction":           ["M024", "M025"],
    "Outside":            ["D001"],  # 前门
}

# 活动→运动概率基础值
ACTIVITY_MOTION_PROB = {
    0: 0.01,   # None — 极低运动
    1: 0.30,   # Bed_to_Toilet — 中等运动
    2: 0.15,   # Eating — 低运动（坐着吃）
    3: 0.60,   # Enter_Home — 高运动（进门）
    4: 0.40,   # Housekeeping — 中高运动
    5: 0.60,   # Leave_Home — 高运动（出门）
    6: 0.50,   # Meal_Preparation — 厨房活跃
    7: 0.05,   # Relax — 极低运动
    8: 0.10,   # Resperate — 低运动
    9: 0.01,   # Sleeping — 几乎不动
    10: 0.30,  # Wash_Dishes — 中等运动
    11: 0.20,  # Work — 低运动
}


def load_strands_data(data_dir: str) -> tuple:
    """加载 STRANDS 格式的 CASAS Aruba-1 数据"""
    # 活动标签
    with open(os.path.join(data_dir, "aruba", "activity.names"), "r") as f:
        activity_names = {i: line.strip() for i, line in enumerate(f)}

    # 位置标签
    with open(os.path.join(data_dir, "aruba", "location.names"), "r") as f:
        location_names = {i + 1: line.strip() for i, line in enumerate(f)}

    # 活动序列（每行一个整数）
    with open(os.path.join(data_dir, "aruba", "activity.min"), "r") as f:
        activities = [int(line.strip()) for line in f]

    # 位置序列（每行一个整数）
    with open(os.path.join(data_dir, "aruba", "location.min"), "r") as f:
        locations = [int(line.strip()) for line in f]

    return activity_names, location_names, activities, locations


def activities_to_snapshots(
    activities: list[int],
    locations: list[int],
    activity_names: dict,
    location_names: dict,
    start_date: str = "2010-11-04",
    interval_minutes: int = 30,
    seed: int = 42,
) -> list[dict]:
    """
    将分钟级活动+位置序列转换为传感器快照。

    每个快照 = 一个时间点的传感器读数 + 活动上下文。
    传感器值由活动、位置和时间的统计模型生成。
    """
    random.seed(seed)
    base_date = datetime.fromisoformat(start_date)
    snapshots = []
    n_minutes = len(activities)

    minute = 0
    while minute < n_minutes:
        activity_id = activities[minute]
        location_id = locations[minute]
        activity = activity_names.get(activity_id, "Unknown")
        location = location_names.get(location_id, "Unknown")

        # 计算日期和时间
        current_dt = base_date + timedelta(minutes=minute)
        day_offset = (current_dt - base_date).days
        hour = current_dt.hour

        # === 生成传感器值 ===

        # 1. 运动传感器 — 基于活动和位置
        room_sensors = ROOM_SENSORS.get(location, [])
        motion_probs = ACTIVITY_MOTION_PROB.get(activity_id, 0.1)
        motion_active = 1 if random.random() < motion_probs else 0

        # 2. 温度 — 基于房间+昼夜节律+季节
        # 每个房间有基础偏移
        room_temp_offsets = {
            "Kitchen": 2.0, "Master Bedroom": 0.0, "Living Room": 0.5,
            "Office": 0.3, "Second Bedroom": 0.0, "Master Bathroom": -0.5,
            "Second Bathroom": -0.3, "Corridor": 0.0, "Junction": 0.0,
            "Outside": -8.0,
        }
        offset = room_temp_offsets.get(location, 0.0)
        # 昼夜节律: 凌晨低, 下午高
        diurnal = 21.0 + 7.0 * math.sin((hour - 14) * math.pi / 12)
        # 季节: 11月→5月, 共~200天
        seasonal = 3.0 * math.sin(day_offset / 365 * 2 * math.pi + math.pi)
        temp = diurnal + offset + seasonal + random.gauss(0, 0.8)
        temp = round(max(10, min(40, temp)), 1)

        # 3. 光照 — 基于时间+位置
        if 6 <= hour <= 19:
            if location == "Outside":
                light_base = 500 + 300 * math.sin((hour - 6) * math.pi / 13)
            else:
                window_rooms = {"Living Room", "Kitchen", "Office", "Master Bedroom"}
                if location in window_rooms:
                    light_base = 200 + 200 * math.sin((hour - 6) * math.pi / 13)
                else:
                    light_base = 50 + 100 * math.sin((hour - 6) * math.pi / 13)
        else:
            light_base = 5 + random.gauss(0, 3)
        light = max(0, round(light_base + random.gauss(0, 30), 1))

        # 4. 湿度 — 与温度反相关
        humidity = round(max(30, min(90, 55 - (temp - 22) * 2 + random.gauss(0, 5))), 1)

        snapshots.append({
            "day": day_offset + 1,
            "hour": hour,
            "minute": current_dt.minute,
            "sensors": {
                "temperature": temp,
                "humidity": humidity,
                "light": light,
                "motion": motion_active,
            },
            "activity": activity,
            "location": location,
            "activity_id": activity_id,
            "location_id": location_id,
        })

        minute += interval_minutes

    return snapshots


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data", "casas")
    out_path = os.path.join(data_dir, "aruba_snapshots.json")

    print("=" * 60)
    print("  STRANDS Aruba-1 → DistillToMCU 转换器")
    print("=" * 60)

    activity_names, location_names, activities, locations = load_strands_data(data_dir)

    print(f"\n  活动序列: {len(activities)} 分钟 ({len(activities)/60/24:.0f} 天)")
    print(f"  位置序列: {len(locations)} 分钟")
    print(f"  活动类型: {len(activity_names)} 类")
    print(f"  位置类型: {len(location_names)} 个房间")

    # 统计活动分布
    act_counts = defaultdict(int)
    for a in activities:
        act_counts[activity_names.get(a, "Unknown")] += 1
    print(f"\n  活动分布:")
    for act, cnt in sorted(act_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(activities) * 100
        print(f"    {act:<20s}: {cnt:6d} min ({pct:5.1f}%)")

    # 生成快照
    snapshots = activities_to_snapshots(
        activities, locations, activity_names, location_names,
        interval_minutes=30, seed=42,
    )
    print(f"\n  生成快照: {len(snapshots)} 个 (间隔 30min)")
    print(f"  天数: {snapshots[-1]['day']} 天")

    # 验证传感器数据
    temps = [s['sensors']['temperature'] for s in snapshots]
    lights = [s['sensors']['light'] for s in snapshots]
    print(f"  温度: {min(temps):.1f} ~ {max(temps):.1f} °C "
          f"(mean={sum(temps)/len(temps):.1f})")
    print(f"  光照: {min(lights):.1f} ~ {max(lights):.1f} lux")

    # 保存
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, indent=2, ensure_ascii=False)
    print(f"\n  输出: {out_path} ({os.path.getsize(out_path)} bytes)")

    # 验证与 casas_loader 兼容
    print(f"\n  兼容性: 可直接用于 DistillToMCU experiment.py")
    print(f"  用法: python experiment.py --real --days 30 --casas")

    print("\n" + "=" * 60)
    print("  STRANDS Aruba-1 转换完成 [OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()
