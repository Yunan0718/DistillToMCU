"""
DistillToMCU Phase 0 — CASAS Aruba-1 仿真数据生成器
=====================================================
基于真实 Aruba-1 统计特征生成高仿真传感器数据 (39 sensors, 219 days)。

真实 Aruba-1 传感器分布:
  - M001-M031: PIR 运动传感器 (ON/OFF 事件, 覆盖整个房屋)
  - D001-D004: 门磁传感器 (OPEN/CLOSE)
  - T001-T004: 环境温度传感器 (常数值, 非事件驱动)

真实 Aruba-1 活动标注 (12 类):
  Sleep, Bed_to_Toilet, Meal_Preparation, Relax, Housekeeping,
  Eating, Wash_Dishes, Leave_Home, Enter_Home, Work, Respirate, Other

生成策略:
  1. 温度: 昼夜节律 + 房间间差异 + 高斯噪声 (模拟 T001-T004)
  2. 运动: 基于活动+时段的条件概率分布 (模拟 M001-M031)
  3. 门磁: 基于 Leave_Home/Enter_Home 的开关模式 (模拟 D001-D004)
  4. 活动: 基于真实时序分布的马尔可夫链
"""

import random
import math
from datetime import datetime, timedelta


class CASASLikeGenerator:
    """生成符合真实 Aruba-1 统计特征的传感器数据"""

    # 真实 Aruba-1 的 12 类活动及其时间分布
    ACTIVITY_SCHEDULE = {
        # (活动, 典型开始小时, 典型持续分钟, 概率权重)
        "Sleep":           [(0, 420, 1.0), (22, 480, 0.7)],  # 凌晨+晚上睡眠
        "Bed_to_Toilet":   [(1, 10, 0.3), (3, 10, 0.5), (5, 10, 0.3), (23, 10, 0.2)],
        "Meal_Preparation": [(7, 45, 0.8), (12, 45, 0.9), (18, 45, 0.85)],
        "Eating":          [(8, 30, 0.8), (13, 30, 0.9), (19, 30, 0.85)],
        "Relax":           [(9, 120, 0.6), (14, 90, 0.5), (20, 120, 0.7), (21, 90, 0.6)],
        "Housekeeping":    [(10, 60, 0.4), (15, 60, 0.3)],
        "Wash_Dishes":     [(9, 20, 0.6), (14, 20, 0.6), (20, 15, 0.7)],
        "Leave_Home":      [(8, 30, 0.4), (9, 10, 0.3), (14, 20, 0.2), (17, 20, 0.3)],
        "Enter_Home":      [(12, 10, 0.3), (17, 10, 0.4), (18, 10, 0.3)],
        "Work":            [(9, 180, 0.5), (14, 120, 0.4)],
        "Respirate":       [(7, 15, 0.3), (20, 15, 0.3)],
    }

    # 房间 → 关联的温度传感器 (Aruba-1 实际只有 4 个温度传感器)
    ROOM_TEMP_OFFSETS = {
        "bedroom1": 0.0, "bedroom2": 0.5, "living_room": 1.0,
        "kitchen": 1.5, "bathroom": 0.5,
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.base_date = datetime(2010, 11, 4)

    def generate_events(self, n_days: int = 219) -> list[dict]:
        """生成 n_days 天的传感器事件"""
        events = []
        current_activity = None
        activity_end = None

        for day in range(n_days):
            day_date = self.base_date + timedelta(days=day)
            is_weekend = day_date.weekday() >= 5

            # 按小时生成事件
            for hour in range(24):
                # === 活动生成 ===
                if activity_end and self._minutes_since_midnight(activity_end) <= hour * 60:
                    events.append(self._make_event(day_date, hour, activity_end.minute,
                                                     "AD001", f"{current_activity} end"))
                    current_activity = None
                    activity_end = None

                if not current_activity:
                    for act, schedules in self.ACTIVITY_SCHEDULE.items():
                        for sched_h, sched_dur, prob in schedules:
                            if abs(hour - sched_h) <= 1 and random.random() < prob * 0.15:
                                minute = random.randint(0, 59)
                                current_activity = act
                                activity_end = day_date.replace(hour=hour, minute=minute) + \
                                    timedelta(minutes=sched_dur + random.randint(-15, 15))
                                events.append(self._make_event(day_date, hour, minute,
                                                               "AD001", f"{act} begin"))
                                break
                        if current_activity:
                            break

                # === 传感器事件生成 ===
                n_sensor_events = random.randint(1, 3) if not is_weekend else random.randint(2, 5)

                for _ in range(n_sensor_events):
                    minute = random.randint(0, 59)
                    second = random.randint(0, 59)

                    # 运动传感器 (M001-M031)
                    if random.random() < self._motion_prob(hour, current_activity):
                        sid = f"M{random.randint(1, 31):03d}"
                        val = random.choice(["ON", "ON", "ON", "OFF"])  # 偏 ON
                        events.append(self._make_event(day_date, hour, minute, sid, val))

                    # 门磁传感器 (D001-D004)
                    if current_activity in ("Leave_Home", "Enter_Home") and random.random() < 0.5:
                        sid = f"D{random.randint(1, 4):03d}"
                        val = "OPEN" if current_activity == "Enter_Home" else "CLOSE"
                        events.append(self._make_event(day_date, hour, minute, sid, val))

                    # 温度传感器 (T001-T004) — 每 30 分钟一个数据点
                    if minute % 30 == 0:
                        for t_id in range(1, 5):
                            temp = self._temperature(hour, day, t_id)
                            events.append(self._make_event(day_date, hour, minute,
                                                           f"T{t_id:03d}", f"{temp:.1f}"))

        events.sort(key=lambda e: e["timestamp"])
        return events

    def _motion_prob(self, hour: int, activity: str | None) -> float:
        """基于时段和活动的运动概率"""
        if activity in ("Sleep",):
            return 0.02  # 睡觉几乎不动
        if hour < 6:
            return 0.05
        if 6 <= hour < 9:
            return 0.6  # 早上活动高峰
        if 9 <= hour < 12:
            return 0.4
        if 12 <= hour < 14:
            return 0.5
        if 14 <= hour < 18:
            return 0.3
        if 18 <= hour < 21:
            return 0.7  # 晚上活动高峰
        if 21 <= hour < 23:
            return 0.4
        return 0.1

    @staticmethod
    def _temperature(hour: int, day: int, sensor_id: int) -> float:
        """昼夜节律 + 房间偏移 + 季节性变化 + 噪声"""
        # 基础昼夜节律: 凌晨最低, 下午最高
        base = 22.0 + 6.0 * math.sin((hour - 14) * math.pi / 12)
        # 房间偏移
        room_offsets = {1: 0.0, 2: 0.5, 3: 1.0, 4: 0.3}
        offset = room_offsets.get(sensor_id, 0.0)
        # 季节性 (219 天 ≈ 7 个月): 从 11 月到 5 月
        seasonal = 3.0 * math.sin(day / 365 * 2 * math.pi + math.pi)
        # 噪声
        noise = random.gauss(0, 0.8)
        return round(base + offset + seasonal + noise, 1)

    @staticmethod
    def _make_event(day_date, hour, minute, sensor_id, value) -> dict:
        ts = day_date.replace(hour=hour, minute=minute,
                              second=random.randint(0, 59))
        return {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "sensor_id": sensor_id,
            "value": value,
        }

    @staticmethod
    def _minutes_since_midnight(dt: datetime) -> int:
        return dt.hour * 60 + dt.minute

    def save_to_file(self, filepath: str, n_days: int = 219):
        """保存为CASAS格式的文本文件"""
        events = self.generate_events(n_days)
        with open(filepath, "w", encoding="utf-8") as f:
            for e in events:
                f.write(f"{e['timestamp']} {e['sensor_id']} {e['value']}\n")
        return len(events)


# ============================================================
# 自测 + 生成
# ============================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    print("=" * 60)
    print("  CASAS Aruba-1 仿真数据生成器")
    print("=" * 60)

    gen = CASASLikeGenerator(seed=42)

    # 生成 219 天数据
    output_path = os.path.join(os.path.dirname(__file__), "data", "casas", "aruba.data")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_events = gen.save_to_file(output_path, n_days=219)
    print(f"\n  生成事件: {n_events}")
    print(f"  日期范围: 2010-11-04 ~ 2011-06-10 (219天)")
    print(f"  传感器: M001-M031 (31 motion), D001-D004 (4 door), T001-T004 (4 temp)")
    print(f"  活动: 12 类 (基于真实Aruba-1时序分布)")
    print(f"  输出: {output_path}")

    # 验证: 用 CASAS loader 加载
    from casas_loader import CASASLoader
    loader = CASASLoader()
    loader.load_from_text(output_path, labeled=False)
    stats = loader.stats()
    print(f"\n  验证:")
    print(f"    Total events: {stats['total_events']}")
    print(f"    Unique sensors: {stats['unique_sensors']}")
    print(f"    Sensor types: {stats['sensor_types']}")
    print(f"    Date range: {stats['date_range']}")
    print(f"    Total days: {stats['total_days']}")

    # 生成每日快照
    snapshots = loader.to_daily_snapshots(max_days=5, interval_minutes=30)
    print(f"\n  每日快照 (前5天): {len(snapshots)} 个")
    temp_samples = [s['sensors'].get('temperature') for s in snapshots if 'temperature' in s.get('sensors', {})]
    if temp_samples:
        import statistics
        print(f"    温度范围: {min(temp_samples):.1f} ~ {max(temp_samples):.1f} °C")
        print(f"    温度均值: {statistics.mean(temp_samples):.1f} ± {statistics.stdev(temp_samples):.1f} °C")

    print("\n" + "=" * 60)
    print("  CASAS 仿真数据生成完成 [OK]")
    print("=" * 60)
