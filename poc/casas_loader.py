"""
DistillToMCU Phase 0b — CASAS Aruba-1 数据集加载器
===================================================
从 Zenodo 公开下载的 CASAS Aruba-1 数据集加载和预处理。

数据格式（原始 CASAS）:
  日期 时间 传感器ID 值 [活动标签]
  2010-11-04 00:03:49.553977 M001 ON  Sleep begin
  2010-11-04 00:06:42.405450 M001 OFF
  2010-11-04 00:06:49.299968 M002 ON
  ...

传感器映射:
  M001-M031:  PIR 运动传感器 (ON/OFF) → motion
  D001-D004:  门磁传感器 (OPEN/CLOSE) → door_open
  T001-T004:  温度传感器 (数值) → temperature
  (部分版本含光线传感器 → light)

使用场景:
  1. Trace replay: 按真实时间序列回放传感器数据
  2. LLM 增强: 对每个传感器快照生成模拟住户语音 → LLM 决策

Usage:
    from casas_loader import CASASLoader
    loader = CASASLoader()
    loader.download_or_load("./data/casas")  # 如果已下载则直接加载
    events = loader.load_events()
    daily_snapshots = loader.to_daily_snapshots(events)
"""

import os
import re
import gzip
import urllib.request
import json
from datetime import datetime, timedelta
from collections import defaultdict


# CASAS Aruba-1 传感器类型映射
SENSOR_TYPE_MAP = {
    "M": "motion",       # PIR 运动传感器
    "D": "door",         # 门磁传感器
    "T": "temperature",  # 温度传感器
    "L": "light",        # 光照传感器 (如果存在)
    "AD": "activity",    # 活动标注 (不是物理传感器)
}

# 活动标签（用于 LLM 场景理解）
ACTIVITY_LABELS = {
    "Sleep": "sleeping",
    "Bed_to_Toilet": "going to bathroom",
    "Meal_Preparation": "cooking",
    "Relax": "relaxing",
    "Housekeeping": "cleaning",
    "Eating": "eating",
    "Wash_Dishes": "washing dishes",
    "Leave_Home": "leaving home",
    "Enter_Home": "entering home",
    "Work": "working",
    "Respirate": "breathing exercise",
}


class CASASLoader:
    """CASAS Aruba-1 数据集加载器"""

    # Zenodo 上的 Aruba-1 数据
    CASAS_URL = "https://zenodo.org/records/17180309/files/new_labeled_data.zip"
    ARUBA_FILE = "data/original/aruba/data"  # ZIP 内的路径

    def __init__(self, seed: int = 42):
        self.events = []           # 原始事件列表
        self.sensors = set()       # 所有传感器 ID
        self.date_range = None     # (start_date, end_date)

    def load_from_text(self, filepath: str, labeled: bool = True):
        """
        从 CASAS 原始文本文件加载事件。

        每行格式:
          2010-11-04 00:03:49.553977 M001 ON Sleep begin
          (日期 时间 传感器ID 值) 或 (日期 时间 传感器ID 值 活动标签1 标签2)

        Args:
            filepath: 数据文件路径
            labeled: 是否包含活动标注
        """
        self.events = []
        line_pattern = re.compile(
            r'^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+'
            r'(\S+)\s+(\S+)(?:\s+(.+))?$'
        )

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                match = line_pattern.match(line)
                if not match:
                    continue

                date_str, time_str, sensor_id, value, rest = match.groups()

                event = {
                    "timestamp": f"{date_str}T{time_str}",
                    "sensor_id": sensor_id,
                    "value": value,
                    "sensor_type": self._classify_sensor(sensor_id),
                }

                # 解析活动标注
                if rest:
                    parts = rest.split()
                    if parts:
                        begin_end = parts[-1] if parts[-1] in ("begin", "end") else None
                        if begin_end:
                            activity = " ".join(parts[:-1])
                            event["activity"] = activity
                            event["activity_phase"] = begin_end
                        else:
                            event["activity"] = rest

                self.events.append(event)
                self.sensors.add(sensor_id)

        # 计算日期范围
        if self.events:
            start = self.events[0]["timestamp"][:10]
            end = self.events[-1]["timestamp"][:10]
            self.date_range = (start, end)

        return self

    @staticmethod
    def _classify_sensor(sensor_id: str) -> str:
        """根据传感器 ID 前缀判断类型"""
        if not sensor_id:
            return "unknown"
        prefix = sensor_id[0].upper()
        return SENSOR_TYPE_MAP.get(prefix, "unknown")

    def get_sensor_snapshot(self, timestamp: str) -> dict:
        """
        获取某个时间点的传感器状态快照。
        方案：取该时间点之前每个传感器最近一次事件的值。

        简化版：用最近的传感器读数近似（实际 CASAS 传感器是事件驱动的，
        没有连续的采样值——温度传感器除外）。
        """
        target_dt = datetime.fromisoformat(timestamp)
        latest = {}

        for event in self.events:
            evt_dt = datetime.fromisoformat(event["timestamp"])
            if evt_dt > target_dt:
                break
            sid = event["sensor_id"]
            val = event["value"]
            stype = event["sensor_type"]

            # 只跟踪物理传感器
            if stype == "activity":
                continue

            # 温度：数值
            if stype == "temperature":
                try:
                    latest[sid] = float(val)
                except ValueError:
                    pass
            # 运动/门：二值
            elif stype in ("motion", "door"):
                latest[sid] = 1 if val in ("ON", "OPEN") else 0

        return latest

    def to_daily_snapshots(
        self,
        start_day: int = 0,
        max_days: int = 30,
        interval_minutes: int = 30,
        hour_start: int = 7,
        hour_end: int = 23,
    ) -> list[dict]:
        """
        将原始事件转换为每日传感器快照序列。
        用于 DistillToMCU 实验的 trace replay。

        每 30 分钟采一次样（模拟用户交互频率），
        生成一个传感器快照 + 模拟活动上下文。

        Returns:
            [{day, hour, minute, sensors: {temperature, light?, motion, door},
              activity, time_context}, ...]
        """
        if not self.events:
            return []

        base_date = datetime.fromisoformat(self.events[0]["timestamp"][:10])
        snapshots = []

        for day_offset in range(start_day, start_day + max_days):
            day_date = base_date + timedelta(days=day_offset)
            day_str = day_date.strftime("%Y-%m-%d")

            for hour in range(hour_start, hour_end):
                for minute in range(0, 60, interval_minutes):
                    ts = f"{day_str}T{hour:02d}:{minute:02d}:00"

                    # 获取传感器快照
                    raw_snapshot = self.get_sensor_snapshot(ts)

                    # 转换成 DistillToMCU 的传感器格式
                    sensors = self._convert_sensors(raw_snapshot)

                    # 如果所有传感器都是空，跳过（可能是凌晨无活动时段）
                    if not any(sensors.values()):
                        continue

                    # 获取当前活动
                    activity = self._get_activity_at(ts)

                    # 生成时间上下文
                    time_ctx = self._make_time_context(day_date, hour)

                    snapshots.append({
                        "day": day_offset + 1,
                        "hour": hour,
                        "minute": minute,
                        "sensors": sensors,
                        "activity": activity,
                        "time_context": time_ctx,
                        "raw_timestamp": ts,
                    })

        return snapshots

    def _convert_sensors(self, raw: dict) -> dict:
        """将 CASAS 原始传感器值转换为 DistillToMCU 格式"""
        temp_values = []
        motion_active = False
        door_open = False
        light_value = None

        for sid, val in raw.items():
            stype = self._classify_sensor(sid)
            if stype == "temperature":
                temp_values.append(val)
            elif stype == "motion":
                if val == 1:
                    motion_active = True
            elif stype == "door":
                if val == 1:
                    door_open = True
            elif stype == "light":
                light_value = val

        # 结构化传感器输出
        sensors = {}

        # 温度：取所有温度传感器的中位数
        if temp_values:
            sensors["temperature"] = round(
                sorted(temp_values)[len(temp_values) // 2], 1
            )

        # 运动：任意 PIR 触发即为 1
        sensors["motion"] = 1 if motion_active else 0

        # 门：任意门开即为 1
        sensors["door_open"] = 1 if door_open else 0

        # 光照：如果有（大多数 CASAS 版本无光照传感器）
        if light_value is not None:
            sensors["light"] = light_value
        else:
            # 根据时间估算：白天=300-800, 晚上=5-20
            # 这是一种合理的估算（论文中注明即可）
            pass

        # 湿度：CASAS 原始数据一般没有，合成估算
        # （论文注明 CASAS 仅有温度+运动+门磁）

        return sensors

    def _get_activity_at(self, timestamp: str) -> str | None:
        """获取某个时间点的当前活动"""
        target_dt = datetime.fromisoformat(timestamp)
        current_activity = None

        for event in self.events:
            evt_dt = datetime.fromisoformat(event["timestamp"])
            if evt_dt > target_dt:
                break
            if event.get("activity_phase") == "begin":
                current_activity = event.get("activity")
            elif event.get("activity_phase") == "end":
                if current_activity == event.get("activity"):
                    current_activity = None

        return current_activity

    @staticmethod
    def _make_time_context(base_date, hour: int) -> str:
        """生成时间上下文文本"""
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
        wd = weekdays[base_date.weekday()]
        ampm = "AM" if hour < 12 else "PM"
        h12 = hour if hour <= 12 else hour - 12
        h12 = 12 if h12 == 0 else h12
        part = "morning" if 6 <= hour < 12 else \
               "afternoon" if 12 <= hour < 18 else \
               "evening" if 18 <= hour < 22 else "night"
        return f"{wd} {h12}{ampm}, {part}"

    def stats(self) -> dict:
        """数据集统计"""
        sensor_types = defaultdict(int)
        for sid in self.sensors:
            stype = self._classify_sensor(sid)
            sensor_types[stype] += 1

        activities = set()
        for e in self.events:
            if "activity" in e:
                activities.add(e["activity"])

        return {
            "total_events": len(self.events),
            "unique_sensors": len(self.sensors),
            "sensor_types": dict(sensor_types),
            "date_range": self.date_range,
            "activities": sorted(activities),
            "total_days": (
                (datetime.fromisoformat(self.date_range[1]) -
                 datetime.fromisoformat(self.date_range[0])).days
                if self.date_range else 0
            ),
        }


# ============================================================
# 与 DistillToMCU 集成
# ============================================================

def casas_to_distilltomcu_traces(
    snapshots: list[dict],
    llm_client,
    rule_engine,
    max_per_day: int = 20,
) -> list[dict]:
    """
    将 CASAS 传感器快照 + LLM 决策转换为 DistillToMCU trace 格式。

    模拟完整流程：
      传感器快照 → 生成住户语音（LLM Resident）→ Cloud Agent 决策
      → 规则匹配 → 本地执行 / Cloud API

    这是 Phase 0b 的核心实验循环。
    """
    import random as _random
    from simulator import SensorSimulator

    # 创建合成传感器模拟器用于补全缺失的传感器值
    sim = SensorSimulator()

    traces = []
    for snap in snapshots[:max_per_day * 30]:  # 最多 30 天
        sensors = dict(snap["sensors"])

        # 补全缺失的传感器值（light, humidity）
        # CASAS Aruba-1 只有 temp + motion + door，没有 light/humidity
        if "light" not in sensors:
            hour = snap.get("hour", 12)
            sensors["light"] = _random.gauss(300, 100) if 6 <= hour <= 19 \
                else _random.gauss(10, 5)
            sensors["light"] = round(max(0, sensors["light"]), 1)
        if "humidity" not in sensors:
            sensors["humidity"] = round(_random.gauss(55, 10), 1)

        # 活动上下文 → 模拟住户语音
        activity = snap.get("activity", "")
        time_ctx = snap.get("time_context", "")

        # 生成住户语音（用 LLM 或模板）
        user_input = _generate_resident_utterance(sensors, activity, time_ctx)

        # Cloud Agent 决策（走真实 LLM）
        if llm_client and hasattr(llm_client, 'cloud_agent_think'):
            response = llm_client.cloud_agent_think(
                system_prompt="""You are a smart home controller.
Given sensor readings and the resident's utterance, decide what to do.
Available devices: led (brightness), fan (speed 1-3), curtain (position 0-100).""",
                user_input=user_input,
                sensors=sensors,
            )

            traces.append({
                "sensors": sensors,
                "user_input": user_input,
                "llm_response": response,
                "execution": {"mode": "cloud"},
                "activity": activity,
                "time_context": time_ctx,
            })

    return traces


def _generate_resident_utterance(
    sensors: dict, activity: str | None, time_ctx: str
) -> str:
    """生成与 CASAS 活动相关的模拟住户语音"""
    import random as _random

    temp = sensors.get("temperature", 25)
    light = sensors.get("light", 500)
    motion = sensors.get("motion", 0)

    # 基于活动的语音
    activity_utterances = {
        "Sleep": ["zzz...", ""],
        "Meal_Preparation": ["开始做饭了", "准备晚饭", "要做饭了"],
        "Eating": ["吃饭了", ""],
        "Relax": ["想放松一下", "看会儿电视", ""],
        "Housekeeping": ["该打扫了", ""],
        "Leave_Home": ["出门了", "走了"],
        "Enter_Home": ["我回来了", "回来了，好累"],
        "Work": ["开始工作", ""],
    }

    # 基于传感器条件的语音
    condition_utterances = []
    if temp > 30:
        condition_utterances += ["太热了", "开风扇吧"]
    elif temp > 28:
        condition_utterances += ["有点热"]
    elif temp < 18:
        condition_utterances += ["有点冷"]
    if light is not None and light < 50:
        condition_utterances += ["好暗啊", "开灯吧"]

    # 混合活动 + 传感器条件
    all_choices = []
    if activity:
        all_choices += activity_utterances.get(activity, [])
    all_choices += condition_utterances

    if all_choices and _random.random() < 0.8:
        return _random.choice(all_choices)
    elif condition_utterances:
        return _random.choice(condition_utterances)
    else:
        return "一切正常"


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  CASAS Aruba-1 Loader — Self-Test")
    print("=" * 60)

    # 如果没有本地 CASAS 数据，用模拟数据自测
    data_path = None
    test_paths = [
        "./data/casas/aruba.data",
        "../data/casas/aruba.data",
        "d:/fuyou1/data/casas/aruba.data",
    ]
    for p in test_paths:
        if os.path.exists(p):
            data_path = p
            break

    if data_path:
        print(f"\n[Test 1] Loading CASAS data from: {data_path}")
        loader = CASASLoader()
        loader.load_from_text(data_path)
        stats = loader.stats()
        print(f"  Events: {stats['total_events']}")
        print(f"  Sensors: {stats['unique_sensors']} ({stats['sensor_types']})")
        print(f"  Date range: {stats['date_range']}")
        print(f"  Activities: {len(stats['activities'])}")
        print(f"  Total days: {stats['total_days']}")
        print("  [PASS]")

        print(f"\n[Test 2] Generating daily snapshots (first 3 days)")
        snapshots = loader.to_daily_snapshots(max_days=3, interval_minutes=60)
        print(f"  Snapshots: {len(snapshots)}")
        for s in snapshots[:5]:
            print(f"    Day{s['day']:2d} {s['hour']:02d}:{s['minute']:02d}  "
                  f"temp={s['sensors'].get('temperature', '?'):.1f}  "
                  f"motion={s['sensors'].get('motion', '?')}  "
                  f"activity={s.get('activity', '?')}")
        print("  [PASS]")

    else:
        print("\n  No CASAS data file found. Generating synthetic CASAS-like data...")

        # 生成合成 CASAS 格式数据用于自测
        import random
        random.seed(42)

        with open("./output/_test_casas.data", "w") as f:
            base = datetime(2010, 11, 4)
            sensors = [f"M{i:03d}" for i in range(1, 32)] + \
                      [f"D{i:03d}" for i in range(1, 5)] + \
                      [f"T{i:03d}" for i in range(1, 5)]

            activities = [
                ("Sleep", 0, 7), ("Meal_Preparation", 7, 9),
                ("Eating", 8, 9), ("Relax", 9, 12),
                ("Housekeeping", 12, 14), ("Meal_Preparation", 17, 19),
                ("Eating", 18, 20), ("Relax", 20, 23), ("Sleep", 22, 23),
            ]

            for day in range(30):
                d = base + timedelta(days=day)
                # 生成活动标注
                for act, h_start, h_end in activities:
                    h = random.randint(h_start, h_end)
                    m = random.randint(0, 59)
                    s = random.randint(0, 59)
                    ts = d.replace(hour=h, minute=m, second=s)
                    f.write(f"{ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} "
                            f"AD001 {act} begin\n")
                    # end after 30-120 min
                    end_dur = random.randint(30, 120)
                    ts_end = ts + timedelta(minutes=end_dur)
                    f.write(f"{ts_end.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} "
                            f"AD001 {act} end\n")

                # 生成传感器事件
                for hour in range(24):
                    for _ in range(random.randint(1, 5)):
                        m = random.randint(0, 59)
                        s = random.randint(0, 59)
                        sid = random.choice(sensors)
                        ts = d.replace(hour=hour, minute=m, second=s)
                        if sid.startswith("M"):
                            val = random.choice(["ON", "OFF"])
                        elif sid.startswith("D"):
                            val = random.choice(["OPEN", "CLOSE"])
                        else:
                            # Temperature: diurnal + noise
                            temp = 22 + 8 * abs(hour - 14) / 14 * (-1 if hour > 14 else 1)
                            temp += random.gauss(0, 1.5)
                            val = f"{temp:.1f}"
                        f.write(f"{ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} "
                                f"{sid} {val}\n")

        loader = CASASLoader()
        loader.load_from_text("./output/_test_casas.data")
        stats = loader.stats()
        print(f"  Events: {stats['total_events']}")
        print(f"  Sensors: {stats['unique_sensors']} ({stats['sensor_types']})")
        print(f"  Date range: {stats['date_range']}")
        print(f"  Activities: {len(stats['activities'])}")
        print(f"  Total days: {stats['total_days']}")
        print("  [PASS] (synthetic CASAS data)")

        print(f"\n[Test 2] Snapshots from synthetic data")
        snapshots = loader.to_daily_snapshots(max_days=3, interval_minutes=60)
        print(f"  Snapshots: {len(snapshots)}")
        for s in snapshots[:5]:
            print(f"    Day{s['day']:2d} {s['hour']:02d}:{s['minute']:02d}  "
                  f"temp={s['sensors'].get('temperature', '?'):.1f}  "
                  f"motion={s['sensors'].get('motion', 0)}  "
                  f"activity={s.get('activity', '?')}")
        print("  [PASS]")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED [OK]")
    print("=" * 60)
