"""
DistillToMCU Phase 0 PoC — 环境模拟器
======================================
生成传感器数据和用户指令。支持两种模式：
1. SYNTHETIC: 随机传感器 + LLM 模拟住户
2. CASAS:   真实传感器 trace 回放 + LLM 模拟住户（待实现）

当前只实现 SYNTHETIC 模式用于快速验证。
"""

import random
import math
from datetime import datetime, timedelta
from config import (
    SENSOR_RANGES, SIMULATION_DAYS, INTERACTIONS_PER_DAY_MIN,
    INTERACTIONS_PER_DAY_MAX, ROUTINE_TASK_RATIO, SEED,
)


class SensorSimulator:
    """模拟真实物理传感器的行为，包括昼夜节律和随机波动"""

    def __init__(self, seed=SEED):
        random.seed(seed)
        self._hour = 0
        self._day = 0

    def sample(self, hour, day=0):
        """根据时间生成有物理意义的传感器读数"""
        self._hour = hour
        self._day = day

        # 温度：中午高，凌晨低，加入日内波动和长期趋势
        base_temp = 25.0 + 5 * math.sin((hour - 6) * math.pi / 12)
        temp = base_temp + random.gauss(0, 1.5)

        # 湿度：和温度反相关 + 随机
        hum = 55 - (temp - 25) * 2 + random.gauss(0, 5)

        # 光照：白天亮夜晚暗，中午峰值
        if 6 <= hour <= 19:
            light = 500 * max(0, math.sin((hour - 6) * math.pi / 13)) ** 0.5
            light += random.gauss(0, 50)
        else:
            light = max(0, random.gauss(5, 3))

        # 运动：白天活动概率高
        motion_prob = 0.3 if 8 <= hour <= 22 else 0.05
        motion = 1 if random.random() < motion_prob else 0

        return {
            "temperature": round(max(10, min(40, temp)), 1),
            "humidity": round(max(20, min(95, hum)), 1),
            "light": round(max(0, light), 1),
            "motion": motion,
        }

    def get_time_context(self, day, hour):
        """生成时间上下文文本"""
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
        weekday = weekdays[day % 7]
        ampm = "AM" if hour < 12 else "PM"
        h12 = hour if hour <= 12 else hour - 12
        h12 = 12 if h12 == 0 else h12
        time_str = f"{h12}:{random.choice(['00', '15', '30', '45'])} {ampm}"
        return f"Day {day+1}, {weekday}, {time_str}. The resident is at home."


class InteractionGenerator:
    """
    LLM-driven interaction generator。
    Phase 0 使用 SYNTHETIC 模式：随机传感器 + LLM 住户模拟。
    每个 interaction = 1 次完整的 "住户说话 → Agent 决策" 循环。
    """

    def __init__(self, sensor_sim, llm_client, seed=SEED):
        self.sensor_sim = sensor_sim
        self.llm = llm_client
        random.seed(seed)
        self.interaction_count = 0

    def generate_day(self, day, start_hour=7, end_hour=23):
        """生成一天的所有交互"""
        n = random.randint(INTERACTIONS_PER_DAY_MIN, INTERACTIONS_PER_DAY_MAX)
        interactions = []

        for _ in range(n):
            hour = random.randint(start_hour, end_hour)
            sensors = self.sensor_sim.sample(hour, day)
            time_ctx = self.sensor_sim.get_time_context(day, hour)

            # 生成 user_input：有 LLM 走 LLM，没有用 mock
            if self.llm and hasattr(self.llm, 'resident_simulate'):
                resident_result = self.llm.resident_simulate(sensors, time_ctx)
                user_input = (resident_result.get("content") or "").strip().strip('"')
            else:
                user_input = self._mock_resident_input(sensors, hour)

            if not user_input:
                user_input = "..."

            interactions.append({
                "day": day + 1,
                "hour": hour,
                "time_context": time_ctx,
                "sensors": sensors,
                "user_input": user_input,
                "is_routine": random.random() < ROUTINE_TASK_RATIO,
            })
            self.interaction_count += 1

        interactions.sort(key=lambda x: x["hour"])
        return interactions

    def _mock_resident_input(self, sensors, hour):
        """传感器驱动的住户语音生成（优先产生需要设备控制的输入）"""
        temp = sensors.get("temperature", 25)
        light = sensors.get("light", 500)
        motion = sensors.get("motion", 0)
        actions = []   # 需要设备控制
        queries = []   # 纯查询（不产生 tool_call）

        # --- 温度相关 ---
        if temp > 32:
            actions += ["太热了！快开风扇！", "闷死了，风扇开到最大",
                        "热得受不了", "能不能开一下风扇"]
        elif temp > 29:
            actions += ["有点热", "开风扇吧", "好热啊", "怎么这么闷"]
        elif temp < 16:
            actions += ["冷死了", "太冷了", "冻死了"]
        elif temp < 20:
            actions += ["有点凉", "稍微有点冷"]
        else:
            queries += ["温度挺舒适的", "还行"]

        # --- 光线相关 ---
        if light < 20:
            actions += ["太暗了看不清", "开灯！", "灯在哪", "好黑啊"]
        elif light < 60:
            actions += ["有点暗", "开灯吧", "光线不太好"]
        elif light > 900:
            actions += ["太刺眼了", "好亮啊", "拉上窗帘"]
        else:
            if random.random() < 0.2:
                queries += ["现在什么温度"]

        # --- 运动 ---
        if motion == 1 and random.random() < 0.5:
            actions += ["我回来了"]
        if motion == 0 and random.random() < 0.05:
            queries += ["有人吗"]

        # --- 时段 ---
        if 6 <= hour <= 8:
            actions += ["早上了", "起床了", "拉开窗帘"]
        elif 19 <= hour <= 21:
            if random.random() < 0.3:
                actions += ["天黑了，开灯吧"]
        elif 21 <= hour <= 23:
            actions += ["准备睡觉了", "关灯吧", "晚安"]

        # 80% 概率选动作，20% 概率选查询
        if actions and (random.random() < 0.8 or not queries):
            return random.choice(actions)
        elif queries:
            return random.choice(queries)
        else:
            return "还行"
