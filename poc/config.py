"""
DistillToMCU Phase 0 PoC — 全局配置
=====================================
所有可调参数集中在这里，方便实验时快速修改。
"""

# ========== DeepSeek API ==========
LLM_MODEL = "deepseek-v4-flash"        # deepseek-v4-flash (便宜) / deepseek-v4-pro
LLM_BASE_URL = "https://api.deepseek.com"
# v7: 不再默认提供占位符——必须设置环境变量。没有 Key 就用 --mock。
LLM_API_KEY = __import__("os").environ.get("DEEPSEEK_API_KEY", "")
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE_AGENT = 0.0            # Cloud Agent — 低温度，决策一致
LLM_TEMPERATURE_RESIDENT = 0.8         # Resident Simulator — 高温度，行为多样
LLM_TEMPERATURE_CHECK = 0.1            # Sanity Check — 低温度，判断严格

# ========== 实验参数 ==========
SIMULATION_DAYS = 30
INTERACTIONS_PER_DAY_MIN = 10
INTERACTIONS_PER_DAY_MAX = 25
ROUTINE_TASK_RATIO = 0.6               # 60% routine tasks, 40% open-ended
SEED = 42

# ========== 规则蒸馏 ==========
RULE_MIN_EVIDENCE = 3                  # 至少 N 次证据才从 candidate → verified
RULE_CONFIDENCE_THRESHOLD_LOCAL = 0.8  # confidence > 0.8 直接本地执行
RULE_CONFIDENCE_THRESHOLD_ASYNC = 0.5  # 0.5-0.8 之间本地执行 + 异步 LLM 确认
RULE_WILSON_Z = 1.96                   # 95% confidence
RULE_DECAY_TAU_BASE = 7.0              # 时间衰减基期 (天)
RULE_MAX_ACTIVE = 500                  # 活跃规则上限
RULE_RETIRE_AFTER_DAYS = 14            # degraded 超过 N 天 → retired
DISCRETE_FREQUENCY_THRESHOLD = 0.8     # 离散传感器纳入条件的最低频率
MAJORITY_PARAM_THRESHOLD = 0.7         # 参数值被纳入偏好规则的最低多数比例
RULE_LIFECYCLE_CONFIDENCE_CANDIDATE = 0.70   # candidate → verified 最低置信度
RULE_LIFECYCLE_CONFIDENCE_ACTIVE = 0.85      # verified → active 最低置信度
RULE_LIFECYCLE_CONFIDENCE_DEGRADE = 0.60     # 低于此值 → degraded

# ========== 安全等级 ==========
SAFETY_L0 = 0   # 只读/查询 — 可自动学习
SAFETY_L1 = 1   # 舒适类（灯/风扇）— 可自动学习
SAFETY_L2 = 2   # 高能耗（空调/热水器）— 需用户首次确认
SAFETY_L3 = 3   # 安全关键（门锁/报警）— 永不自动本地化

# ========== 传感器配置 ==========
SENSORS = ["temperature", "humidity", "light", "motion"]

# 传感器正常范围 (用于随机模拟，CASAS 模式下用真实数据)
SENSOR_RANGES = {
    "temperature": (15.0, 35.0),   # °C
    "humidity":    (30.0, 90.0),   # %
    "light":       (5.0,  1000.0), # lux
    "motion":      (0,    1),      # 0/1
}

# ========== 执行器配置 ==========
ACTUATORS = {
    "led":     {"safety": SAFETY_L1, "type": "pwm",   "params": ["brightness", "color_temp"]},
    "fan":     {"safety": SAFETY_L1, "type": "pwm",   "params": ["speed"]},
    "curtain": {"safety": SAFETY_L1, "type": "servo", "params": ["position"]},
}

# ========== 输出路径 ==========
OUTPUT_DIR = "./output"
TRACE_FILE = "traces.jsonl"
RULES_FILE = "rules_snapshot.json"
METRICS_FILE = "metrics.jsonl"
