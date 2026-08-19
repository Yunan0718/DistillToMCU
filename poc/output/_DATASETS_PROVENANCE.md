# 数据集标注与出处（v10.7，9 个数据集）

全部数据集已按统一标准接入：4 个固定种子（42/123/999/777）在线运行
（`run4b_*_seed{seed}` 目录，30 天 held-out：1-21 天预热、22-30 天评估）
+ 60 快照 teacher-replay（每快照 3×DeepSeek T=0 + 2×T=0.7 + 1×Qwen）。

## 合成数据（虚拟，用于理想条件验证）

| key | 内容 | 状态 |
|-----|------|------|
| synthetic_seed42/123/999/777 | 30 天正弦温湿度/光照/运动，种子固定交互日程 | ✅ 完成 |

诚实声明：传感器为虚拟正弦波；仅用于方法在理想条件下的正确性验证，
不声称真实生态效度。

## 真实数据（5 个）

### UCI Occupancy Detection（uci_v3）
- 来源：UCI 仓库（CC BY 4.0），`poc/data/uci/snapshots_enriched.json`
- 传感器：temperature/humidity/light/co2/motion + 派生字段（真实办公室，恒温 19-24°C）
- 600 快照/30 天；教师动作率约 22%（低动作）
- 在线 held-out AR：48.2%（4 seed，第 22–30 天）；全周期 6 次重复 AR 32.4 ± 19.2%（CI 18.2–46.5，含 2026/31415 两次额外重复；池化 bootstrap CI 源文件：`statistics_4x.json` → `per_dataset.uci_v3.ci95`）
- teacher-replay：Ours P=50.0% (5/10) / DT 11.7% / ESP-Claw 100.0% (10/10, 覆盖 16.7%)
- UCI 额外 480 快照 × 3 seed teacher-replay：Ours P=50.0 ± 2.1%（合并 n_teacher=316）

### SML2010（sml2010）
- 来源：UCI 274，DOI 10.24432/C5RS3S，`poc/data/sml2010/`
- 内容：domotic house 室内气候，日历跨度约 50 天（2012-03-13 至 05-02，中间 4/11–4/18 无数据）；按 15 分钟采样连续计约 43 天，4,137 条
- 传感器：室内温度/湿度/光照/CO₂（卧室一路），无运动传感器（motion 诚实空缺，不发 None 给 LLM）
- 在线 AR：63.8 ± 4.2%（4 seed）
- teacher-replay：Ours P=83.1% / DT 70.0% / ESP-Claw 70.2%（Ours 占优）
- 注意：CO₂ 范围 187-609 ppm（室外通风水平），"CO₂ 高→通风"几乎不触发

### 钢铁工业能耗（steel）
- 来源：UCI 851，`poc/data/steel/Steel_industry_data.csv`
- 内容：钢铁厂能耗，35040 行，2018-2019 一年，15 分钟采样
- 传感器：usage_kwh / lagging_power / power_factor / co2 / load_level(1-3)
- 场景：工业能源控制（能耗高→冷却、功率因数低→节能、CO₂ 高→减载）
- 在线 AR：47.7 ± 10.9%（4 seed）
- teacher-replay：Ours P=52.8% / DT 30.0% / ESP-Claw 47.4%（Ours 占优）

### 城市空气质量（airquality）
- 来源：UCI 360，`poc/data/airquality/AirQualityUCI.csv`
- 内容：意大利城市空气质量，9471 行，2004-2005 一年，每小时
- 传感器：co / nox / no2 / temperature / humidity（缺失 -200 → None）
- 场景：空气质量/通风控制（污染物高→通风、NO₂ 高→净化、温度高→开窗）
- 在线 AR：68.1 ± 5.8%（4 seed）
- teacher-replay：Ours P=33.3% / DT 83.3% / ESP-Claw 97.8%（Ours 明显落后）

### STRANDS Aruba-1（strands_aruba1）
- 来源：CASAS Aruba-1 活动/位置标注（真实），传感器值合成补全
- 在线 AR：100%（held-out 窗口几乎全本地）
- teacher-replay：Ours warm P=0%（合成传感器补全导致区间规则过度泛化）→
  论文已排除 STRANDS 的保真度声明，仅用于覆盖率/AR

## 关键修复记录（必须保留，避免回归）

1. **motion:None 泄漏**（2026-08-19 修复）：`cloud_agent_think` 和
   `_agent_call` 之前把 `motion: None` 字面量发给 LLM。已改为过滤 None。
   修复后 SML2010 的 Ours 精度从 65.5% 变为 83.1%（结果反转）。
2. **基线字段硬编码**（2026-08-19 修复）：决策树/ESP-Claw/手写规则原来
   只认 temperature/humidity/light/motion。已改为自动推断数据集数值字段
   （`baselines.py`），手写规则按场景参数化（`USER_RULES_BY_LABEL`）。
   修复后 UCI 的 ESP-Claw 精度从 22.3% 变为 100.0%（用全部传感器字段）。

## 已知诚实声明（写入论文 Limitations）

- STRANDS 保真度失败（合成传感器补全），仅用覆盖率声明
- UCI 上 ESP-Claw 精度 100%（修复基线后），但覆盖仅 16.7%，与 Ours 精度 50% 对比
- Air Quality 上 Ours 精度 33.3%，落后于 DT（83.3%）和 ESP-Claw（97.8%）
- LLM 决策有 run-to-run 噪声（T=0 非完全确定），AR 需报告多次重复均值±std
- prompt 细节敏感（motion:None 修复改变 SML2010 结果）

## 关键数据文件位置

- 在线运行：`poc/output/run4b_{dataset}_seed{seed}/`（metrics/traces/rules）
- teacher-replay 主表：`poc/output/teacher_replay_results.json`（9 数据集）
- 一致性数据：`poc/output/llm_consistency_results*.json`
- 统计：`poc/output/statistics_4x.json`（AR）、
  `poc/output/statistics_4x_baselines.json`（36 blocks AR + 9 blocks agreement）、
  `poc/output/baseline_results_4x.json`（36 块全量）
- UCI 扩展：`poc/output/llm_consistency_results_uci_480_seed*.json`、
  `poc/output/uci_extended_replay_results_multiseed.json`
- 图表：`figures/fig_*.pdf/png`（11 张，v10.7）
- oracle 满信息容量上限：`poc/output/oracle_replay_all.json`
  （2026-08-19 扩展至全部 9 个数据集：sml2010=98.8%、steel=77.3%、
  airquality=96.8%，由 `poc/oracle_replay.py --all` 纯本地重算，无 API 调用）
