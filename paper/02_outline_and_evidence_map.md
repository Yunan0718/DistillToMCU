# DistillToMCU — Paper Outline + Evidence Map

> academic-paper skill · Phase 2（structure_architect_agent 交付物）
> 结构：IMRaD（系统论文变体，Related Work 独立成节）· 正文 ~9,000 词

## Overview

一句话论点：**观察云端 LLM 的控制行为轨迹，蒸馏出带逐规则置信度与生命周期的区间规则，
让 MCU 以 1.48ms 匹配、零 LLM 依赖地本地执行——自主率与手写规则同档，保真度落在教师
自一致性包络内，且在真实传感器数据上以校准后的克制执行稳定占优（UCI 合并回放：50.0±2.1% vs 批量树 16.5%；SML2010 83.1% vs 70.0%；Steel 52.8% vs 30.0%；Air Quality 落后需诚实报告）。**
结构逻辑：先立"第三类空白"（Introduction + Related Work），再给系统（COMIC + PMS +
生命周期 + MCU 映射），再给公平评估协议（held-out + teacher-replay），再给结果
（先 AR/CCR，再 AGREE/P/R，再硬件），最后讨论局限。

---

## Detailed Outline

### 1. Introduction（~900 词）

**Purpose**：三句话立空白——所有人让 LLM 留在回路（HearthNet/EdgeTalk-MCU/DCP/ESP-Claw），
或把 trace 蒸馏给仍跑 LLM 的云 Agent（ClawTrace/RIMRULE/Skill-DisCo）；本项目做第三件事：
trace → MCU 可执行规则，运行时零 LLM 依赖。

**Content**：
- 1.1 场景与动机：智能家居/嵌入式控制中 LLM 的延迟、成本、隐私、离线可用性问题；MCU
  资源约束（~520KB RAM）。
- 1.2 观察：LLM 的结构化 tool-call 轨迹本身是高质量"行为标签"——正样本、可在线累积。
- 1.3 主张与三个贡献：**C1 COMIC**（核心：CKMS+Welford+Split-Conformal+MDL+判别力
  筛选+留一剪枝的在线区间规则蒸馏）；**C2 PMS**（支撑：uint8 α/β 的 FTPL 规则选择，
  非平稳漂移场景）；**C3 HIL**（支撑：Trace-Driven 硬件在环 + 状态感知幂等执行）。
- 1.4 结果预览（数量化）：held-out AR（36 blocks）与手写规则/One-shot/ESP-Claw-style 无显著差异；UCI 合并回放
  精度 50.0±2.1% vs 批量 DT 16.5%；教师自一致天花板 80.0–96.7%，Ours fidelity
  efficiency 57–89%（合成）/ 40.5%（UCI）/ 81.9%（Steel）/ 38.4%（Air）；板载
  p50=1.48ms；同数据换教师 AR 差 1.7–3.7pp。

**Sources**：ECS 2604.15877（L3 空白）、HearthNet 2604.09618、EdgeTalk-MCU、
ClawTrace 2604.23853、RIMRULE、Skill-DisCo 2606.26669、WireClaw（工程源）。

**Transition**：空白成立 → Related Work 系统化铺开三支文献。

### 2. Related Work（~1,400 词）

**Purpose**：三支文献 + 一句话差异化；主动引用 WireClaw 而非回避。

**Content**：
- 2.1 LLM 在回路的边缘控制：HearthNet（多 Agent 编排、LLM 常驻）、EdgeTalk-MCU
  （本地 LLM+shield）、ESP-Claw（Lua+实时 LLM）、DCP（协议安全）、CIDER（sLLM 意图
  推理）。共同点：LLM 仍在回路。
- 2.2 trace → 技能/规则蒸馏：ClawTrace/CostCraft（patch 给云 Agent）、RIMRULE
  （MDL 规则注入 prompt，LLM 仍需运行）、Skill-DisCo（PFSM 子图编译给 Agent）、
  AgentReuse、Skill-as-Pseudocode、SkCC、Skill-SD。共同点：输出给"仍跑 LLM 的 Agent"。
- 2.3 规则挖掘与资源受限选择：HomeSGN、Kaufman & Hoffner（传感器→规则，非 LLM→规则）；
  Krentz（UCB on CC2538，"TS 对 MCU 太贵"）、Darak（FPGA TS）、QuBan、MINTS、
  Agrawal & Goyal、Kalai & Vempala（FTPL 谱系）；Split Conformal
  （Angelopoulos & Bates）、CKMS（Greenwald-Khanna）、Welford、MDL（Grünwald）。
- 2.4 定位声明（精确到词）：ECS 称 L3 declarative-rule 层 "largely empty"——本文提供
  其自动化实例；WireClaw 已做"LLM 生成规则 + ESP32 离线执行"，但为一次性 chat 配置、
  无 trace 蒸馏/增量学习/置信度校准/生命周期，且非同行评审。因此本文贡献限定为
  **peer-reviewed 空白**。

**Sources**：见 Evidence Map 文献列；引用修正五件套在此节落实。

**Transition**：文献缺口 → 系统设计（COMIC/PMS 的每个部件都有理论出处）。

### 3. System Design and Method（~2,000 词）

**Purpose**：给出一套可复现的完整管线；每个算法部件标注理论出处，诚实定位
"组合而非新发明"。

**Content**：
- 3.1 问题形式化与假设：五层架构（交互→轨迹→蒸馏→生命周期→执行）；正样本-only、
  在线、物理约束；MCU 只做采集/匹配/执行。
- 3.2 C1 COMIC 管线（核心，最重笔墨）：CKMS 在线分位数 → Welford 在线方差 →
  区间候选 → 判别力条件筛选（背景分布）→ Split Conformal 校准 → MDL 合并 →
  留一法有害条件剪枝；Wilson 置信度；逐规则置信区间。
- 3.3 五状态生命周期：candidate→verified→degraded→retired + freshness 指数衰减 +
  hysteresis + 状态感知幂等执行。
- 3.4 C2 PMS：uint8 α/β（2B/规则）、xorshift32、均值+均匀扰动；明确定位为 FTPL 变种，
  不声称 TS 的 regret bound；漂移验证机制。
- 3.5 MCU 映射与实现：ESP32-S3、FreeRTOS 双核、SPIFFS、cJSON 规则引擎、WebSocket
  遥测、USB 串口传感器注入（Trace-Driven HIL）。

**Sources**：C1 各标准件出处（conformal/CKMS/Welford/MDL/Wilson）；C2（Krentz/Darak/
QuBan/Kalai-Vempala）。

**Transition**：系统就绪 → 必须用公平协议评估（引出 held-out + teacher-replay）。

### 4. Experimental Setup（~1,100 词）

**Purpose**：评估协议本身是贡献——定义 held-out 时间切分与 teacher-replay 指标，堵住
审稿人所有"对比不公平/指标没定义"的质疑。

**Content**：
- 4.1 数据集：合成正弦（4 数据集 × 4 种子，生成器与 LLM 反馈解耦、同种子跨教师数据
  一致）；UCI Occupancy（真实物理传感器，CC BY 4.0，n=6 独立重复）；SML2010（室内
  气候）、Steel（工业能耗）、Air Quality（城市空气质量）各 4 种子——共 9 数据集 ×
  4 种子 = 36 held-out blocks；STRANDS Aruba-1（活动标注真实、传感器为合成补全）。
- 4.2 基线：Pure Cloud、Exact Cache、Sensor-Vector Cache（+MiniLM 语义缓存为
  supplementary）、User-defined、LLM One-shot、批量 DT、**在线每日重训 DT**（与 Ours
  同协议）、ESP-Claw-style。
- 4.3 评估协议：held-out 窗口（days 1–21 训练/预热，days 22–30 评估）；AR/CCR 定义；
  teacher-replay 60 快照 × 3 次 T=0（+T=0.7 参照）→ AGREE/P/R 定义（与 PCR Notes #6
  一致）；统计：Friedman+Nemenyi（36 blocks×8 / 9×8）+ bootstrap CI；UCI 480 快照 × 3 seed
  扩展回放稳定小样本精度对比。
- 4.4 硬件：ESP32-S3 板载闭环、200 条注入 0 丢失、esp_timer 实测匹配延迟。

**Sources**：UCI 数据集原始论文；统计方法标准引用。

**Transition**：协议立住 → 结果按"先覆盖率、再保真度、再硬件"顺序呈现。

### 5. Results（~2,000 词）

**Purpose**：数据全部从 JSON 文件引用，不出现任何文件外的数字。

**Content**：
- 5.1 自主率与云端调用削减（held-out 表）：诚实呈现 DT 家族更高，突出 Ours 与手写
  规则/One-shot 无显著差异；UCI n=6 的 32.4±19.2 及解释。
- 5.2 决策保真度（AGREE vs 教师自一致天花板）：fidelity efficiency 57–89%（合成）/
  40.5%（UCI）/ 81.9%（Steel）/ 38.4%（Air）；SML2010 比值 109.4 因分母口径不同不作
  硬上限；STRANDS 单独脚注（仅 AR 声明）。
- 5.3 Precision/Recall（**主打图**）：UCI 合并回放（480×3=1,440 快照）Ours P=50.0±2.1%
  vs 批量 DT 16.5% / 在线 DT 18.3% / ESP 22.3%（60 快照 100% 为 10/10 小样本假象）；
  SML2010/Steel Ours 精度最高；Air Quality Ours 落后（33.3% vs 83.3%/97.8%）——诚实报告。
- 5.4 跨教师泛化：同数据在线 AR 差 1.7–3.7pp；同 prompt 模型间一致 62–88%（strict）/
  83–98%（device）；规则迁移 25–47%。
- 5.5 消融：三源蒸馏、泛化（held-out +100pp）、生命周期（TTL AR=0 为 intended
  finding）、PMS 平稳/非平稳双表、Flash 写放大。
- 5.6 硬件：匹配延迟 p50=1.48ms、SRAM/PSRAM 遥测、板载 AR。

**Sources**：全部本地数据文件（见 Evidence Map）。

**Transition**：结果 → 讨论：这些数字在教师噪声包络下意味着什么。

### 6. Discussion（~1,000 词）

**Purpose**：解释三个"看起来不利"的数字并转为洞察。

**Content**：AR 非最高但无监督/在线/克制；AGREE 受教师自一致性（82–95%）上限约束；
DT 的 AR-精度脱钩（UCI）说明覆盖率指标误导；PMS 在非平稳才显价值；prompt 对跨模型
一致性的影响（94.4%→62–88% 的差异解释）。

**Transition**：讨论 → 局限清单。

### 7. Limitations（~400 词）

**Content**：USB 注入非物理采样；ground truth 来自教师（fidelity 非 correctness，
self-loop bias 显式声明）；STRANDS 传感器合成补全 + fidelity 退出声明；UCI 方差
32.4±19.2；消融单数据集、AGREE n=2；功耗/Flash 无硬件实测；板载高强度运行偶发重启。

### 8. Conclusion（~300 词）

**Content**：复述一句话论点 + 三贡献 + 未来工作（加权 specificity 冲突解析、真实
传感器长周期、多设备）。

---

## Evidence Map（数字 ↔ 文件 ↔ 主张，写作时唯一引用源）

| 论文表/图 | 数据文件 | 主张 |
|---|---|---|
| Table: held-out AR/CCR（8 方法 × 36 blocks） | `poc/output/statistics_4x_baselines.json` | 自主率排名与显著性 |
| Table: held-out AR/CCR 的公平性对照 | `poc/output/statistics_4x_baselines.json` | 全基线同窗口 held-out（`baseline_split_compare.json` 为早期产物，已标注 deprecated，不再引用） |
| Table: AGREE / P / R（60 快照 × 9 数据集） | `poc/output/teacher_replay_results.json` | 保真度、精度、UCI 反例 |
| Table: UCI 扩展 P/R（480×3） | `poc/output/uci_extended_replay_results_multiseed.json` | n=13 质疑的修复 |
| Figure: AGREE vs 天花板 + CI | `poc/output/agree_reference.json` | fidelity efficiency 57–89%/40.5% |
| Figure: 跨教师 | `poc/output/cross_model_same_data.json` + `rule_transfer.json` | 同数据 AR 差、规则迁移 |
| Table: 语义缓存扫参 | `poc/output/semantic_sweep_results.json` | MiniLM vs 4 维向量缓存 |
| Table: 消融 | `poc/output/ablation_results_4x.json` + `pms_regret.json` | 泛化/生命周期/PMS/Flash |
| Table: UCI n=6 CI | `poc/output/statistics_4x.json` | 32.4±19.2，CI[18.2,46.5] |
| Table: 板载硬件 | `poc/output/mcu_metrics.json` | p50=1.48ms、SRAM/PSRAM、AR |
| Table: 教师自一致性 | `poc/output/llm_consistency_results.json` | 天花板 82–95%（2160 次调用） |
| Figures | `figures/fig_*.pdf`（11 张） | 架构/生命周期/学习曲线/基线/Nemenyi/P-R/跨LLM/消融/延迟/oracle |

### 19,234 次真实云端调用口径（可追溯分解）

- 在线 held-out 运行 **6,226**（36 个 held-out 块对应的 `run4b_*` 目录，
  另加两个 UCI 方差重复目录 `run4b_uci_seed2026/31415`，共 38 个目录，
  trace `execution.mode == "cloud"` 计数；36 块与方差重复合计 5,414+812）。
- 同数据跨教师运行 **806**（`xrun_ds_*` + `xrun_qwen_*` + `qwen_*` 目录）。
- teacher-replay **3,240**（9 数据集 × 60 快照 × 6 次调用；旧 6 数据集
  `llm_consistency_results.json` 2,160 + 新 3 数据集各 360）。
- UCI 扩展回放 **8,640**（3 seed × 480 快照 × 6 次调用，`uci_480_*`）。
- 脚本化跨 LLM 配对 **322**（26+35+100 快照 × 2 模型，见 `cross_llm_*.json`）。

合计 6,226 + 806 + 3,240 + 8,640 + 322 = 19,234。

### 文献清单（Related Work 引用池，均已核实）

- LLM 在回路：HearthNet（2604.09618）、EdgeTalk-MCU、ESP-Claw、DCP（2605.26159）、CIDER
- trace→技能：ClawTrace/CostCraft（2604.23853）、RIMRULE（ACL 2026 long.1599）、
  Skill-DisCo（2606.26669）、AgentReuse（2512.21309）、Skill-as-Pseudocode
  （2605.27955）、SkCC（2605.03353）、Skill-SD（2604.10674）
- 规则挖掘：HomeSGN（ASP-DAC 2024）、Kaufman & Hoffner（IoTBDS 2024）
- 选择/bandit：Krentz（CC2538）、Darak（FPGA）、QuBan（AISTATS 2022）、MINTS
  （NeurIPS 2025）、Agrawal & Goyal（JACM 2017）、Kalai & Vempala（2005）、
  Gopalan（ICML 2014）、Zheng（2024）、Skorski（2021）、Yamin & Bhat（TCAD 2024）
- 方法标准件：Split Conformal（Angelopoulos & Bates）、CKMS（Greenwald & Khanna）、
  Welford（1962）、MDL（Grünwald）、Wilson score
- 框架/工程：ECS（2604.15877）、WireClaw（GitHub，非同行评审）
- 数据：UCI Occupancy Detection（CC BY 4.0）、SML2010（UCI 274）、Steel（UCI 851）、
  Air Quality（UCI 360）、STRANDS Aruba-1

---

**质量门自检**：结构=IMRaD（系统变体）✓；每节有 Purpose ✓；词数合计 ~9,100（±5% 内）✓；
全部文献源已映射 ✓；相邻节 Transition 已写 ✓；待用户批准后进入 Phase 3。
