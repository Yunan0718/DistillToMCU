# Paper Configuration Record — DistillToMCU

> academic-paper skill · Phase 0 · 待用户确认（IRON RULE：确认前不进入正式写作）

| Parameter | Value |
|-----------|-------|
| **Topic** | DistillToMCU: 从云端 LLM 控制行为轨迹蒸馏置信校准的区间规则，在 ESP32-S3 上零 LLM 依赖本地执行 |
| **Research Question** | 能否仅通过观察云端 LLM 的结构化控制决策（tool-call 轨迹），自动蒸馏出 MCU 可执行、带置信度与生命周期的符号化规则，并在无任何监督信号下达到与手写规则相当的自主率、落在教师自一致性包络内的保真度、以及显著高于监督基线的真实数据精度？ |
| **Paper Type** | IMRaD（系统+实验论文） |
| **Discipline** | IoT / Embedded Systems / LLM Agents（CS，交叉） |
| **Target Journal** | Elsevier Internet of Things（中科院 2 区，IF 7.1）；备选 IEEE IoT-J / IEEE Sensors J |
| **Citation Format** | IEEE |
| **Output Format** | Markdown（草稿）→ LaTeX + PDF（终稿） |
| **Body Language** | English（讨论用中文，论文只出英文） |
| **Abstract** | English only（投英文期刊；中文摘要仅供自用，不提交） |
| **Word Count Target** | 正文 ~9,000 词（不含参考文献），Elsevier IoT 期刊体量 |
| **Existing Materials** | 全部实验数据（poc/output/*.json）、11 张论文级图表（figures/）、已核实竞品文献清单、固件（main/）、CLAUDE.md 贡献框架 |
| **Co-Authors** | 独作（Solo author）——Acknowledgments 需具体感谢讨论者以对冲 |
| **Funding** | no funding（需显式声明） |
| **Style Profile** | null（无历史样稿；严格按 IEEE/Elsevier 风格） |
| **Domain Evidence Profile** | cs_ml（承认 arXiv 预印本与会议录；本领域 2026 竞争论文多为 arXiv/workshop/ACM 会议） |
| **Citation Verification** | advisory（mark only）——存在 GitHub 工程源、arXiv 预印本等未索引文献 |
| **Operational Mode** | outline-only（本轮产出大纲 + 证据地图，待确认后转 full） |

## Notes（叙事与口径锁定，写作全程不得违反）

1. **主打不是 AR**。held-out AR 中 DT 家族（99.4/99.6）高于 Ours（90.2±21.6）。主叙事 =
   无监督在线增量学习 + 真实数据上的高精度克制（UCI：Ours P=50% vs 批量 DT 11.7%）+
   2B/规则 PMS + 1.48ms 板载实测 + 零 LLM 依赖。
2. **STRANDS 只用于 AR/覆盖率声明**，退出 fidelity 叙事：warm 规则 60/60 命中但 0/43
   动作一致，根因为合成传感器补全下 COMIC 区间过宽、多规则重叠（写入 Limitations）。
3. **PMS 平稳+非平稳双表同报**：平稳 Greedy 407.0 略优于 PMS 414.6；非平稳 PMS 937.0
   ≈ ExactTS 939.1 且优于 Greedy 943.3 / ε-greedy 962.8。论证非平稳（freshness/漂移）
   才是部署现实。
4. **消融标注口径**：单数据集（run4b_seed42_seed42）、held-out 协议、AGREE n=2——
   论文表格脚注必须写明，不得让审稿人误以为大样本。
5. **Flash 消融只声称"写放大降低"**（17→1.1 次/天），不声称寿命提升（128.9 vs 120.9 年
   是估算且未变好）。
6. **P/R 与 AGREE 定义**（Methods 必须原样给出）：60 快照/数据集 × 教师 3 次 T=0 重复，
   ground truth = 多数决策；precision = P(规则==教师 | 本地执行)；recall =
   P(本地执行且一致 | 教师有动作)；AGREE 同 recall 定义；Pure Cloud=0（无本地策略）、
   Exact Cache=100（精确回放教师决策）为构造性边界。
7. **引用修正五件套**：WireClaw 主动引用并差异化（非同行评审、一次性配置、无 trace
   蒸馏/置信度/生命周期）；AFD-KD venue=ACM；RIMRULE=7 作者合著；CIDER 完整标题
   "On-Device Intent Reasoning ... Ontology-Augmented sLLMs"；ECS 措辞为
   "L3 declarative-rule level largely empty → we instantiate an automated L3 instance"。
8. **诚实声明**：传感器经 USB 串口注入（Trace-Driven HIL），非物理采样；AGREE/P 的
   ground truth 来自教师 LLM（fidelity 而非 correctness）；功耗/Flash 擦写无硬件实测。

---

**下一步（等待确认）**：用户确认本记录 → 进入 Phase 3-4（论证蓝图 + 全稿）。
大纲与证据地图见 `paper/02_outline_and_evidence_map.md`。
