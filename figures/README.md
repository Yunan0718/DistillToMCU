# 论文级图表（v10.6）

`figures/gen_all.py` 按 IEEE/Elsevier 期刊规范生成数据图表，输出 **PDF（矢量）+ PNG（300 DPI）**。
H5 顶栏「论文图」按钮一键生成；也可命令行运行：

```
C:\Espressif\tools\python\v5.2.6\venv\Scripts\python.exe figures\gen_all.py
```

## 图表清单

| 文件 | 内容 | 对应论文指标 |
|------|------|-------------|
| fig_architecture | 系统架构图（数据源 → PC 蒸馏 → MCU 执行 / 云端回退闭环） | Figure 1 |
| fig_rule_lifecycle | 规则生命周期状态机（candidate→…→retired + 阈值） | 生命周期 |
| fig_ar_learning | 自主率增长曲线：(a) 4 个合成数据集 × 4 重复均值±Bootstrap CI（v10.5），(b) STRANDS/UCI V3（4x 重复） | Autonomy Rate |
| fig_baselines | 7 基线（含在线每日重训 DT）+ Ours 的 AR 与 CCR（24 blocks，held-out 窗口 days 22–30，v10.6） | AR, CCR |
| fig_latency | 本地 vs 云端延迟分布 + 各数据集 p50 | Latency local/cloud |
| fig_precision_recall | 各数据集 teacher-replay Precision/Recall（60 快照/数据集，教师多数决策为 ground truth；替代 Phase 0 占位） | Precision, Recall |
| fig_rules_size | 规则库规模随天增长 | Rule Store Size |
| fig_nemenyi_cd | Friedman + Nemenyi CD 图（**8 方法：7 基线 + Ours，24 blocks，held-out 窗口**） | 统计显著性 |
| fig_cross_llm | (a) DeepSeek vs Qwen 配对决策一致性；(b) 同数据在线 AR（教师互换）；(c) 蒸馏规则 fidelity/transfer | Cross-LLM 验证 |
| fig_ablations | 2×2 消融：蒸馏来源 / 泛化(含 held-out) / 生命周期 / PMS bandit | 消融 |
| fig_chat_stats | 对话实验统计（有对话实验时生成） | 对话实验 |

## 严谨性与数据口径（v7.3 审查后）

1. **统计检验**：CD 图把 DistillToMCU 一起纳入 Friedman + Nemenyi
   （8 方法 × 24 blocks（6 数据集 × 4 种子），α=0.05），并画出非显著配对连接线；
   rank 方向已修正（higher-is-better 指标）；
   Friedman p 值直接标在图上。
2. **held-out 协议（v10.6）**：所有方法统一在 days 22–30 评估（days 1–21
   仅训练/预热）；批量基线（DT/One-shot）只在前 70% 天训练；在线基线
   （Online-DT/ESP-Claw/Ours）只用过去信息增量学习。此前"train 前 70%、
   evaluate 全部"混入了训练数据。
3. **置信区间**：4 个合成数据集 × 4 重复的 AR 均值带用 Bootstrap 95% CI；
   UCI 追加 2 次独立重复（n=6）。
4. **延迟 provenance**：云端延迟为真实 DeepSeek API；本地规则匹配延迟为板载
   esp_timer 实测（p50=1.48ms，mcu_metrics.json）；本地执行延迟仍为 Phase 0 仿真值。
5. **Precision/Recall（v10.6）**：改为 teacher-replay 定义——
   precision = P(规则动作==教师 | 系统本地执行)，recall = P(系统本地执行且一致 |
   教师有动作)，ground truth 为教师 3 次 T=0 重复的多数决策，60 快照/数据集。
   无真实用户 veto 反馈的限制在 Limitations 中声明。
6. **未作图的数据（避免编造）**：功耗 / Flash 擦写无硬件实测 → supplementary
   或 future work（消融 5 为解析估算，已明确标注）。

## 期刊规范（依据 IEEE/Elsevier 官方指南）

- 字体：Times New Roman（回退 DejaVu Serif），标题 10pt、正文 8-9pt；
- 分辨率：PNG 300 DPI；提交时用 **PDF 矢量**（Elsevier 线稿 1000 DPI / 组合 500 DPI 要求由 PDF 天然满足）；
- 配色：Okabe-Ito 色盲安全色；**Ours = 朱红 #D55E00** 高亮；
- 风格：无顶/右边框、浅网格、误差棒/置信带、直接值标签；
- 尺寸：单栏 3.35in / 双栏 6.9in（IEEE/Elsevier 双栏 88mm/183mm 兼容）。

## 后续补图建议（有真实数据后）

- **Flash 擦写次数 / 功耗**：需要功率计与擦写计数遥测，暂以 future work 处理；
- **真实用户 veto 数据**：板载按钮路径已接线，但尚无真实交互数据。
