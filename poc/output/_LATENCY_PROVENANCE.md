# 云端延迟口径说明（Cloud latency provenance，2026-08-19 更新）

论文正文的“约 4.8 秒云端决策延迟”来自全部在线 held-out 运行与同数据
跨教师运行 trace 中记录的 `llm_response.latency_ms`，n = 7,032，
均值 4.81 s（中位 3.92 s）。这是 §5.6“比实测云端往返低三个数量级”的对照值。

分数据集抽样（run4b_*_seed42 的 cloud traces）：
SML2010 均值 4.25 s、Steel 均值 4.68 s、Air Quality 均值 4.85 s。

`baseline_results_4x.json` 里的 `avg_cloud_latency_ms`（约 1,071–2,362 ms，
Pure Cloud 各块约 1,650–1,700 ms）是早期基线汇总的遗留/模拟值，与
`run4b_*/metrics.jsonl` 不一致，**不要用该字段引用云端延迟**。
论文的 AR/CCR 等主表数字不依赖该文件，引用云端延迟一律以 trace 为准。

## 细分复核（2026-08-19 独立重算，与正文 4.8 s / 3.9 s 一致）

聚合结果同时归档于 `cloud_latency_summary.json`（含各子集 n/均值/中位与合并
n=7,032 的 4.808 s / 3.924 s），审稿人可直接打开该文件核对，无需重算。

从 `run4b_*/traces.jsonl`、`xrun_*/traces.jsonl`、`qwen_*/traces.jsonl` 的
`llm_response.latency_ms` 直接统计（仅统计真实模型调用的条目，排除
`local` / `local_rule_engine` 本地执行条目）：

| 子集 | 条目数 | 均值 | 中位数 |
|---|---|---|---|
| 在线 held-out（run4b_*，deepseek-v4-flash） | 6,226 | 4.27 s | 3.82 s |
| 同数据跨教师（xrun_ds 109 + xrun_qwen 78 + qwen 完整运行 619） | 806 | 8.97 s | 9.18 s |
| **合并（论文 n = 7,032）** | **7,032** | **4.808 s** | **3.924 s** |
| 独立对照：脚本化跨模型配对（cross_llm_*） | 322 | 4.816 s | 4.202 s |

注意：`run4b_*/metrics.jsonl` 的 `avg_cloud_lat_ms`（每日约 4.5–5 s）与上述
trace 统计一致；`baseline_results_4x.json` 的 `avg_cloud_latency_ms`
（约 1.07–2.36 s）来自早期基线汇总的遗留/模拟值，与真实 trace 矛盾，勿用。
