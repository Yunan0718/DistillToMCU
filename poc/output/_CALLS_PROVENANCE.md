# API 调用计数溯源（19,234 次，2026-08-19 复核）

论文 §4.4 的 "19,234 real cloud calls" 由五个互不重叠的组件构成，
每个组件都能在输出目录中按文件级证据数出：

| 组件 | 计数 | 文件级证据 |
|---|---|---|
| 在线 held-out（36 块 + 2 次 UCI 方差重复） | 6,226 | `run4b_*/traces.jsonl` 中 `llm_response.model == "deepseek-v4-flash"` 的条目数（38 个目录合计） |
| 同数据跨教师 | 806 | `xrun_ds_*`（109）+ `xrun_qwen_*`（78）+ `qwen_*`（619）traces 中真实模型调用条目数 |
| teacher replay（9 数据集 × 60 快照 × 6 次查询） | 3,240 | `llm_consistency_results_*_60_*.json` 的 `summary.total_api_calls`，9 个文件各 360 |
| 扩展 UCI 回放（3 种子 × 480 快照 × 6 次查询） | 8,640 | `llm_consistency_results_uci_480_seed{42,123,999}.json` 各 2,880 |
| 脚本化跨模型配对 | 322 | `cross_llm_*_qwen3.7-flash-2026-07-15.json` 的 `details[].latency_ms` 条目（2 次调用/快照 × 161 快照） |
| **合计** | **19,234** | — |

## 复核命令（等价口径）

- 在线：对每个 `run4b_*/traces.jsonl`，统计 `llm_response.model == "deepseek-v4-flash"` 且
  `llm_response.latency_ms` 非空的条目 → 合计 6,226。
- 跨教师：同上口径统计 `xrun_ds_*`、`xrun_qwen_*`、`qwen_*` → 合计 806。
- 一致性文件：读取各 `llm_consistency_results_*.json` 的 `summary.total_api_calls` 相加。
- 脚本化配对：读取 `cross_llm_*.json` 的 `details` 中 `latency_ms` 条目数相加 → 322。

注意：在线与跨教师组件只统计**真实模型调用**，不含 `local` / `local_rule_engine`
本地执行条目（这些在 traces 中同样带毫秒级 latency，但属于本地匹配，不是云端调用）。
