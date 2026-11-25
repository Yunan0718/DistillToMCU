# 实验服务与 H5 新建实验（v7.0）

## 一、这是什么

H5 仪表盘从"只能看已跑完的数据"升级为"可以直接新建实验"：

- **普通实验**：选择内置数据源（合成 / UCI / CASAS）或上传任意 CSV/JSON/Excel，
  AI 辅助列映射后，一键在 PC 上跑真实 LLM 实验，结果自动进下拉列表。
- **MCU 真机实验**：同一套数据源，由电脑按节奏分块喂给 ESP32 板子，
  板子真实执行（规则匹配 → 云端 LLM → GPIO/LED），执行记录回传电脑算指标出图。
- **对话实验**：AI 对话会话自动保存为实验条目，带统计（消息/动作/本地云端/延迟/自主率），
  与普通实验在下拉列表中分组显示。
- **论文图**：一键生成 IEEE/Elsevier 期刊规范的矢量图（PDF+PNG 300dpi），见 `figures/README.md`。

## 二、怎么用

### 1. 启动服务

双击项目根目录的 **`启动实验服务.bat`**（会从 `sdkconfig.user` 自动读取 DeepSeek Key）。
看到 `DistillToMCU 实验服务已启动: http://127.0.0.1:18800` 即成功。
服务只监听本机；关闭窗口即停止。

### 2. 打开仪表盘

打开 `poc/dashboard.html`，顶栏出现 `● 实验服务在线` 徽章。

### 3. 新建实验

顶栏点 **`＋ 新建实验`**：

1. 填实验名称，选数据源（内置），或点 **上传数据文件** 上传 CSV/JSON/XLSX；
2. 上传后出现列映射表：点 **🤖 AI 智能识别** 自动把列映射到
   temperature / humidity / light / co2 / motion / temp_trend / hour / user_input，
   可手动改，然后 **确认映射并作为数据源**；
3. 选运行位置：
   - **PC 模拟**：本机跑完整 Python 实验（真实 DeepSeek）；
   - **MCU 真机**：填板子 IP/端口和喂送间隔（默认 2.5s，真实 LLM 决策约 5-6s/条，
     建议 ≥2s 防止队列堆积）；板子上的面包板 LED 会真实亮灭；
4. 点 **开始实验**，弹窗内实时显示进度和日志；完成后自动加入下拉列表并切换过去。

### 4. 查看/回放

- 下拉列表分两组：**普通实验**、**对话实验**；
- 对话实验点开自动跳到 AI 对话页，显示完整会话与统计条；
- 顶部新增 **论文图** Tab：页面上直接预览全部论文级图表（卡片网格 + 点击放大灯箱），
  每张图带说明和 **PDF 矢量 / PNG 300dpi** 下载链接，也可在页面上直接点“重新生成全部图”。
  图表文件同时保存在 `figures/`。

## 三、架构

```
H5 (poc/dashboard.html, file://)
   │  fetch (CORS *)
   ▼
tools/experiment_server.py  (127.0.0.1:18800, 纯标准库)
   ├── /api/datasets       内置+上传数据源、AI 列映射（DeepSeek）
   ├── /api/experiments    创建/查询/停止实验（PC 或 MCU）
   ├── /api/chat-experiments  对话实验自动保存+统计
   └── /api/figures/export 调用 figures/gen_all.py 出论文图
          │
          ├── PC 模式: subprocess → poc/experiment*.py（真实 LLM）
          └── MCU 模式: WebSocket → ESP32 (192.168.2.6:18789)
                喂 sensor 快照 → 板子规则匹配/LLM/GPIO
                → trace_read 回传 → PC 算 metrics.jsonl
```

## 四、关键设计

- **MCU 存不下大数据源**：电脑分块喂送。固件 v7 增加注入 FIFO 队列
  （`sensor.c`，最多 8 条），agent loop 每条处理一次，不丢快照。
- **行数控制**：MCU/自定义实验自动均匀抽样，≤ 天数×20 条，避免跑十几个小时。
- **真实 LLM**：按用户要求不做离线 mock；PC 与 MCU 实验都调用真实 DeepSeek。
- **统计口径**：PC 与 MCU 统一输出 `metrics.jsonl`
  （autonomy_rate / cloud_calls / local_calls / 延迟 / 规则数），对话实验统计参考普通实验。

## 五、API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/health | 服务状态 + 是否有所需 API Key |
| GET | /api/datasets | 数据源列表（含已上传） |
| POST | /api/datasets/upload | 上传文件（base64）→ 返回列与样例 |
| POST | /api/datasets/ai-map | DeepSeek 辅助列映射（失败回退启发式） |
| POST | /api/datasets/map | 确认映射并生成标准快照数据源 |
| POST | /api/experiments | 新建并启动实验（target: pc/mcu） |
| GET | /api/experiments | 全部实验 + 运行中日志/进度 |
| GET | /api/experiments/<id> | 单个实验 metrics/rules/traces |
| POST | /api/experiments/stop | 停止 MCU 喂送 |
| POST | /api/chat-experiments | 保存对话实验（upsert + 统计） |
| GET | /api/chat-experiments | 对话实验列表 |
| POST | /api/figures/export | 生成论文图 |

## 六、注意事项

- 上传文件只保存在本机 `poc/data/uploads/<id>/`；
- 真实 LLM 实验会消耗 API 额度（每 300 次交互约 ¥0.2）；
- MCU 实验期间请勿在 AI 对话里发消息（会与喂送竞争执行队列）；
- 串口 CLI 的中文仍受终端编码限制，中文对话请用 H5。
