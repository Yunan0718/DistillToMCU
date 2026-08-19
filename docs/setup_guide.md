# 环境搭建（与方案表一致）

## 系统

- Windows 10/11、Ubuntu 20.04+ 或 macOS 12+ 任选。

## ESP-IDF

- **版本**：方案要求 **v5.2.1**。
- Windows 安装器：<https://dl.espressif.com/dl/esp-idf/idf-tool-setup-idf-v5.2.1.exe>
- Linux/macOS：`git clone -b v5.2.1 --recursive https://github.com/espressif/esp-idf.git`

首次进入工程目录：

```bash
idf.py set-target esp32s3
idf.py menuconfig
```

建议在 menuconfig 中确认：**Flash 16MB**、**PSRAM 8MB**、WiFi 已启用（与 N16R8 硬件一致）。

## 密钥配置

1. 复制 `main/mimi_secrets.h.example` → `main/mimi_secrets.h`
2. 填写 WiFi、以及所选 LLM 的 **API Key**（Telegram Bot Token 可选）
3. 可选：在 `main/ai_config.h` 中调整 `AI_MAX_HISTORY`、`AI_MAX_HEIGHT_M`、`AI_MAX_DISTANCE_M`

修改 `mimi_secrets.h` 后需：`idf.py fullclean && idf.py build`

### DeepSeek（OpenAI 兼容）

本工程已在 `llm_proxy` 中支持 **`deepseek`** 提供商，请求发往 `https://api.deepseek.com/v1/chat/completions`。

在 `mimi_secrets.h` 中设置示例：

- `MIMI_SECRET_MODEL_PROVIDER` → `"deepseek"`
- `MIMI_SECRET_API_KEY` → [DeepSeek 开放平台](https://platform.deepseek.com/) 申请的 Key  
- `MIMI_SECRET_MODEL` → `deepseek-chat` 或 `deepseek-reasoner`（见官方文档）

烧录后也可用串口：`mimi> set_model_provider deepseek`、`mimi> set_api_key sk-...`、`mimi> set_model deepseek-chat`。

代理：若使用 HTTP 代理访问 API，仍通过既有的 `MIMI_SECRET_PROXY_*` 配置。

## WebSocket 前端（推荐）

1. 固件默认提供 WebSocket 网关（端口 `18789`）  
2. 先用你自己的前端（网页/桌面）直连局域网内设备，完成消息收发闭环  
3. Telegram 作为可选备用通道，不再作为主线必需项

## Telegram（可选）

1. @BotFather 创建 Bot，获取 Token（仅在需要 Telegram 通道时）  
2. Chat ID：若需白名单，见后续固件扩展；默认任意用户可与 Bot 对话  
3. 串口 CLI：`mimi> set_tg_token ...` 可写入 NVS，免重编译  

## 工程结构说明

- 入口：`main/app_main.c` → `mimi_start()`（`main/mimi.c`）
- 方案扩展：`main/ai_command.c`、`main/flight_tools.c`、工具注册在 `main/tools/tool_registry.c`
- 上游 MimiClaw 源码在 `main/` 各子目录；`components/mimiclaw/` 为预留说明（见该目录 README）

## 分区表

当前仓库使用上游 **OTA + SPIFFS** 布局（`partitions.csv`），与方案表中「简化四分区示例」不同；若需完全对齐简表，需自行替换分区并验证 SPIFFS 偏移。

## 编译与烧录

```bash
idf.py build
idf.py -p COMx flash monitor
```

波特率 115200。双 USB 口开发板请参考上游 README 区分 JTAG 与 UART。

## 引脚与 PCB

千问/早期方案表中的 **GPIO 分配与 PCB 示意仅作参考**。正式画板时请按模组数据手册、天线、电源与走线重新分配，并同步修改 `main/board_pins.h` 与 `main/tools/gpio_policy.h`（白名单）。
