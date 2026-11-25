/*
 * DistillToMCU Firmware — Master Config Header
 * ===========================================
 * ESP32-S3 + FreeRTOS + LittleFS
 * All compile-time constants centralized here.
 */

#ifndef MIMI_CONFIG_H
#define MIMI_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

/* ===== WiFi ===== */
#define WIFI_SSID           CONFIG_MIMI_WIFI_SSID
#define WIFI_PASS           CONFIG_MIMI_WIFI_PASS
#define WIFI_CONNECT_TIMEOUT_MS  30000
#define WIFI_RETRY_MAX      5

/* ===== DeepSeek API ===== */
#define LLM_API_KEY         CONFIG_MIMI_API_KEY
#define LLM_BASE_URL        "https://api.deepseek.com"
#define LLM_MODEL           "deepseek-v4-flash"
#define LLM_MAX_TOKENS      1024
#define LLM_HTTP_TIMEOUT_MS 15000
#define LLM_TEMPERATURE      0.0f

/* ===== Memory Layout ===== */
#define PSRAM_JSON_BUF_SIZE      (32 * 1024)   /* LLM response buffer */
#define PSRAM_SYSPROMPT_BUF_SIZE (16 * 1024)   /* system prompt buffer */
#define PSRAM_TOOL_OUTPUT_SIZE   (8 * 1024)    /* tool output buffer */
#define RAM_TRACE_RINGBUF_SIZE   (16 * 1024)   /* trace ring buffer in PSRAM */
#define RAM_RULE_JSON_MAX        (4 * 1024)    /* max single rule JSON size */

/* ===== Trace Logging ===== */
#define TRACE_FLUSH_INTERVAL_MS  60000         /* ringbuffer flush interval */
#define TRACE_FLUSH_COUNT        15            /* flush after N traces */
#define TRACE_MAX_FILE_SIZE      (512 * 1024)  /* rotate trace file */

/* ===== Rule Engine ===== */
#define RULE_MAX_TOTAL           500           /* max rules in Flash */
#define RULE_CONFIDENCE_LOCAL    0.80f         /* confidence > this → local */
#define RULE_CONFIDENCE_ASYNC    0.50f         /* 0.5–0.8 → local + async LLM */
#define RULE_EVIDENCE_MIN        3             /* min evidence → candidate→verified */
#define RULE_DECAY_TAU_BASE      30.0f         /* freshness decay tau (days), base */
#define RULE_RETIRE_AFTER_DAYS   14            /* degraded N days → retired */
#define RULE_FRESHNESS_DEGRADE   0.20f
#define RULE_MUTEX_COOLDOWN_MS   5000          /* actuator cooldown */
#define RULE_HYSTERESIS_DELTA    0.15f

/* ===== FreeRTOS ===== */
#define TASK_STACK_AGENT          (24 * 1024)
#define TASK_STACK_OUTBOUND       (8 * 1024)
#define TASK_STACK_CLI            (4 * 1024)
#define TASK_STACK_CONFIRM        (8 * 1024)   /* async LLM confirm worker (HTTP+TLS) */
#define TASK_PRIO_AGENT           6
#define TASK_PRIO_CONFIRM         (tskIDLE_PRIORITY + 1)

/* ===== GPIO Pin Map (面包板) ===== */
#define PIN_LED_R      4    /* GPIO4  → 黄色 LED (灯) */
#define PIN_LED_G      5    /* GPIO5  → 蓝色 LED (风扇) */
#define PIN_LED_B      6    /* GPIO6  → 绿色 LED (窗帘) */
#define PIN_BUTTON_ACCEPT  13   /* 接受按钮 */
#define PIN_BUTTON_CORRECT 14   /* 纠正按钮 */

#ifdef CONFIG_MIMI_LED_WS2812_ENABLE
#define PIN_LED_WS2812      CONFIG_MIMI_LED_WS2812_GPIO  /* 默认 48 = DevKitC-1 板载 RGB */
#endif

/* ===== 传感器注入超时 =====
 * serial_injector.py 默认 5s 一条；超时必须明显大于注入间隔，
 * 否则 5s 边界抖动会丢失注入数据（v6 修复：5s → 15s）。 */
#define SENSOR_INJECT_TIMEOUT_US (15 * 1000 * 1000)

/* ===== Safety Levels ===== */
#define SAFETY_L0  0  /* 查询/只读, auto-learn */
#define SAFETY_L1  1  /* 舒适类 (灯/风扇), auto-learn */
#define SAFETY_L2  2  /* 高能耗, needs first confirm */
#define SAFETY_L3  3  /* 安全关键, NEVER auto-local */

/* ===== Storage (SPIFFS VFS, flat paths) ===== */
#define LFS_BASE_PATH      "/littlefs"
#define LFS_TRACE_FILE     LFS_BASE_PATH "/trace_001.jsonl"
#define LFS_RULES_FILE     LFS_BASE_PATH "/rules.json"
/* SPIFFS doesn't support real directories, keep paths flat */

#endif
