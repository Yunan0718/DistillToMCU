/*
 * DistillToMCU �?Firmware Entry Point (v2: real agent loop)
 * ==========================================================
 * ESP32-S3 + FreeRTOS + SPIFFS (VFS)
 *
 * Architecture:
 *   Core 1: Agent Loop (sensor �?match �?execute / cloud LLM)
 *   Core 0: ESP-IDF system tasks (WiFi, UART, etc.)
 *
 * USB Serial: sensor data injection from PC datasets (UCI/CASAS)
 *             via 'sensor <json>' CLI command or automatic PC script.
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_system.h"
#include "esp_spiffs.h"
#include "nvs_flash.h"

#include "mimi_config.h"
#include "wifi_mgr.h"
#include "http_client.h"
#include "sensor.h"
#include "actuator.h"
#include "trace_logger.h"
#include "rule_store.h"
#include "rule_engine.h"
#include "executor.h"
#include "serial_cli.h"
#include "llm_confirm_worker.h"
#include "ws_server.h"

static const char *TAG = "d2mcu";

/* ---- SPIFFS mount ---- */
static esp_err_t init_spiffs(void) {
    esp_vfs_spiffs_conf_t c = {
        .base_path = "/littlefs", .partition_label = "littlefs",
        .max_files = 5, .format_if_mount_failed = true,
    };
    esp_err_t r = esp_vfs_spiffs_register(&c);
    if (r != ESP_OK) { ESP_LOGE(TAG, "SPIFFS fail: %s", esp_err_to_name(r)); return r; }
    size_t t = 0, u = 0;
    esp_spiffs_info("littlefs", &t, &u);
    ESP_LOGI(TAG, "SPIFFS: %d KB total, %d KB used", t / 1024, u / 1024);
    return ESP_OK;
}

/*
 * Agent Loop (Core 1)
 * ====================
 * The main execution cycle:
 *   1. Check if sensor data is available (USB injected or mock default)
 *   2. Run rule matching �?local execution
 *   3. If no match �?cloud LLM fallback
 *   4. Periodic rule lifecycle maintenance
 *
 * This runs continuously on Core 1. The serial CLI REPL runs on
 * the ESP-IDF default core, allowing interactive commands alongside
 * the agent loop.
 */
static TaskHandle_t s_agent_handle = NULL;
TaskHandle_t agent_task_handle(void) { return s_agent_handle; }

static void agent_loop_task(void *pv) {
    ESP_LOGI(TAG, "Agent loop started on Core %d", xPortGetCoreID());

    uint32_t cycle = 0;
    const TickType_t interval = pdMS_TO_TICKS(2000);  /* 2s between cycles */

    while (1) {
        cycle++;

        /* Only run the loop when REAL sensor data is injected.
           Mock values would call the cloud LLM every cycle for no reason
           (burning API budget) and fight manual GPIO/LED testing.
           Phase 2 will enable autonomous polling. */
        if (sensor_has_injection()) {
            cJSON *result = executor_handle("sensor_injected");
            if (result) cJSON_Delete(result);
            /* v6 fix: 每个注入快照只决策一次，避免 2s 循环重复执行
               （重复执行会反复写 flash / 重复调 LLM）。
               v7: 注入队列 FIFO，弹出队首，PC 端喂送不丢快照。 */
            sensor_pop_injection();
        }

        /* v10.5f (D5): resolve pending user-override feedback window */
        executor_feedback_poll();

        /* Periodic: persist rules */
        if (cycle % 90 == 0) {  /* every ~3 min */
            rule_store_persist();
            ESP_LOGD(TAG, "Rules persisted");
        }

        /* v8: Periodic freshness update (every ~30 min = 900 cycles) */
        if (cycle % 900 == 0) {
            rule_engine_update_freshness();
        }

        /* v9: Online rule interval update (every ~6 min = 180 cycles) */
        if (cycle % 180 == 0) {
            rule_engine_online_update_intervals();
        }

        vTaskDelay(interval);
    }
}


void app_main(void) {
    ESP_LOGI(TAG, "DistillToMCU v0.2.0 | ESP32-S3 + FreeRTOS + SPIFFS");
    ESP_LOGI(TAG, "Phase 1: Real LLM + USB Serial Sensor Injection");

    /* Boot sequence */
    esp_err_t r = nvs_flash_init();
    if (r == ESP_ERR_NVS_NO_FREE_PAGES || r == ESP_ERR_NVS_NEW_VERSION_FOUND)
        { nvs_flash_erase(); nvs_flash_init(); }
    /* esp_event_loop_create_default() is called inside wifi_mgr_init() */
    ESP_ERROR_CHECK(init_spiffs());
    ESP_ERROR_CHECK(sensor_init());
    ESP_ERROR_CHECK(actuator_init());
    /* v6：上电 LED 自检（红→绿→蓝→白→灭），
       无论板载 WS2812 还是外接 LEDC LED，至少一种可见，便于确认硬件通路。 */
    actuator_led_test_sequence();
    /* v6：USB-JTAG 串口断开会触发芯片复位，自检后从 NVS 恢复上次的 LED 亮度，
       保证"断开串口灯不灭"。 */
    actuator_led_restore_state();
    ESP_ERROR_CHECK(wifi_mgr_init());
    ESP_ERROR_CHECK(wifi_mgr_start());

    /* Don't block on WiFi �?start CLI immediately.
       Commands work with mock fallback when offline.
       WiFi connects in background; agent loop uses real LLM once connected. */
    ESP_LOGI(TAG, "WiFi connecting in background...");

    ESP_ERROR_CHECK(trace_logger_init());
    ESP_ERROR_CHECK(rule_store_init());
    ESP_ERROR_CHECK(rule_engine_init());
    ESP_ERROR_CHECK(http_client_init());
    ESP_ERROR_CHECK(executor_init());
    ESP_ERROR_CHECK(llm_confirm_worker_init());

    /* WebSocket server for H5 dashboard (port 18789) */
    esp_err_t ws_ret = ws_server_start();
    if (ws_ret != ESP_OK) ESP_LOGW(TAG, "WS server start failed: %s", esp_err_to_name(ws_ret));

    /* Agent loop: 有注入数据时执行完整闭环（收数据→匹配→本地/云端），
       无注入时保持空闲（避免空转调 LLM 浪费额度）。 */
    xTaskCreatePinnedToCore(agent_loop_task, "agent", TASK_STACK_AGENT, NULL, TASK_PRIO_AGENT, &s_agent_handle, 1);

    ESP_LOGI(TAG, "Boot complete. CLI ready.");
    serial_cli_start();
}
