/*
 * DistillToMCU — Sensor Manager (v2: USB serial injection)
 * ========================================================
 * Phase 1: USB serial sensor injection from PC datasets (UCI/CASAS).
 *   Falls back to mock values when no injection data is available.
 * Phase 2: DHT22, BH1750 via I2C (optional).
 */

#include "sensor.h"
#include "driver/gpio.h"
#include "cJSON.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#define TAG "sensor"

/* ---- injected sensor state ---- */
/* v7: FIFO ring queue — PC 按节奏喂送时 agent loop 每条处理一次，不丢快照 */
#define INJECT_QUEUE_MAX 8
static cJSON *s_inject_q[INJECT_QUEUE_MAX] = {0};
static int s_q_head = 0;
static int s_q_count = 0;
static int64_t s_injected_at_us = 0;
static SemaphoreHandle_t s_q_mutex = NULL;  /* v7: protect FIFO from concurrent inject/pop */

/* v10: injection diagnostics — count parse/success/drop to debug
   "100 injections, only 7 handled" on the PC side. */
static uint32_t s_inject_ok = 0;
static uint32_t s_inject_dropped = 0;
static uint32_t s_inject_parse_fail = 0;

void sensor_get_inject_stats(uint32_t *ok, uint32_t *dropped, uint32_t *parse_fail)
{
    if (ok) *ok = s_inject_ok;
    if (dropped) *dropped = s_inject_dropped;
    if (parse_fail) *parse_fail = s_inject_parse_fail;
}

void sensor_note_parse_fail(void)
{
    s_inject_parse_fail++;
}

esp_err_t sensor_init(void)
{
    s_q_mutex = xSemaphoreCreateMutex();
    gpio_config_t btn_cfg = {
        .pin_bit_mask = (1ULL << PIN_BUTTON_ACCEPT) | (1ULL << PIN_BUTTON_CORRECT),
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&btn_cfg);
    ESP_LOGI(TAG, "Ready (USB serial injection + mock fallback)");
    return ESP_OK;
}

cJSON *sensor_snapshot(void)
{
    /* ---- prefer injected data (from USB serial) if fresh ---- */
    if (s_q_count > 0 && sensor_has_injection()) {
        cJSON *snap = cJSON_Duplicate(s_inject_q[s_q_head], 1);
        /* add button state */
        cJSON_AddNumberToObject(snap, "btn_accept", sensor_button_accept_pressed() ? 1 : 0);
        cJSON_AddNumberToObject(snap, "btn_correct", sensor_button_correct_pressed() ? 1 : 0);
        return snap;
    }

    /* ---- fallback: mock defaults ---- */
    cJSON *obj = cJSON_CreateObject();
    cJSON_AddNumberToObject(obj, "temperature", 25.0);
    cJSON_AddNumberToObject(obj, "humidity",    55.0);
    cJSON_AddNumberToObject(obj, "light",       500.0);
    cJSON_AddNumberToObject(obj, "motion",      1);
    cJSON_AddNumberToObject(obj, "btn_accept",  sensor_button_accept_pressed() ? 1 : 0);
    cJSON_AddNumberToObject(obj, "btn_correct", sensor_button_correct_pressed() ? 1 : 0);
    return obj;
}

void sensor_inject(cJSON *data)
{
    if (!data) return;

    if (s_q_mutex) xSemaphoreTake(s_q_mutex, pdMS_TO_TICKS(100));
    if (s_q_count >= INJECT_QUEUE_MAX) {
        s_inject_dropped++;
        ESP_LOGW(TAG, "Injection queue full, dropping snapshot (total dropped=%u)",
                 (unsigned)s_inject_dropped);
        if (s_q_mutex) xSemaphoreGive(s_q_mutex);
        cJSON_Delete(data);
        return;
    }
    int tail = (s_q_head + s_q_count) % INJECT_QUEUE_MAX;
    s_inject_q[tail] = data;  /* takes ownership */
    s_q_count++;
    s_inject_ok++;
    s_injected_at_us = esp_timer_get_time();
    if (s_q_mutex) xSemaphoreGive(s_q_mutex);

    char *js = cJSON_PrintUnformatted(data);
    ESP_LOGI(TAG, "Injected: %s", js);
    free(js);
}

bool sensor_has_injection(void)
{
    if (s_q_mutex) xSemaphoreTake(s_q_mutex, pdMS_TO_TICKS(10));
    bool has = s_q_count > 0;
    if (has) {
        int64_t elapsed = esp_timer_get_time() - s_injected_at_us;
        has = elapsed < SENSOR_INJECT_TIMEOUT_US;
    }
    if (s_q_mutex) xSemaphoreGive(s_q_mutex);
    return has;
}

void sensor_clear_injection(void)
{
    if (s_q_mutex) xSemaphoreTake(s_q_mutex, pdMS_TO_TICKS(100));
    while (s_q_count > 0) {
        int idx = (s_q_head + s_q_count - 1) % INJECT_QUEUE_MAX;
        if (s_inject_q[idx]) cJSON_Delete(s_inject_q[idx]);
        s_inject_q[idx] = NULL;
        s_q_count--;
    }
    s_q_head = 0;
    s_injected_at_us = 0;
    if (s_q_mutex) xSemaphoreGive(s_q_mutex);
}

void sensor_pop_injection(void)
{
    if (s_q_mutex) xSemaphoreTake(s_q_mutex, pdMS_TO_TICKS(10));
    if (s_q_count <= 0) { if (s_q_mutex) xSemaphoreGive(s_q_mutex); return; }
    if (s_inject_q[s_q_head]) cJSON_Delete(s_inject_q[s_q_head]);
    s_inject_q[s_q_head] = NULL;
    s_q_head = (s_q_head + 1) % INJECT_QUEUE_MAX;
    s_q_count--;
    if (s_q_count <= 0) s_q_head = 0;
    if (s_q_mutex) xSemaphoreGive(s_q_mutex);
}

bool sensor_button_accept_pressed(void) {
    return gpio_get_level(PIN_BUTTON_ACCEPT) == 0;
}
bool sensor_button_correct_pressed(void) {
    return gpio_get_level(PIN_BUTTON_CORRECT) == 0;
}
