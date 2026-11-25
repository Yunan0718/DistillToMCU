#pragma once
#include "esp_err.h"
#include <stdbool.h>
#include "cJSON.h"
#include "mimi_config.h"

esp_err_t sensor_init(void);

/* Get current sensor snapshot. Returns injected values if available,
 * otherwise mock defaults. Caller must cJSON_Delete. */
cJSON *sensor_snapshot(void);

/* Inject sensor data from USB serial (PC-side data source).
 * Called by serial_cli or dedicated UART RX handler.
 * Takes ownership of the cJSON object. */
void sensor_inject(cJSON *data);

/* Check if injected data is available (non-expired) */
bool sensor_has_injection(void);

/* Clear injected data (e.g. after processing) */
void sensor_clear_injection(void);

/* v7: 弹出队首注入快照（agent loop 处理完一条后调用）。
 * 注入队列最多 INJECT_QUEUE_MAX 条，PC 端按节奏喂送不会丢数据。 */
void sensor_pop_injection(void);

/* v10: injection diagnostics counters (parse success/drop/parse-fail) */
void sensor_get_inject_stats(uint32_t *ok, uint32_t *dropped, uint32_t *parse_fail);
void sensor_note_parse_fail(void);

/* Button state */
bool sensor_button_accept_pressed(void);
bool sensor_button_correct_pressed(void);
