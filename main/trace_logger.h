/*
 * DistillToMCU — Trace Logger Header
 */

#pragma once
#include "esp_err.h"
#include "cJSON.h"

esp_err_t trace_logger_init(void);
esp_err_t trace_logger_record(cJSON *trace_obj);
/* Read all traces from SPIFFS file as JSON array. Caller must cJSON_Delete. */
cJSON *trace_logger_read_all(void);
/* Clear the trace file after PC has processed it */
esp_err_t trace_logger_clear(void);
