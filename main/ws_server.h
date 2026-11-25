/*
 * DistillToMCU — WebSocket Server Header
 * ======================================
 * Dashboard (H5) connection gateway on port 18789.
 */

#pragma once
#include "esp_err.h"

#define D2M_WS_PORT         18789
#define D2M_WS_MAX_CLIENTS   4

esp_err_t ws_server_start(void);
esp_err_t ws_server_stop(void);

/* Broadcast a telemetry message to all connected dashboards.
 * type: "sensor" | "stats" | "rules" | "log"
 * payload: cJSON object to broadcast (deep-copied internally, caller keeps ownership). */
esp_err_t ws_server_broadcast(const char *type, void *payload);
