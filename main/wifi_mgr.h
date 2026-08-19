/*
 * DistillToMCU — WiFi Manager
 * ===========================
 * ESP32-S3 station mode, exponential backoff.
 */

#pragma once
#include "esp_err.h"
esp_err_t wifi_mgr_init(void);
esp_err_t wifi_mgr_start(void);
esp_err_t wifi_mgr_wait_connected(int timeout_ms);
