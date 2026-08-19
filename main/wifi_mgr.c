#include "wifi_mgr.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_log.h"
#include <string.h>
#include "mimi_config.h"

#define TAG "wifi"
static EventGroupHandle_t s_evt;
#define WIFI_BIT BIT0

static void _handler(void *a, esp_event_base_t b, int32_t id, void *d) {
    if (b == WIFI_EVENT && id == WIFI_EVENT_STA_START) esp_wifi_connect();
    else if (b == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "WiFi disco, reconnecting");
        xEventGroupClearBits(s_evt, WIFI_BIT);
        esp_wifi_connect();
    } else if (b == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(s_evt, WIFI_BIT);
    }
}

esp_err_t wifi_mgr_init(void) {
    s_evt = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &_handler, NULL, NULL));
    wifi_config_t wcfg = {{{0}}};
    strncpy((char*)wcfg.sta.ssid, WIFI_SSID, 31);
    strncpy((char*)wcfg.sta.password, WIFI_PASS, 63);
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wcfg));
    ESP_LOGI(TAG, "WiFi init OK: %s", WIFI_SSID);
    return ESP_OK;
}
esp_err_t wifi_mgr_start(void) { return esp_wifi_start(); }
esp_err_t wifi_mgr_wait_connected(int ms) {
    EventBits_t b = xEventGroupWaitBits(s_evt, WIFI_BIT, pdTRUE, pdFALSE, pdMS_TO_TICKS(ms));
    if (b & WIFI_BIT) { ESP_LOGI(TAG, "Connected!"); return ESP_OK; }
    return ESP_ERR_TIMEOUT;
}
