#include "rule_store.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "cJSON.h"
#include "mimi_config.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#define TAG "rule_store"
static cJSON *s_rules = NULL;
static bool s_dirty = false;
static SemaphoreHandle_t s_mutex = NULL;  /* v7: protect s_rules from concurrent access */

esp_err_t rule_store_init(void) {
    s_mutex = xSemaphoreCreateMutex();
    s_rules = cJSON_CreateArray();
    if (xSemaphoreTake(s_mutex, pdMS_TO_TICKS(1000)) != pdTRUE) {
        ESP_LOGE(TAG, "Mutex timeout on init");
        return ESP_FAIL;
    }
    FILE *f = fopen(LFS_RULES_FILE, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz > 0 && sz < 512 * 1024) {
            char *buf = heap_caps_calloc(1, sz + 1, MALLOC_CAP_SPIRAM);
            if (buf) {
                fread(buf, 1, sz, f);
                cJSON *loaded = cJSON_Parse(buf);
                if (loaded && cJSON_IsArray(loaded)) {
                    cJSON_Delete(s_rules);
                    s_rules = loaded;
                    ESP_LOGI(TAG, "Loaded %d rules", cJSON_GetArraySize(s_rules));
                } else cJSON_Delete(loaded);
                free(buf);
            }
        }
        fclose(f);
    }
    xSemaphoreGive(s_mutex);
    return ESP_OK;
}
/* v7: lock/unlock for rule_engine to hold across match+update cycles */
void rule_store_lock(void)   { if (s_mutex) xSemaphoreTake(s_mutex, pdMS_TO_TICKS(5000)); }
void rule_store_unlock(void) { if (s_mutex) xSemaphoreGive(s_mutex); }

cJSON *rule_store_get_all(void) { return s_rules; }
void rule_store_add(cJSON *obj) {
    rule_store_lock();
    if (cJSON_GetArraySize(s_rules) >= RULE_MAX_TOTAL) { rule_store_unlock(); return; }
    cJSON_AddItemToArray(s_rules, cJSON_Duplicate(obj, 1));
    s_dirty = true;
    rule_store_unlock();
}

void rule_store_mark_dirty(void) {
    s_dirty = true;
}
void rule_store_update(const char *id, cJSON *obj) {
    rule_store_lock();
    int n = cJSON_GetArraySize(s_rules);
    for (int i = 0; i < n; i++) {
        cJSON *r = cJSON_GetArrayItem(s_rules, i);
        cJSON *rid = cJSON_GetObjectItem(r, "id");
        if (rid && rid->valuestring && strcmp(rid->valuestring, id) == 0) {
            cJSON_ReplaceItemInArray(s_rules, i, cJSON_Duplicate(obj, 1));
            s_dirty = true; rule_store_unlock(); return;
        }
    }
    rule_store_unlock();
}
void rule_store_persist(void) {
    if (!s_dirty) return;
    rule_store_force_persist();
}

void rule_store_force_persist(void) {
    rule_store_lock();
    char *s = cJSON_PrintUnformatted(s_rules);
    if (!s) { rule_store_unlock(); return; }
    FILE *f = fopen(LFS_RULES_FILE, "wb");
    if (f) { fwrite(s, 1, strlen(s), f); fclose(f); s_dirty = false; }
    cJSON_free(s);
    rule_store_unlock();
}
