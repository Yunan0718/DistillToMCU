#include "trace_logger.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "freertos/FreeRTOS.h"
#include "freertos/timers.h"
#include "mimi_config.h"

#define TAG "trace"
static char  *s_rb = NULL;
static size_t s_wp = 0, s_cnt = 0;
static TimerHandle_t s_tmr = NULL;
static char s_path[128];

/* C callback — no lambdas in C */
static void _flush_cb(TimerHandle_t x) {
    if (s_cnt == 0) return;
    FILE *f = fopen(s_path, "ab");
    if (!f) { ESP_LOGE(TAG, "Cannot open trace file"); return; }
    size_t w = fwrite(s_rb, 1, s_wp, f); fclose(f);
    ESP_LOGD(TAG, "Flushed %d traces (%d B)", s_cnt, w);
    s_wp = 0; s_cnt = 0;
    struct stat st;
    if (stat(s_path, &st) == 0 && st.st_size > TRACE_MAX_FILE_SIZE)
        snprintf(s_path, sizeof(s_path), "%s/trace_%03d.jsonl",
                 LFS_BASE_PATH, (int)(esp_timer_get_time() / 1000000ULL) % 1000);
}

esp_err_t trace_logger_init(void) {
    s_rb = heap_caps_calloc(1, RAM_TRACE_RINGBUF_SIZE, MALLOC_CAP_SPIRAM);
    if (!s_rb) { ESP_LOGE(TAG, "OOM ringbuf"); return ESP_ERR_NO_MEM; }
    snprintf(s_path, sizeof(s_path), "%s", LFS_TRACE_FILE);
    s_tmr = xTimerCreate("tf", pdMS_TO_TICKS(TRACE_FLUSH_INTERVAL_MS),
                         pdTRUE, NULL, _flush_cb);
    xTimerStart(s_tmr, 0);
    ESP_LOGI(TAG, "Trace logger ready, %d B ringbuf", RAM_TRACE_RINGBUF_SIZE);
    return ESP_OK;
}

esp_err_t trace_logger_record(cJSON *obj) {
    if (!s_rb || !obj) return ESP_ERR_INVALID_ARG;
    char *s = cJSON_PrintUnformatted(obj);
    if (!s) return ESP_ERR_NO_MEM;
    size_t sl = strlen(s);
    if (s_wp + sl + 2 > RAM_TRACE_RINGBUF_SIZE) _flush_cb(NULL);
    memcpy(s_rb + s_wp, s, sl); s_wp += sl;
    s_rb[s_wp++] = '\n'; s_cnt++;
    cJSON_free(s);
    if (s_cnt >= TRACE_FLUSH_COUNT) _flush_cb(NULL);
    return ESP_OK;
}

/* ---- read traces from SPIFFS for PC distillation ---- */
cJSON *trace_logger_read_all(void)
{
    /* flush any pending ringbuf data first */
    if (s_cnt > 0) _flush_cb(NULL);

    cJSON *arr = cJSON_CreateArray();
    FILE *f = fopen(s_path, "r");
    if (!f) return arr;  /* no traces yet */

    char line[2048];
    while (fgets(line, sizeof(line), f)) {
        /* trim trailing newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;
        cJSON *obj = cJSON_Parse(line);
        if (obj) cJSON_AddItemToArray(arr, obj);
    }
    fclose(f);
    ESP_LOGI(TAG, "Read %d traces from %s", cJSON_GetArraySize(arr), s_path);
    return arr;
}

esp_err_t trace_logger_clear(void)
{
    if (s_cnt > 0) _flush_cb(NULL);
    FILE *f = fopen(s_path, "w");
    if (f) { fclose(f); }
    return ESP_OK;
}
