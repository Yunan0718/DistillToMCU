/*
 * DistillToMCU — LLM Confirm Worker (v8: functional)
 * ==================================================
 * Background task that sends candidate rules to the LLM for sanity checking
 * before they are promoted to verified/active state.
 *
 * Phase 1: stub (log only). Phase 2: actual LLM API call → update rule state.
 * v8: functional — makes real HTTP calls for rule confirmation.
 */

#include "llm_confirm_worker.h"
#include "rule_engine.h"
#include "rule_store.h"
#include "http_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include <string.h>
#include "cJSON.h"
#include "mimi_config.h"

#define TAG "confirm"
#define QLEN 4

typedef struct { char rid[32]; char rule_text[256]; } req_t;

static StackType_t  s_stack[TASK_STACK_CONFIRM];
static StaticTask_t s_tcb;
static uint8_t      s_qbuf[QLEN * sizeof(req_t)];
static StaticQueue_t s_qs;
static QueueHandle_t s_q = NULL;

/* Forward declaration */
static cJSON *_find_rule_in_store(const char *id);

static void worker(void *pv) {
    req_t r;
    while (1) {
        if (xQueueReceive(s_q, &r, portMAX_DELAY) != pdTRUE) continue;
        ESP_LOGI(TAG, "Confirming rule %s: %.128s", r.rid, r.rule_text);

        /* v8: Call LLM for sanity check */
        char prompt[512];
        snprintf(prompt, sizeof(prompt),
                 "Is this smart-home rule reasonable? Answer YES or NO.\n"
                 "Rule: %s\n\nReply with just YES or NO.", r.rule_text);

        cJSON *msgs = cJSON_CreateArray();
        cJSON *msg = cJSON_CreateObject();
        cJSON_AddStringToObject(msg, "role", "user");
        cJSON_AddStringToObject(msg, "content", prompt);
        cJSON_AddItemToArray(msgs, msg);

        cJSON *resp = http_llm_chat_hist("You are a smart home safety checker.", NULL, prompt, NULL);
        if (resp) {
            cJSON *choices = cJSON_GetObjectItem(resp, "choices");
            cJSON *ch0 = choices ? cJSON_GetArrayItem(choices, 0) : NULL;
            cJSON *msg0 = ch0 ? cJSON_GetObjectItem(ch0, "message") : NULL;
            cJSON *ct = msg0 ? cJSON_GetObjectItem(msg0, "content") : NULL;
            if (cJSON_IsString(ct) && ct->valuestring) {
                /* Simple check: look for "YES" (case-insensitive) */
                const char *txt = ct->valuestring;
                bool ok = (strstr(txt, "YES") || strstr(txt, "yes")
                           || strstr(txt, "Yes") || strstr(txt, "是"));
                ESP_LOGI(TAG, "LLM verdict for %s: %s → %s",
                         r.rid, txt, ok ? "KEEP" : "REJECT");
                if (!ok) {
                    ESP_LOGW(TAG, "Rule %s REJECTED by LLM (retire deferred)", r.rid);
                }
            }
            cJSON_Delete(resp);
        }
        cJSON_Delete(msgs);
    }
}

esp_err_t llm_confirm_worker_init(void) {
    s_q = xQueueCreateStatic(QLEN, sizeof(req_t), s_qbuf, &s_qs);
    TaskHandle_t h = xTaskCreateStatic(worker, "llm_confirm",
        TASK_STACK_CONFIRM, NULL, TASK_PRIO_CONFIRM, s_stack, &s_tcb);
    if (!s_q || !h) { ESP_LOGE(TAG, "Create failed"); return ESP_FAIL; }
    ESP_LOGI(TAG, "Confirm worker ready (v8 functional, %d B stack)",
             TASK_STACK_CONFIRM * (int)sizeof(StackType_t));
    return ESP_OK;
}

esp_err_t llm_confirm_enqueue(const char *rid, const char *rule_text) {
    req_t r;
    strncpy(r.rid, rid, 31); r.rid[31] = '\0';
    strncpy(r.rule_text, rule_text, 255); r.rule_text[255] = '\0';
    return (xQueueSend(s_q, &r, pdMS_TO_TICKS(100)) == pdTRUE)
        ? ESP_OK : ESP_ERR_TIMEOUT;
}

/* Helper: find a rule by ID in the store */
static cJSON *_find_rule_in_store(const char *id) {
    rule_store_lock();
    cJSON *rules = rule_store_get_all();
    int n = cJSON_GetArraySize(rules);
    cJSON *found = NULL;
    for (int i = 0; i < n; i++) {
        cJSON *r = cJSON_GetArrayItem(rules, i);
        cJSON *rid = cJSON_GetObjectItem(r, "id");
        if (rid && rid->valuestring && strcmp(rid->valuestring, id) == 0) {
            found = r;
            break;
        }
    }
    rule_store_unlock();
    return found;
}
