/*
 * DistillToMCU — HTTP Client (DeepSeek API)
 * Phase 1: basic connectivity + mock fallback.
 * Phase 2: full tool_call parsing loop.
 */

#include "http_client.h"
#include "esp_http_client.h"
#include "esp_crt_bundle.h"
#include "esp_log.h"
#include <stdlib.h>
#include <string.h>
#include "mimi_config.h"

#define TAG "http"
static char *s_key = NULL;

esp_err_t http_client_init(void) {
    s_key = strdup(LLM_API_KEY);
    ESP_LOGI(TAG, "HTTP ready, model=%s", LLM_MODEL);
    return ESP_OK;
}

/* response accumulator */
typedef struct { char *buf; size_t len, cap; } rb_t;

static esp_err_t _on_data(esp_http_client_event_t *e) {
    rb_t *rb = e->user_data;
    if (e->event_id == HTTP_EVENT_ON_DATA && rb) {
        size_t n = rb->len + e->data_len;
        if (n + 1 > rb->cap) { rb->cap = n + 1024; rb->buf = realloc(rb->buf, rb->cap); }
        memcpy(rb->buf + rb->len, e->data, e->data_len);
        rb->len = n; rb->buf[rb->len] = '\0';
    }
    return ESP_OK;
}

cJSON *http_llm_chat_hist(const char *sys, cJSON *history_msgs, const char *usr, cJSON *tools) {
    cJSON *req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "model", LLM_MODEL);
    cJSON *msgs = cJSON_CreateArray();
    cJSON *sm = cJSON_CreateObject();
    cJSON_AddStringToObject(sm, "role", "system");
    cJSON_AddStringToObject(sm, "content", sys);
    cJSON_AddItemToArray(msgs, sm);
    /* v6.2: 历史对话插在 system 之后、当前 user 之前（顺序很重要） */
    if (history_msgs && cJSON_IsArray(history_msgs)) {
        for (int i = 0; i < cJSON_GetArraySize(history_msgs); i++) {
            cJSON *hm = cJSON_GetArrayItem(history_msgs, i);
            cJSON *role = cJSON_GetObjectItem(hm, "role");
            cJSON *content = cJSON_GetObjectItem(hm, "content");
            if (!cJSON_IsString(role) || !cJSON_IsString(content)) continue;
            cJSON *hmsg = cJSON_CreateObject();
            cJSON_AddStringToObject(hmsg, "role", role->valuestring);
            cJSON_AddStringToObject(hmsg, "content", content->valuestring);
            cJSON_AddItemToArray(msgs, hmsg);
        }
    }
    cJSON *um = cJSON_CreateObject();
    cJSON_AddStringToObject(um, "role", "user");
    cJSON_AddStringToObject(um, "content", usr);
    cJSON_AddItemToArray(msgs, um);
    cJSON_AddItemToObject(req, "messages", msgs);
    /* Deep-copy tools so ownership stays with the caller (executor).
       cJSON_AddItemToObject transfers ownership — if we attached directly,
       cJSON_Delete(req) would free tools, leaving the caller with a dangling
       pointer and causing a double-free on cJSON_Delete(tools). */
    if (tools) cJSON_AddItemToObject(req, "tools", cJSON_Duplicate(tools, 1));
    cJSON_AddNumberToObject(req, "max_tokens", LLM_MAX_TOKENS);
    cJSON_AddNumberToObject(req, "temperature", LLM_TEMPERATURE);
    char *body = cJSON_PrintUnformatted(req);
    cJSON_Delete(req);

    esp_http_client_config_t c = {
        .url = LLM_BASE_URL "/v1/chat/completions",
        .method = HTTP_METHOD_POST,
        .timeout_ms = LLM_HTTP_TIMEOUT_MS,
        .event_handler = _on_data,
        .buffer_size = 2048,
        .crt_bundle_attach = esp_crt_bundle_attach,
        .transport_type = HTTP_TRANSPORT_OVER_SSL,
    };
    rb_t rb = {0};
    esp_http_client_handle_t h = esp_http_client_init(&c);
    esp_http_client_set_header(h, "Content-Type", "application/json");
    char ah[256];
    snprintf(ah, sizeof(ah), "Bearer %s", s_key);
    esp_http_client_set_header(h, "Authorization", ah);
    esp_http_client_set_post_field(h, body, strlen(body));
    esp_http_client_set_user_data(h, &rb);

    esp_err_t err = esp_http_client_perform(h);
    int st = esp_http_client_get_status_code(h);
    esp_http_client_cleanup(h);
    free(body);

    if (err != ESP_OK) { ESP_LOGE(TAG, "HTTP err: %s", esp_err_to_name(err)); free(rb.buf); return NULL; }
    if (st != 200) { ESP_LOGE(TAG, "HTTP %d: %.100s", st, rb.buf ? rb.buf : ""); free(rb.buf); return NULL; }

    cJSON *resp = cJSON_Parse(rb.buf);
    free(rb.buf);
    if (!resp) ESP_LOGE(TAG, "JSON parse failed");
    return resp;
}

cJSON *http_llm_chat(const char *sys, const char *usr, cJSON *tools) {
    return http_llm_chat_hist(sys, NULL, usr, tools);
}
