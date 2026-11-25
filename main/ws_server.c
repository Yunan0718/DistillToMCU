/*
 * DistillToMCU — WebSocket Server
 * ================================
 * Dashboard (H5) connection gateway on port 18789.
 * Based on the mimiclaw ws_server pattern (MIT), adapted for DistillToMCU:
 *   - No message bus dependency
 *   - Handles: message (command to executor), sensor (inject data)
 *   - Broadcasts: sensor/stats/rules telemetry to all dashboards
 *
 * WebSocket protocol:
 *   client → {"type":"message","content":"led r 50"}
 *   client → {"type":"sensor","sensors":{"temperature":23.5,...}}
 *   server → {"type":"response","content":"..."}
 *   server → {"type":"sensor","sensors":{...}}     (telemetry push)
 *   server → {"type":"stats","data":{...}}
 */

#include "ws_server.h"
#include "executor.h"
#include "sensor.h"
#include "actuator.h"
#include "rule_engine.h"
#include "rule_store.h"
#include "trace_logger.h"

#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "cJSON.h"

#define TAG "ws"

static httpd_handle_t s_server = NULL;

/* ---- AI 对话 worker（v6.1）----
 * LLM 调用最长 15s，不能在 httpd 任务里同步执行；
 * 对话请求放入独立任务，完成后广播 {"type":"chat","data":{...}}。 */
#define CHAT_MSG_BUF 768
typedef struct {
    char content[CHAT_MSG_BUF];
    char sid[40];
    cJSON *history;
} chat_req_t;
static TaskHandle_t s_chat_task = NULL;

static void chat_worker(void *pv)
{
    chat_req_t *req = (chat_req_t *)pv;
    cJSON *res = executor_handle_hist(req->content, req->history);
    if (!res) res = cJSON_CreateObject();
    if (req->sid[0]) cJSON_AddStringToObject(res, "sid", req->sid);
    cJSON_AddStringToObject(res, "src", "ws_chat");
    /* v6.2: broadcast 只借用（内部深拷贝），调用方负责释放 */
    ws_server_broadcast("chat", res);
    cJSON_Delete(res);
    if (req->history) cJSON_Delete(req->history);
    free(req);
    s_chat_task = NULL;
    vTaskDelete(NULL);
}

static void start_chat(const char *text, cJSON *history, const char *sid)
{
    if (s_chat_task) {
        cJSON *r = cJSON_CreateObject();
        cJSON_AddStringToObject(r, "status", "busy");
        ws_server_broadcast("response", r);
        cJSON_Delete(r);
        return;
    }
    chat_req_t *req = calloc(1, sizeof(chat_req_t));
    if (!req) return;
    strncpy(req->content, text ? text : "", sizeof(req->content) - 1);
    req->content[sizeof(req->content) - 1] = '\0';
    strncpy(req->sid, sid ? sid : "", sizeof(req->sid) - 1);
    req->sid[sizeof(req->sid) - 1] = '\0';
    if (history && cJSON_IsArray(history)) {
        req->history = cJSON_Duplicate(history, 1);
        if (!req->history) { free(req); return; }
    }
    if (xTaskCreate(chat_worker, "chat", 16 * 1024, req, 5, &s_chat_task) != pdPASS) {
        if (req->history) cJSON_Delete(req->history);
        free(req);
        s_chat_task = NULL;
    }
}

/* ---- client tracking ---- */
typedef struct {
    int fd;
    char chat_id[32];
    bool active;
} ws_client_t;

static ws_client_t s_clients[D2M_WS_MAX_CLIENTS];

static ws_client_t *find_client_by_fd(int fd)
{
    for (int i = 0; i < D2M_WS_MAX_CLIENTS; i++)
        if (s_clients[i].active && s_clients[i].fd == fd)
            return &s_clients[i];
    return NULL;
}

static ws_client_t *add_client(int fd)
{
    for (int i = 0; i < D2M_WS_MAX_CLIENTS; i++) {
        if (!s_clients[i].active) {
            s_clients[i].fd = fd;
            snprintf(s_clients[i].chat_id, sizeof(s_clients[i].chat_id), "ws_%d", fd);
            s_clients[i].active = true;
            ESP_LOGI(TAG, "Client connected: %s (fd=%d)", s_clients[i].chat_id, fd);
            return &s_clients[i];
        }
    }
    ESP_LOGW(TAG, "Max clients reached, rejecting fd=%d", fd);
    return NULL;
}

static void remove_client(int fd)
{
    for (int i = 0; i < D2M_WS_MAX_CLIENTS; i++) {
        if (s_clients[i].active && s_clients[i].fd == fd) {
            ESP_LOGI(TAG, "Client disconnected: %s", s_clients[i].chat_id);
            s_clients[i].active = false;
            return;
        }
    }
}

/* ---- handle a command from dashboard (runs in httpd task, light ops only) ---- */
static void handle_command(const char *cmd)
{
    /* AI 对话：say <文本> 触发完整 agent 闭环 */
    if (strncmp(cmd, "say ", 4) == 0) { start_chat(cmd + 4, NULL, NULL); return; }
    if (strcmp(cmd, "say") == 0) { start_chat("", NULL, NULL); return; }

    /* Direct GPIO control: "led r 50" / "led r 0" / "led g 30" / "led b 60" */
    if (strncmp(cmd, "led ", 4) == 0) {
        const char *color = cmd + 4;
        int pct = 50;
        const char *p = strchr(color, ' ');
        if (p) { pct = atoi(p + 1); color = cmd + 4; }
        if (pct < 0) pct = 0;
        if (pct > 100) pct = 100;
        if (strncmp(color, "r", 1) == 0)      actuator_led_r_set((uint8_t)pct);
        else if (strncmp(color, "g", 1) == 0) actuator_led_g_set((uint8_t)pct);
        else if (strncmp(color, "b", 1) == 0) actuator_led_b_set((uint8_t)pct);
        else if (strncmp(color, "off", 3) == 0) {
            actuator_led_r_set(0); actuator_led_g_set(0); actuator_led_b_set(0);
        }
        cJSON *res = cJSON_CreateObject();
        cJSON_AddStringToObject(res, "status", "led_set");
        cJSON_AddStringToObject(res, "color", color);
        cJSON_AddNumberToObject(res, "pct", pct);
        ws_server_broadcast("response", res);
        cJSON_Delete(res);
        return;
    }

    /* For Phase 1: execute only lightweight commands synchronously */
    if (strncmp(cmd, "stats", 5) == 0) {
            int t, l, c;
            executor_get_stats(&t, &l, &c);
            cJSON *res = cJSON_CreateObject();
            cJSON_AddNumberToObject(res, "total", t);
            cJSON_AddNumberToObject(res, "local", l);
            cJSON_AddNumberToObject(res, "cloud", c);
            cJSON_AddNumberToObject(res, "ar", t > 0 ? l * 100.0 / t : 0);
            ws_server_broadcast("stats", res);
            cJSON_Delete(res);
        } else if (strncmp(cmd, "rules", 5) == 0) {
            cJSON *rules = rule_engine_get_all_rules();
            /* v6.2 crash fix: 绝不能把实时规则库直接交给 broadcast——
               broadcast 会 cJSON_AddItemToObject + cJSON_Delete(msg)，
               等于把固件自己的 s_rules 释放掉，下一次规则匹配直接
               LoadProhibited 崩溃。必须发深拷贝。 */
            if (rules) {
                cJSON *dup = cJSON_Duplicate(rules, 1);
                if (dup) {
                    ws_server_broadcast("rules", dup);
                    cJSON_Delete(dup);
                }
            }
        } else if (strncmp(cmd, "heap", 4) == 0) {
            cJSON *res = cJSON_CreateObject();
            cJSON_AddNumberToObject(res, "free_heap",
                esp_get_free_heap_size());
            cJSON_AddNumberToObject(res, "free_psram",
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
            ws_server_broadcast("heap", res);
            cJSON_Delete(res);
        } else {
            /* Unknown / heavy command — log only for Phase 1 */
            cJSON *res = cJSON_CreateObject();
            cJSON_AddStringToObject(res, "note", "command_ignored_phase1");
            ws_server_broadcast("response", res);
            cJSON_Delete(res);
        }
}

/* ---- WebSocket handler ---- */
static esp_err_t ws_handler(httpd_req_t *req)
{
    if (req->method == HTTP_GET) {
        int fd = httpd_req_to_sockfd(req);
        add_client(fd);
        return ESP_OK;
    }

    httpd_ws_frame_t ws_pkt = {0};
    ws_pkt.type = HTTPD_WS_TYPE_TEXT;

    esp_err_t ret = httpd_ws_recv_frame(req, &ws_pkt, 0);
    if (ret != ESP_OK) return ret;
    if (ws_pkt.len == 0) return ESP_OK;

    ws_pkt.payload = calloc(1, ws_pkt.len + 1);
    if (!ws_pkt.payload) return ESP_ERR_NO_MEM;

    ret = httpd_ws_recv_frame(req, &ws_pkt, ws_pkt.len);
    if (ret != ESP_OK) {
        free(ws_pkt.payload);
        return ret;
    }

    cJSON *root = cJSON_Parse((char *)ws_pkt.payload);
    free(ws_pkt.payload);

    if (!root) return ESP_OK;

    cJSON *type = cJSON_GetObjectItem(root, "type");
    if (type && cJSON_IsString(type)) {
        const char *t = type->valuestring;

        if (strcmp(t, "sensor") == 0) {
            /* Inject sensor data from dashboard */
            cJSON *sensors = cJSON_GetObjectItem(root, "sensors");
            if (sensors) {
                sensor_inject(cJSON_Duplicate(sensors, 1));
                cJSON *ack = cJSON_CreateObject();
                cJSON_AddStringToObject(ack, "status", "injected");
                ws_server_broadcast("response", ack);
                cJSON_Delete(ack);
            }
        } else if (strcmp(t, "message") == 0) {
            cJSON *content = cJSON_GetObjectItem(root, "content");
            if (content && cJSON_IsString(content)) {
                handle_command(content->valuestring);
            }
        } else if (strcmp(t, "trace_read") == 0) {
            /* PC reads traces from SPIFFS for distillation */
            cJSON *traces = trace_logger_read_all();
            if (traces) {
                ws_server_broadcast("trace_data", traces);
                cJSON_Delete(traces);  /* broadcast does a deep copy */
            } else {
                cJSON *empty = cJSON_CreateArray();
                ws_server_broadcast("trace_data", empty);
                cJSON_Delete(empty);
            }
        } else if (strcmp(t, "rule_push") == 0) {
            /* PC pushes a distilled rule back to MCU */
            cJSON *rule = cJSON_GetObjectItem(root, "rule");
            if (cJSON_IsObject(rule)) {
                /* Validate: device and command must be non-empty strings.
                   NULL/empty values cause executor to crash on ->valuestring. */
                cJSON *act = cJSON_GetObjectItem(rule, "action");
                if (!cJSON_IsObject(act)) {
                    ESP_LOGW(TAG, "rule_push: no valid action object");
                } else {
                    cJSON *dev = cJSON_GetObjectItem(act, "device");
                    cJSON *cmd = cJSON_GetObjectItem(act, "command");
                    if (!cJSON_IsString(dev) || !dev->valuestring || !dev->valuestring[0]
                        || !cJSON_IsString(cmd) || !cmd->valuestring || !cmd->valuestring[0]) {
                        ESP_LOGW(TAG, "rule_push: device/command missing or invalid");
                    } else {
                        cJSON *dup = cJSON_Duplicate(rule, 1);
                        rule_store_add(dup);
                        rule_store_force_persist();
                        cJSON *ack = cJSON_CreateObject();
                        cJSON_AddStringToObject(ack, "status", "rule_added");
                        ws_server_broadcast("response", ack);
                    }
                }
            }
        } else if (strcmp(t, "trace_clear") == 0) {
            trace_logger_clear();
            cJSON *ack = cJSON_CreateObject();
            cJSON_AddStringToObject(ack, "status", "traces_cleared");
            ws_server_broadcast("response", ack);
        } else if (strcmp(t, "chat") == 0) {
            /* AI 对话专用消息：{"type":"chat","content":"开灯"} */
            cJSON *content = cJSON_GetObjectItem(root, "content");
            if (content && cJSON_IsString(content)) {
                cJSON *history = cJSON_GetObjectItem(root, "history");
                cJSON *sid = cJSON_GetObjectItem(root, "sid");
                start_chat(content->valuestring, history,
                           cJSON_IsString(sid) ? sid->valuestring : NULL);
            }
        }
    }

    cJSON_Delete(root);
    return ESP_OK;
}

/* ---- server lifecycle ---- */
esp_err_t ws_server_start(void)
{
    memset(s_clients, 0, sizeof(s_clients));

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = D2M_WS_PORT;
    config.ctrl_port = D2M_WS_PORT + 1;
    config.max_open_sockets = D2M_WS_MAX_CLIENTS;
    config.stack_size = 8192;  /* enough for cJSON + command dispatch */

    esp_err_t ret = httpd_start(&s_server, &config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start WebSocket server: %s", esp_err_to_name(ret));
        return ret;
    }

    httpd_uri_t ws_uri = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = ws_handler,
        .is_websocket = true,
    };
    httpd_register_uri_handler(s_server, &ws_uri);

    ESP_LOGI(TAG, "WebSocket server started on port %d", D2M_WS_PORT);
    return ESP_OK;
}

esp_err_t ws_server_stop(void)
{
    if (s_server) {
        httpd_stop(s_server);
        s_server = NULL;
    }
    return ESP_OK;
}

/* ---- broadcast telemetry to all clients ---- */
esp_err_t ws_server_broadcast(const char *type, void *payload)
{
    if (!s_server) return ESP_ERR_INVALID_STATE;
    if (!payload) return ESP_ERR_INVALID_ARG;

    /* v6.2: 深拷贝 payload——调用方保留所有权并自行释放，
       避免把固件内部对象（如实时规则库）误传给 broadcast 后被释放。 */
    cJSON *msg = cJSON_CreateObject();
    cJSON_AddStringToObject(msg, "type", type);
    cJSON_AddItemToObject(msg, "data", cJSON_Duplicate((cJSON *)payload, 1));

    char *json_str = cJSON_PrintUnformatted(msg);
    cJSON_Delete(msg);
    if (!json_str) return ESP_ERR_NO_MEM;

    esp_err_t ret_all = ESP_OK;
    for (int i = 0; i < D2M_WS_MAX_CLIENTS; i++) {
        if (!s_clients[i].active) continue;
        httpd_ws_frame_t frame = {
            .type = HTTPD_WS_TYPE_TEXT,
            .payload = (uint8_t *)json_str,
            .len = strlen(json_str),
        };
        esp_err_t r = httpd_ws_send_frame_async(s_server, s_clients[i].fd, &frame);
        if (r != ESP_OK) {
            ESP_LOGW(TAG, "Send to %s failed: %s", s_clients[i].chat_id, esp_err_to_name(r));
            remove_client(s_clients[i].fd);
            ret_all = r;
        }
    }

    free(json_str);
    return ret_all;
}
