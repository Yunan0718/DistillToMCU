/*
 * DistillToMCU — Execution Engine (v2: real LLM + serial inject)
 * ============================================================
 * Phase 1: 真实 DeepSeek API 调用 + tool_call 解析 + USB 串口传感器注入
 *
 * 闭环流程:
 *   传感器数据 (串口注入 / mock默认) → 规则匹配 → 本地执行
 *                                              ↓ 未命中
 *                                         http_llm_chat()
 *                                              ↓
 *                                         tool_call 解析
 *                                              ↓
 *                                         GPIO 执行
 */

#include "executor.h"
#include "rule_engine.h"
#include "trace_logger.h"
#include "actuator.h"
#include "sensor.h"
#include "http_client.h"
#include "llm_confirm_worker.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "cJSON.h"
#include <string.h>
#include "mimi_config.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#define TAG "executor"
static int s_t = 0, s_l = 0, s_c = 0;

/* v10.5f: real match-latency telemetry (D3).
 * Ring buffer of rule_engine_match() durations in microseconds.
 * Exposed via serial CLI `latstats` (p50/p95/p99). */
#define LAT_SAMPLES_MAX 256
static uint32_t s_match_lat_us[LAT_SAMPLES_MAX];
static int s_match_lat_n = 0;
static int s_match_lat_head = 0;
static uint64_t s_match_lat_sum = 0;
static uint32_t s_match_lat_max = 0;

/* v10.5f (D5): user-override feedback window.
 * After a local rule execution, the user has FEEDBACK_WINDOW_US to press
 * the CORRECT button (GPIO14) to veto the rule; otherwise the execution
 * auto-accepts. This gives real precision/recall/override signals
 * (metrics_calculator previously had to mark them not reportable). */
#define FEEDBACK_WINDOW_US (5 * 1000 * 1000)
static char s_pending_rid[24] = {0};
static int64_t s_pending_at_us = 0;
static uint32_t s_override_count = 0;

void executor_pending_feedback(const char *rid) {
    if (!rid) return;
    strncpy(s_pending_rid, rid, sizeof(s_pending_rid) - 1);
    s_pending_rid[sizeof(s_pending_rid) - 1] = '\0';
    s_pending_at_us = esp_timer_get_time();
}

void executor_feedback_poll(void) {
    if (s_pending_rid[0] == '\0') return;
    int64_t elapsed = esp_timer_get_time() - s_pending_at_us;
    if (elapsed >= FEEDBACK_WINDOW_US) {
        rule_engine_update_on_exec(s_pending_rid, "accepted");
        s_pending_rid[0] = '\0';
    } else if (sensor_button_correct_pressed()) {
        rule_engine_update_on_exec(s_pending_rid, "corrected");
        s_override_count++;
        s_pending_rid[0] = '\0';
    }
}

uint32_t executor_get_override_count(void) { return s_override_count; }

static void lat_record(uint32_t us) {
    if (s_match_lat_n < LAT_SAMPLES_MAX) {
        s_match_lat_us[s_match_lat_n++] = us;
        s_match_lat_sum += us;
    } else {
        uint32_t old = s_match_lat_us[s_match_lat_head];
        s_match_lat_us[s_match_lat_head] = us;
        s_match_lat_sum = s_match_lat_sum - old + us;
    }
    s_match_lat_head = (s_match_lat_head + 1) % LAT_SAMPLES_MAX;
    if (us > s_match_lat_max) s_match_lat_max = us;
}

void executor_latstats(int *n, uint32_t *mean_us, uint32_t *p50,
                       uint32_t *p95, uint32_t *p99, uint32_t *max_us) {
    if (s_match_lat_n == 0) {
        *n = 0; *mean_us = 0; *p50 = 0; *p95 = 0; *p99 = 0; *max_us = 0;
        return;
    }
    uint32_t tmp[LAT_SAMPLES_MAX];
    for (int i = 0; i < s_match_lat_n; i++) tmp[i] = s_match_lat_us[i];
    /* simple insertion sort (n <= 256, fine for telemetry) */
    for (int i = 1; i < s_match_lat_n; i++) {
        uint32_t v = tmp[i];
        int j = i - 1;
        while (j >= 0 && tmp[j] > v) { tmp[j + 1] = tmp[j]; j--; }
        tmp[j + 1] = v;
    }
    *n = s_match_lat_n;
    *mean_us = (uint32_t)(s_match_lat_sum / s_match_lat_n);
    *max_us = s_match_lat_max;
    *p50 = tmp[s_match_lat_n / 2];
    *p95 = tmp[(int)(s_match_lat_n * 0.95) - 1];
    *p99 = tmp[(int)(s_match_lat_n * 0.99) - 1];
}

/* ---- v9: async cloud worker state ---- */
#define CLOUD_QUEUE_LEN 16
typedef struct {
    char *sensors_json;
    char input[128];
} cloud_req_t;

static QueueHandle_t s_cloud_q = NULL;
static void cloud_worker(void *pv);  /* defined after static decls */

/* ---- AI 对话多轮记忆（v6.2）----
 * 环状缓冲保存最近 N 轮 (用户→助手)，让"关掉它""再暗一点"这类
 * 依赖上下文的指令能被 LLM 正确理解。 */
#define CHAT_HISTORY_MAX 6
#define CHAT_MSG_MAX 160
static char s_hist_user[CHAT_HISTORY_MAX][CHAT_MSG_MAX];
static char s_hist_ai[CHAT_HISTORY_MAX][CHAT_MSG_MAX];
static int s_hist_count = 0;
static int s_hist_head = 0;

static void chat_history_add(const char *user_text, const char *ai_text)
{
    int idx = s_hist_head;
    strncpy(s_hist_user[idx], user_text ? user_text : "", CHAT_MSG_MAX - 1);
    s_hist_user[idx][CHAT_MSG_MAX - 1] = '\0';
    strncpy(s_hist_ai[idx], ai_text ? ai_text : "", CHAT_MSG_MAX - 1);
    s_hist_ai[idx][CHAT_MSG_MAX - 1] = '\0';
    s_hist_head = (s_hist_head + 1) % CHAT_HISTORY_MAX;
    if (s_hist_count < CHAT_HISTORY_MAX) s_hist_count++;
}

static cJSON *chat_history_build(void)
{
    cJSON *arr = cJSON_CreateArray();
    int start = (s_hist_head - s_hist_count + CHAT_HISTORY_MAX) % CHAT_HISTORY_MAX;
    for (int i = 0; i < s_hist_count; i++) {
        int idx = (start + i) % CHAT_HISTORY_MAX;
        if (s_hist_user[idx][0]) {
            cJSON *mu = cJSON_CreateObject();
            cJSON_AddStringToObject(mu, "role", "user");
            cJSON_AddStringToObject(mu, "content", s_hist_user[idx]);
            cJSON_AddItemToArray(arr, mu);
        }
        if (s_hist_ai[idx][0]) {
            cJSON *ma = cJSON_CreateObject();
            cJSON_AddStringToObject(ma, "role", "assistant");
            cJSON_AddStringToObject(ma, "content", s_hist_ai[idx]);
            cJSON_AddItemToArray(arr, ma);
        }
    }
    return arr;
}

static bool chat_is_dialogue(const char *input)
{
    /* agent 循环的注入触发不算对话，不进入上下文记忆 */
    return input && input[0] && strcmp(input, "sensor_injected") != 0;
}

/* 生成动作摘要（含关键参数），用于 WS 回复 / 对话历史 / trace */
static void _action_summary(cJSON *action, char *buf, size_t bufsz)
{
    if (!action || !buf || bufsz == 0) return;
    buf[0] = '\0';
    cJSON *dev = cJSON_GetObjectItem(action, "device");
    cJSON *cmd = cJSON_GetObjectItem(action, "command");
    const char *dn = cJSON_IsString(dev) ? dev->valuestring : "?";
    const char *cn = cJSON_IsString(cmd) ? cmd->valuestring : "?";
    snprintf(buf, bufsz, "%s.%s", dn, cn);
    cJSON *params = cJSON_GetObjectItem(action, "params");
    if (cJSON_IsObject(params)) {
        cJSON *sp = cJSON_GetObjectItem(params, "speed");
        cJSON *po = cJSON_GetObjectItem(params, "position");
        cJSON *br = cJSON_GetObjectItem(params, "brightness");
        size_t used = strlen(buf);
        if (cJSON_IsNumber(sp))
            snprintf(buf + used, bufsz - used, " speed=%d", (int)sp->valuedouble);
        used = strlen(buf);
        if (cJSON_IsNumber(po))
            snprintf(buf + used, bufsz - used, " pos=%d", (int)po->valuedouble);
        used = strlen(buf);
        if (cJSON_IsNumber(br))
            snprintf(buf + used, bufsz - used, " br=%d", (int)br->valuedouble);
    }
}

/* ---- forward decls ---- */
static void _execute_action(const char *dev, const char *cmd, cJSON *params);
static cJSON *_parse_tool_calls(cJSON *llm_response);
static cJSON *_mock_cloud_decision(const char *input, cJSON *sensors);

/* ---- system prompt (Light, no external sensors → use USB-injected data) ---- */
static const char *SYSTEM_PROMPT =
    "You are a smart home controller. "
    "You receive sensor readings from a room. Decide if any device action is needed.\n\n"
    "Available devices:\n"
    "- led: brightness (0-100)\n"
    "- fan: speed (1-3)\n"
    "- curtain: position (0-100, 0=closed, 100=open)\n\n"
    "Rules:\n"
    "- If it's dark and someone is in the room, turn on the light\n"
    "- If CO2 is high, turn on the fan for ventilation\n"
    "- If it's too bright, close the curtain\n"
    "- If the temperature is rising rapidly, turn on the fan\n"
    "- If the user EXPLICITLY asks to control a device (e.g. turn on/off light, "
    "fan, curtain), ALWAYS call the corresponding tool, even if readings seem normal\n"
    "- If the user refers to a device mentioned in the previous conversation "
    "(e.g. 'turn it off', 'dimmer'), use that context\n"
    "- If no action is needed, respond with a brief status ok\n"
    "- Use the available tools when device control is needed\n"
    "- After calling a tool, also reply with a short Chinese summary of the action "
    "you took (e.g. '好的，已打开灯'). Respond in Chinese.";

/* ---- tools definition (OpenAI function-calling format) ---- */
static cJSON *_build_tools(void) {
    cJSON *tools = cJSON_CreateArray();

    /* --- led_control --- */
    cJSON *led = cJSON_CreateObject();
    cJSON_AddStringToObject(led, "type", "function");
    cJSON *led_fn = cJSON_CreateObject();
    cJSON_AddStringToObject(led_fn, "name", "led_control");
    cJSON_AddStringToObject(led_fn, "description", "Control LED light brightness");
    cJSON *led_params = cJSON_CreateObject();
    cJSON_AddStringToObject(led_params, "type", "object");
    cJSON *led_props = cJSON_CreateObject();
    cJSON *led_cmd = cJSON_CreateObject();
    cJSON_AddStringToObject(led_cmd, "type", "string");
    cJSON_AddStringToObject(led_cmd, "description", "on or off");
    cJSON_AddItemToObject(led_props, "command", led_cmd);
    cJSON *led_br = cJSON_CreateObject();
    cJSON_AddStringToObject(led_br, "type", "integer");
    cJSON_AddStringToObject(led_br, "description", "0-100 brightness");
    cJSON_AddItemToObject(led_props, "brightness", led_br);
    cJSON_AddItemToObject(led_params, "properties", led_props);
    cJSON_AddItemToObject(led_fn, "parameters", led_params);
    cJSON_AddItemToObject(led, "function", led_fn);
    cJSON_AddItemToArray(tools, led);

    /* --- fan_control --- */
    cJSON *fan = cJSON_CreateObject();
    cJSON_AddStringToObject(fan, "type", "function");
    cJSON *fan_fn = cJSON_CreateObject();
    cJSON_AddStringToObject(fan_fn, "name", "fan_control");
    cJSON_AddStringToObject(fan_fn, "description", "Control fan speed 1-3");
    cJSON *fan_params = cJSON_CreateObject();
    cJSON_AddStringToObject(fan_params, "type", "object");
    cJSON *fan_props = cJSON_CreateObject();
    cJSON *fan_cmd = cJSON_CreateObject();
    cJSON_AddStringToObject(fan_cmd, "type", "string");
    cJSON_AddStringToObject(fan_cmd, "description", "on or off");
    cJSON_AddItemToObject(fan_props, "command", fan_cmd);
    cJSON *fan_spd = cJSON_CreateObject();
    cJSON_AddStringToObject(fan_spd, "type", "integer");
    cJSON_AddStringToObject(fan_spd, "description", "speed 1-3");
    cJSON_AddItemToObject(fan_props, "speed", fan_spd);
    cJSON_AddItemToObject(fan_params, "properties", fan_props);
    cJSON_AddItemToObject(fan_fn, "parameters", fan_params);
    cJSON_AddItemToObject(fan, "function", fan_fn);
    cJSON_AddItemToArray(tools, fan);

    /* --- curtain_control --- */
    cJSON *cur = cJSON_CreateObject();
    cJSON_AddStringToObject(cur, "type", "function");
    cJSON *cur_fn = cJSON_CreateObject();
    cJSON_AddStringToObject(cur_fn, "name", "curtain_control");
    cJSON_AddStringToObject(cur_fn, "description", "Control curtain position 0-100");
    cJSON *cur_params = cJSON_CreateObject();
    cJSON_AddStringToObject(cur_params, "type", "object");
    cJSON *cur_props = cJSON_CreateObject();
    cJSON *cur_cmd = cJSON_CreateObject();
    cJSON_AddStringToObject(cur_cmd, "type", "string");
    cJSON_AddStringToObject(cur_cmd, "description", "on, off, or set");
    cJSON_AddItemToObject(cur_props, "command", cur_cmd);
    cJSON *cur_pos = cJSON_CreateObject();
    cJSON_AddStringToObject(cur_pos, "type", "integer");
    cJSON_AddStringToObject(cur_pos, "description", "0-100 position");
    cJSON_AddItemToObject(cur_props, "position", cur_pos);
    cJSON_AddItemToObject(cur_params, "properties", cur_props);
    cJSON_AddItemToObject(cur_fn, "parameters", cur_params);
    cJSON_AddItemToObject(cur, "function", cur_fn);
    cJSON_AddItemToArray(tools, cur);

    return tools;
}

/* ---- init ---- */
esp_err_t executor_init(void) {
    s_cloud_q = xQueueCreate(CLOUD_QUEUE_LEN, sizeof(cloud_req_t));
    if (s_cloud_q) {
        xTaskCreate(cloud_worker, "cloud_wkr", 8192, NULL, tskIDLE_PRIORITY + 2, NULL);
    }
    ESP_LOGI(TAG, "Executor ready (v9 async cloud + serial sensor inject)");
    return ESP_OK;
}

/* ---- main entry: sensor → match → local / cloud ---- */
cJSON *executor_handle(const char *input) {
    return executor_handle_hist(input, NULL);
}

cJSON *executor_handle_hist(const char *input, cJSON *client_hist) {
    s_t++;
    cJSON *sensors = sensor_snapshot();
    cJSON *res = cJSON_CreateObject();
    int64_t t0 = esp_timer_get_time();

    int64_t t_match = esp_timer_get_time();
    cJSON *matches = rule_engine_match(sensors);
    lat_record((uint32_t)(esp_timer_get_time() - t_match));
    cJSON *best = rule_engine_resolve(matches);

    if (best) {
        /* ===== LOCAL: rule matched ===== */
        s_l++;
        int64_t lat = esp_timer_get_time() - t0;
        cJSON *act = cJSON_GetObjectItem(best, "action");
        const char *dev = cJSON_GetObjectItem(act, "device")->valuestring;
        const char *cmd = cJSON_GetObjectItem(act, "command")->valuestring;
        cJSON *params = cJSON_GetObjectItem(act, "params");
        _execute_action(dev, cmd, params);
        const char *rid = cJSON_GetObjectItem(best, "id")->valuestring;
        /* v10.5f (D5): defer acceptance so the user can veto within the
         * feedback window (button press = corrected). */
        executor_pending_feedback(rid);
        /* v9: feed sensor values to Welford tracker for this rule */
        rule_engine_welford_feed(rid, sensors);
        rule_engine_actuator_state(dev, cmd);
        /* v8: async LLM confirm (disabled until stack fix verified) */
        /* llm_confirm_enqueue(rid, input ? input : "sensor_injected"); */

        cJSON *trace = cJSON_CreateObject();
        cJSON_AddStringToObject(trace, "id", "trace_local");
        cJSON_AddItemToObject(trace, "sensors", cJSON_Duplicate(sensors, 1));
        cJSON_AddStringToObject(trace, "user_input", input ? input : "");
        /* v8 fix: use nested execution.mode to match Python pipeline */
        cJSON *exec = cJSON_CreateObject();
        cJSON_AddStringToObject(exec, "mode", "local");
        cJSON_AddStringToObject(exec, "rule_id", rid);
        cJSON_AddStringToObject(exec, "result", "success");
        cJSON_AddItemToObject(trace, "execution", exec);
        cJSON_AddNumberToObject(trace, "latency_ms", lat / 1000);
        cJSON_AddItemToObject(trace, "action", cJSON_Duplicate(act, 1));
        trace_logger_record(trace);
        cJSON_Delete(trace);

        cJSON_AddStringToObject(res, "mode", "local");
        cJSON_AddStringToObject(res, "rule_id", rid);
        cJSON_AddStringToObject(res, "reply", "Executed locally by rule");
        cJSON_AddStringToObject(res, "action", dev);
        cJSON_AddNumberToObject(res, "latency_us", lat);
        /* v6.2: 本地规则命中也记入对话上下文（客户端自带历史时不重复记录） */
        if (!client_hist && chat_is_dialogue(input)) {
            char ai_rec[CHAT_MSG_MAX];
            snprintf(ai_rec, sizeof(ai_rec), "本地规则执行: %s.%s", dev, cmd);
            chat_history_add(input, ai_rec);
        }
    } else {
        /* ===== v9 CLOUD: no rule match → async DeepSeek via worker ===== */
        s_c++;
        /* Enqueue to cloud worker (non-blocking) */
        if (s_cloud_q) {
            cloud_req_t req;
            char *sensor_str = cJSON_PrintUnformatted(sensors);
            req.sensors_json = strdup(sensor_str);
            cJSON_free(sensor_str);
            strncpy(req.input, input ? input : "", sizeof(req.input) - 1);
            req.input[sizeof(req.input) - 1] = '\0';
            /* v10: queue full → drop the OLDEST request so the latest
               injection is never silently lost (diagnostics-friendly). */
            if (xQueueSend(s_cloud_q, &req, pdMS_TO_TICKS(100)) != pdTRUE) {
                cloud_req_t stale;
                if (xQueueReceive(s_cloud_q, &stale, 0) == pdTRUE) {
                    if (stale.sensors_json) free(stale.sensors_json);
                }
                if (xQueueSend(s_cloud_q, &req, pdMS_TO_TICKS(100)) != pdTRUE) {
                    ESP_LOGW(TAG, "Cloud queue full, dropping request");
                    free(req.sensors_json);
                }
            }
        }
        /* Return placeholder — cloud worker handles execution */

#if 0  /* === old synchronous cloud path (replaced by async worker above) === */
        char chat_buf[220];
        const char *chat_text = input ? input : "";
        if (strlen(chat_text) > 200) {
            strncpy(chat_buf, chat_text, sizeof(chat_buf) - 1);
            chat_buf[sizeof(chat_buf) - 1] = '\0';
            chat_text = chat_buf;
        }
        /* v6.2: 附带当前设备状态，让"关掉它/调暗/再开一档"这类相对指令可执行 */
        uint8_t st_r = 0, st_g = 0, st_b = 0;
        actuator_get_rgb(&st_r, &st_g, &st_b);
        snprintf(user_msg, sizeof(user_msg),
                 "Current sensor readings:\n%s\n\n"
                 "Current device states: led=%s(%d%%), fan=%s(%d%%), curtain=%s(%d%%)\n\n"
                 "User says: \"%s\"\n\n"
                 "Decide what action to take. Use tools if needed.",
                 sensor_str,
                 st_r > 0 ? "on" : "off", st_r,
                 st_g > 0 ? "on" : "off", st_g,
                 st_b > 0 ? "on" : "off", st_b,
                 chat_text);
        cJSON_free(sensor_str);

        /* Call real LLM（v6.2/6.3: 优先用客户端会话历史，否则用内部记忆） */
        bool hist_owned = false;
        cJSON *hist = client_hist;
        if (!hist) { hist = chat_history_build(); hist_owned = true; }
        cJSON *llm_resp = http_llm_chat_hist(SYSTEM_PROMPT, hist, user_msg, tools);
        if (hist_owned) cJSON_Delete(hist);
        cJSON_Delete(tools);

        int64_t lat = esp_timer_get_time() - t0;
        cJSON *parsed_action = NULL;
        /* v7 fix: model 名必须先拷贝，llm_resp 删除后 model_name 会变成
           悬空指针，写入 trace 造成 JSON 乱码（"deepseek-v4-flas<乱码>"）。 */
        char model_buf[64] = "deepseek-v4-flash";
        const char *model_name = model_buf;
        char reply_buf[512] = "";

        if (llm_resp) {
            /* 记录实际模型名（v6：trace 需要 LLM 信息才能支持 PC 端蒸馏） */
            cJSON *m = cJSON_GetObjectItem(llm_resp, "model");
            if (cJSON_IsString(m) && m->valuestring) {
                strncpy(model_buf, m->valuestring, sizeof(model_buf) - 1);
                model_buf[sizeof(model_buf) - 1] = '\0';
            }
            /* v6.1: 提取 LLM 文字回复供 AI 对话界面显示 */
            cJSON *choices = cJSON_GetObjectItem(llm_resp, "choices");
            cJSON *ch0 = choices ? cJSON_GetArrayItem(choices, 0) : NULL;
            cJSON *msg0 = ch0 ? cJSON_GetObjectItem(ch0, "message") : NULL;
            cJSON *ct = msg0 ? cJSON_GetObjectItem(msg0, "content") : NULL;
            if (cJSON_IsString(ct) && ct->valuestring) {
                strncpy(reply_buf, ct->valuestring, sizeof(reply_buf) - 1);
            }
            parsed_action = _parse_tool_calls(llm_resp);
            cJSON_Delete(llm_resp);
        }

        if (parsed_action) {
            /* LLM returned a tool_call → execute */
            const char *dev = cJSON_GetObjectItem(parsed_action, "device")->valuestring;
            const char *cmd = cJSON_GetObjectItem(parsed_action, "command")->valuestring;
            cJSON *params = cJSON_GetObjectItem(parsed_action, "params");
            _execute_action(dev, cmd, params);
            /* v7: cloud 执行也必须记录 actuator 状态 */
            rule_engine_actuator_state(dev, cmd);
            /* v9: feed sensor values to Welford tracker (device-level) */
            rule_engine_welford_feed(dev, sensors);
        } else {
            /* LLM failed or returned no tool_call → fallback to mock */
            ESP_LOGW(TAG, "LLM call failed or no tool_call, using mock fallback");
            cJSON *mock_act = _mock_cloud_decision(input, sensors);
            if (mock_act) {
                _execute_action(cJSON_GetObjectItem(mock_act, "device")->valuestring,
                               cJSON_GetObjectItem(mock_act, "command")->valuestring,
                               cJSON_GetObjectItem(mock_act, "params"));
                cJSON_Delete(mock_act);
            }
        }

        cJSON *trace = cJSON_CreateObject();
        cJSON_AddStringToObject(trace, "id", "trace_cloud");
        cJSON_AddItemToObject(trace, "sensors", cJSON_Duplicate(sensors, 1));
        cJSON_AddStringToObject(trace, "user_input", input ? input : "");
        cJSON *exec_c = cJSON_CreateObject();
        cJSON_AddStringToObject(exec_c, "mode", "cloud");
        cJSON_AddStringToObject(exec_c, "result", "success");
        cJSON_AddItemToObject(trace, "execution", exec_c);
        cJSON_AddNumberToObject(trace, "latency_ms", lat / 1000);
        cJSON_AddStringToObject(trace, "model", model_name);
        if (parsed_action) cJSON_AddItemToObject(trace, "action", cJSON_Duplicate(parsed_action, 1));
        if (reply_buf[0]) cJSON_AddStringToObject(trace, "reasoning", reply_buf);
        trace_logger_record(trace);
        cJSON_Delete(trace);

        cJSON_AddStringToObject(res, "mode", "cloud");
        if (reply_buf[0]) cJSON_AddStringToObject(res, "reply", reply_buf);
        if (parsed_action) {
            char act_buf[64];
            _action_summary(parsed_action, act_buf, sizeof(act_buf));
            cJSON_AddStringToObject(res, "action", act_buf);
        }
        cJSON_AddNumberToObject(res, "latency_us", lat);

        /* v6.2: 记录对话上下文（"关掉它/再暗一点"依赖前文） */
        if (!client_hist && chat_is_dialogue(input)) {
            char ai_rec[CHAT_MSG_MAX];
            if (reply_buf[0]) {
                snprintf(ai_rec, sizeof(ai_rec), "%.*s", (int)sizeof(ai_rec) - 1, reply_buf);
            } else if (parsed_action) {
                char act_buf[64];
                _action_summary(parsed_action, act_buf, sizeof(act_buf));
                snprintf(ai_rec, sizeof(ai_rec), "action executed: %s", act_buf);
            } else {
                snprintf(ai_rec, sizeof(ai_rec), "no action");
            }
            chat_history_add(input, ai_rec);
        }
        if (parsed_action) cJSON_Delete(parsed_action);
#endif  /* old synchronous cloud path */
    }

    cJSON_Delete(matches);
    cJSON_Delete(sensors);
    /* v9: cloud path returns placeholder — real execution in cloud_worker */
    if (!best) cJSON_AddStringToObject(res, "mode", "cloud");
    return res;
}

/* ---- Parse tool_calls from LLM response ---- */
static cJSON *_parse_tool_calls(cJSON *llm_response) {
    if (!llm_response) return NULL;

    /* Navigate: choices[0].message.tool_calls[0].function */
    cJSON *choices = cJSON_GetObjectItem(llm_response, "choices");
    if (!choices || !cJSON_IsArray(choices)) return NULL;

    cJSON *choice0 = cJSON_GetArrayItem(choices, 0);
    if (!choice0) return NULL;

    cJSON *message = cJSON_GetObjectItem(choice0, "message");
    if (!message) return NULL;

    cJSON *tool_calls = cJSON_GetObjectItem(message, "tool_calls");
    if (!tool_calls || !cJSON_IsArray(tool_calls)) return NULL;

    cJSON *tc0 = cJSON_GetArrayItem(tool_calls, 0);
    if (!tc0) return NULL;

    cJSON *function = cJSON_GetObjectItem(tc0, "function");
    if (!function) return NULL;

    /* v7 fix: check every intermediate pointer before dereference.
       Previous code dereferenced ->valuestring BEFORE the null check. */
    cJSON *fn_name = cJSON_GetObjectItem(function, "name");
    cJSON *fn_args = cJSON_GetObjectItem(function, "arguments");
    if (!fn_name || !fn_args || !cJSON_IsString(fn_name) || !cJSON_IsString(fn_args))
        return NULL;

    const char *name = fn_name->valuestring;
    const char *args_str = fn_args->valuestring;
    if (!name || !args_str) return NULL;

    /* Parse arguments JSON */
    cJSON *args = cJSON_Parse(args_str);
    if (!args) return NULL;

    /* Extract device from function name: "fan_control" → "fan" */
    char device[32] = {0};
    strncpy(device, name, sizeof(device) - 1);
    char *ctrl = strstr(device, "_control");
    if (ctrl) *ctrl = '\0';

    /* v7 fix: command may be non-string (e.g. integer) → don't pass NULL to cJSON_CreateString */
    cJSON *cmd_json = cJSON_GetObjectItem(args, "command");
    const char *command = (cJSON_IsString(cmd_json)) ? cmd_json->valuestring : "on";

    /* Build action object */
    cJSON *action = cJSON_CreateObject();
    cJSON_AddStringToObject(action, "device", device);
    cJSON_AddStringToObject(action, "command", command);
    cJSON *params = cJSON_Duplicate(args, 1);
    cJSON_DeleteItemFromObject(params, "command");  /* remove command from params */
    cJSON_AddItemToObject(action, "params", params);
    ESP_LOGI(TAG, "LLM tool_call: %s.%s", device, command);
    cJSON_Delete(args);
    return action;
}

/* ---- Execute physical action ---- */
static void _execute_action(const char *dev, const char *cmd, cJSON *params) {
    ESP_LOGI(TAG, "EXEC: %s.%s", dev, cmd);
    if (strcmp(dev, "led") == 0) {
        if (strcmp(cmd, "on") == 0) {
            int br = 50;
            if (params) {
                cJSON *b = cJSON_GetObjectItem(params, "brightness");
                if (b) br = (int)b->valuedouble;
            }
            actuator_led_r_set((uint8_t)br);
            actuator_led_g_set((uint8_t)(br / 2));
            actuator_led_b_set(0);
        } else if (strcmp(cmd, "off") == 0) {
            actuator_led_r_set(0); actuator_led_g_set(0); actuator_led_b_set(0);
        }
    } else if (strcmp(dev, "fan") == 0) {
        if (strcmp(cmd, "on") == 0) {
            int speed = 2;
            if (params) {
                cJSON *s = cJSON_GetObjectItem(params, "speed");
                if (s) speed = (int)s->valuedouble;
            }
            actuator_led_g_set((uint8_t)(speed * 33));  /* speed 1-3 → brightness 33-99 */
        } else if (strcmp(cmd, "off") == 0) {
            actuator_led_g_set(0);
        }
    } else if (strcmp(dev, "curtain") == 0) {
        if (strcmp(cmd, "on") == 0 || strcmp(cmd, "set") == 0) {
            int pos = 50;
            if (params) {
                cJSON *p = cJSON_GetObjectItem(params, "position");
                if (p) pos = (int)p->valuedouble;
            }
            /* v6.2 fix: 之前 pos*255/100 会被 actuator 按 0-100 截断，
               导致"窗帘开 30%"实际变成 100%。位置就是 0-100，直接传。 */
            actuator_led_b_set((uint8_t)pos);
        } else if (strcmp(cmd, "off") == 0) {
            actuator_led_b_set(0);
        }
    }
}

/* ---- Cloud decision fallback ----
 * When the LLM call fails (offline / timeout), we do NOT execute anything.
 * Hardcoded thresholds (temp>32, light<100...) contradict the "zero
 * hardcoded thresholds" principle (BUG#2 fix). Instead we wait for the
 * next interaction — the rule engine or a later LLM call will handle it.
 *
 * v6 fix: when LLM returns text without a tool_call (e.g. deepseek-v4-flash
 * occasionally refuses to call functions), parse explicit user commands
 * (开灯/关灯/风扇/窗帘/turn on/turn off) and execute them directly.
 * These are USER COMMANDS, not learned sensor→action mappings — the
 * "zero hardcoded thresholds" principle (BUG#2) is preserved because
 * we never infer device state from sensor values here. */
static cJSON *_mock_cloud_decision(const char *input, cJSON *sensors) {
    if (!input || !sensors) return NULL;
    (void)sensors; /* not inferring anything from sensor values */

    cJSON *a = NULL;
    const char *in = input;

    /* ---- light / 灯 ---- */
    bool want_on = (strstr(in, "开灯") || strstr(in, "开 灯")
        || strstr(in, "turn on") || strstr(in, "turn_on")
        || strstr(in, "light on") || strstr(in, "light_on")
        || strstr(in, "打开灯") || strstr(in, "打开 灯")
        || strstr(in, "亮") || strstr(in, "开光"));
    bool want_off = (strstr(in, "关灯") || strstr(in, "关 灯")
        || strstr(in, "turn off") || strstr(in, "turn_off")
        || strstr(in, "light off") || strstr(in, "light_off")
        || strstr(in, "关闭灯") || strstr(in, "灭灯"));

    if (want_on || want_off) {
        a = cJSON_CreateObject();
        cJSON_AddStringToObject(a, "device", "led");
        cJSON_AddStringToObject(a, "command", want_on ? "on" : "off");
        cJSON *p = cJSON_CreateObject();
        cJSON_AddNumberToObject(p, "brightness", want_on ? 80 : 0);
        cJSON_AddItemToObject(a, "params", p);
        return a;
    }

    /* ---- fan / 风扇 ---- */
    bool fan_on  = strstr(in, "开风扇") || strstr(in, "开 风扇")
        || strstr(in, "热") || strstr(in, "闷")
        || strstr(in, "fan on") || strstr(in, "fan_on");
    bool fan_off = strstr(in, "关风扇") || strstr(in, "关 风扇")
        || strstr(in, "fan off") || strstr(in, "fan_off");

    if (fan_on || fan_off) {
        a = cJSON_CreateObject();
        cJSON_AddStringToObject(a, "device", "fan");
        cJSON_AddStringToObject(a, "command", fan_on ? "on" : "off");
        cJSON *p = cJSON_CreateObject();
        cJSON_AddNumberToObject(p, "speed", 2);
        cJSON_AddItemToObject(a, "params", p);
        return a;
    }

    /* ---- curtain / 窗帘 ---- */
    if (strstr(in, "窗帘") || strstr(in, "curtain")) {
        bool close = strstr(in, "关") || strstr(in, "off") || strstr(in, "close");
        a = cJSON_CreateObject();
        cJSON_AddStringToObject(a, "device", "curtain");
        cJSON_AddStringToObject(a, "command", "set");
        cJSON *p = cJSON_CreateObject();
        cJSON_AddNumberToObject(p, "position", close ? 0 : 80);
        cJSON_AddItemToObject(a, "params", p);
        return a;
    }

    return NULL;
}

void executor_get_stats(int *t, int *l, int *c) { *t = s_t; *l = s_l; *c = s_c; }

/* ---- v9: async cloud worker (non-blocking LLM execution) ---- */
static void cloud_worker(void *pv) {
    cloud_req_t req;
    while (1) {
        if (xQueueReceive(s_cloud_q, &req, portMAX_DELAY) != pdTRUE) continue;
        cJSON *sensors = cJSON_Parse(req.sensors_json);
        if (!sensors) { free(req.sensors_json); continue; }

        cJSON *tools = _build_tools();
        char user_msg[1024];
        uint8_t st_r, st_g, st_b;
        actuator_get_rgb(&st_r, &st_g, &st_b);
        snprintf(user_msg, sizeof(user_msg),
                 "Sensors:\n%s\nDevices: led=%s(%d%%) fan=%s(%d%%) curtain=%s(%d%%)\nUser: \"%s\"",
                 req.sensors_json,
                 st_r > 0 ? "on" : "off", st_r,
                 st_g > 0 ? "on" : "off", st_g,
                 st_b > 0 ? "on" : "off", st_b,
                 req.input);

        cJSON *llm_resp = http_llm_chat_hist(SYSTEM_PROMPT, NULL, user_msg, tools);
        cJSON_Delete(tools);

        cJSON *parsed_action = NULL;
        if (llm_resp) {
            parsed_action = _parse_tool_calls(llm_resp);
            cJSON_Delete(llm_resp);
        }
        if (parsed_action) {
            const char *dev = cJSON_GetObjectItem(parsed_action, "device")->valuestring;
            const char *cmd = cJSON_GetObjectItem(parsed_action, "command")->valuestring;
            _execute_action(dev, cmd, cJSON_GetObjectItem(parsed_action, "params"));
            rule_engine_actuator_state(dev, cmd);
            rule_engine_welford_feed(dev, sensors);
            cJSON_Delete(parsed_action);
        }
        cJSON_Delete(sensors);
        free(req.sensors_json);
    }
}
