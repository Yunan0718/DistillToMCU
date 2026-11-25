/*
 * DistillToMCU — HTTP Client
 * ==========================
 * DeepSeek API wrapper. Handles TLS, JSON, tool_call parsing.
 */

#pragma once
#include "esp_err.h"
#include "cJSON.h"

esp_err_t http_client_init(void);

/*
 * Send a chat completion request to DeepSeek.
 * tools_json: NULL or cJSON array of tool definitions (OpenAI format).
 * Returns: cJSON object (the full API response body), caller must cJSON_Delete.
 */
cJSON *http_llm_chat(const char *system_prompt, const char *user_msg,
                     cJSON *tools_json);

/*
 * v6.2: 多轮对话版本。history_msgs 为 [{role, content}, ...]，
 * 插入 system 与当前 user 之间；可为 NULL。调用方负责释放 history_msgs。
 */
cJSON *http_llm_chat_hist(const char *system_prompt, cJSON *history_msgs,
                          const char *user_msg, cJSON *tools_json);
