/*
 * DistillToMCU — Execution Engine Header
 */

#pragma once
#include "esp_err.h"
#include "cJSON.h"

esp_err_t executor_init(void);
cJSON   *executor_handle(const char *user_input);
/* v6.3: 带客户端会话历史的对话入口（history: [{role,content},...]，可为 NULL）
   用于多会话 AI 对话——会话上下文由仪表盘维护，固件按需使用。 */
cJSON   *executor_handle_hist(const char *user_input, cJSON *history);
void     executor_get_stats(int *total, int *local, int *cloud);
/* v10.5f: real match-latency telemetry (p50/p95/p99 in microseconds) */
void     executor_latstats(int *n, uint32_t *mean_us, uint32_t *p50,
                           uint32_t *p95, uint32_t *p99, uint32_t *max_us);
/* v10.5f (D5): user-override feedback window */
void     executor_pending_feedback(const char *rid);
void     executor_feedback_poll(void);
uint32_t executor_get_override_count(void);
