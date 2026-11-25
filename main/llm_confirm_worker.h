/*
 * DistillToMCU — LLM Confirm Worker Header
 */
#pragma once
#include "esp_err.h"
esp_err_t llm_confirm_worker_init(void);
esp_err_t llm_confirm_enqueue(const char *rule_id, const char *user_input);
