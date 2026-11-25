/*
 * DistillToMCU — Rule Engine Header
 */

#pragma once
#include "esp_err.h"
#include "cJSON.h"

esp_err_t rule_engine_init(void);

/* Match all rules against current sensor snapshot. Returns cJSON array. */
cJSON *rule_engine_match(cJSON *sensors);

/* Resolve conflicts: v8 PMS (Perturbed-Mean Selection) bandit + cooldown mutex. */
cJSON *rule_engine_resolve(cJSON *matches);

/* Update rule stats after execution (accept/corrected) + PMS alpha/beta */
void rule_engine_update_on_exec(const char *rule_id, const char *feedback_type);

/* v8: periodic time-decay freshness: degrade stale rules, mark dirty */
void rule_engine_update_freshness(void);

/* Track actuator state for mutex */
void rule_engine_actuator_state(const char *device, const char *state);

/* Get all rules as JSON array (for CLI 'rules' command) */
cJSON *rule_engine_get_all_rules(void);

/* v9: Online Welford feed — accumulate sensor stats per rule/device */
void rule_engine_welford_feed(const char *rule_or_dev, cJSON *sensors);

/* v9: Online interval update — recompute rule conditions from Welford stats */
void rule_engine_online_update_intervals(void);
