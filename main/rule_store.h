/*
 * DistillToMCU — Rule Store Header
 */

#pragma once
#include "esp_err.h"
#include "cJSON.h"

esp_err_t rule_store_init(void);
cJSON   *rule_store_get_all(void);
void     rule_store_lock(void);       /* v7: mutex for concurrent access safety */
void     rule_store_unlock(void);
void     rule_store_add(cJSON *rule_obj);
void     rule_store_update(const char *rule_id, cJSON *updated_obj);
void     rule_store_mark_dirty(void);
void     rule_store_force_persist(void);
void     rule_store_persist(void);
