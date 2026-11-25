/*
 * DistillToMCU — Actuator Manager
 */

#pragma once
#include "esp_err.h"
#include <stdint.h>
#include <stdbool.h>

esp_err_t actuator_init(void);

void actuator_led_r_set(uint8_t pct);
void actuator_led_g_set(uint8_t pct);
void actuator_led_b_set(uint8_t pct);

void actuator_blink_led(int led_idx, int on_ms, int off_ms);
void actuator_led_test_sequence(void);
void actuator_led_save_state(void);
void actuator_led_restore_state(void);
void actuator_get_rgb(uint8_t *r, uint8_t *g, uint8_t *b);
esp_err_t actuator_gpio_direct(int pin, bool level);
