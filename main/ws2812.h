#pragma once

#include "esp_err.h"
#include <stdint.h>

/* 初始化 RMT 发送通道（GPIO 可配置，默认 48 = DevKitC-1 板载 RGB） */
esp_err_t ws2812_init(int gpio_num);

/* 设置 RGB 颜色（0-255）并立即发送 */
void ws2812_set_rgb(uint8_t r, uint8_t g, uint8_t b);

/* 熄灭 */
void ws2812_off(void);
