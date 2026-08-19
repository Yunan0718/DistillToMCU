/*
 * DistillToMCU — WS2812 RGB LED Driver (RMT)
 * ==========================================
 * ESP32-S3-DevKitC-1 板载 RGB LED 为 WS2812 可寻址灯珠（GPIO48），
 * 不是普通 PWM LED。本驱动用 ESP-IDF 内置 RMT 外设发送 800kHz 时序，
 * 不依赖任何第三方组件。
 *
 * 兼容性：若板子/外接灯条使用其他引脚，通过 menuconfig 修改
 * CONFIG_MIMI_LED_WS2812_GPIO 即可，无需改代码。
 */

#include "ws2812.h"
#include "driver/rmt_tx.h"
#include "driver/rmt_encoder.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include <string.h>

#define TAG "ws2812"

static rmt_channel_handle_t s_tx = NULL;
static rmt_encoder_handle_t s_enc = NULL;
static uint8_t s_pixels[3] = {0, 0, 0};  /* G, R, B 顺序（WS2812 协议） */
static bool s_ready = false;

esp_err_t ws2812_init(int gpio_num)
{
    rmt_tx_channel_config_t tx_cfg = {
        .gpio_num = gpio_num,
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,  /* 10 MHz → 100ns/bit tick */
        .mem_block_symbols = 64,
        .trans_queue_depth = 4,
        .intr_priority = 0,
    };
    esp_err_t r = rmt_new_tx_channel(&tx_cfg, &s_tx);
    if (r != ESP_OK) {
        ESP_LOGE(TAG, "rmt_new_tx_channel failed: %s", esp_err_to_name(r));
        return r;
    }

    /* WS2812 时序（10MHz tick）：
       T0H=0.4us(4), T0L=0.85us(8-9)
       T1H=0.85us(8), T1L=0.4us(4) */
    rmt_bytes_encoder_config_t enc_cfg = {
        .bit0 = {
            .duration0 = 4,  /* T0H */
            .level0 = 1,
            .duration1 = 8,  /* T0L */
            .level1 = 0,
        },
        .bit1 = {
            .duration0 = 8,  /* T1H */
            .level0 = 1,
            .duration1 = 4,  /* T1L */
            .level1 = 0,
        },
        .flags = {
            .msb_first = 1,  /* WS2812 高位在前 */
        },
    };
    r = rmt_new_bytes_encoder(&enc_cfg, &s_enc);
    if (r != ESP_OK) {
        ESP_LOGE(TAG, "rmt_new_bytes_encoder failed: %s", esp_err_to_name(r));
        return r;
    }

    r = rmt_enable(s_tx);
    if (r != ESP_OK) {
        ESP_LOGE(TAG, "rmt_enable failed: %s", esp_err_to_name(r));
        return r;
    }
    s_ready = true;
    ESP_LOGI(TAG, "WS2812 ready on GPIO%d", gpio_num);
    return ESP_OK;
}

void ws2812_set_rgb(uint8_t r, uint8_t g, uint8_t b)
{
    if (!s_ready) return;
    s_pixels[0] = g;
    s_pixels[1] = r;
    s_pixels[2] = b;
    rmt_transmit_config_t tx_cfg = {
        .loop_count = 0,
    };
    rmt_transmit(s_tx, s_enc, s_pixels, 3, &tx_cfg);
    /* v6: 不再阻塞等待（rmt_tx_wait_all_done 在此驱动版本上每次报 flush timeout）。
       s_pixels 是静态缓冲，调用间隔远大于 30us 的发送时长，无覆盖风险。 */
    vTaskDelay(pdMS_TO_TICKS(1));
}

void ws2812_off(void)
{
    ws2812_set_rgb(0, 0, 0);
}
