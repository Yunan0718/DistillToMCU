#include "actuator.h"
#include "driver/ledc.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "mimi_config.h"
#include "ws2812.h"

#define TAG "actuator"
#define LEDC_TIMER   LEDC_TIMER_0
#define LEDC_MODE    LEDC_LOW_SPEED_MODE
#define LEDC_FREQ    5000
#define LEDC_RES     LEDC_TIMER_13_BIT

esp_err_t actuator_init(void)
{
    ledc_timer_config_t t = {.speed_mode = LEDC_MODE, .duty_resolution = LEDC_RES,
                             .timer_num = LEDC_TIMER, .freq_hz = LEDC_FREQ,
                             .clk_cfg = LEDC_AUTO_CLK};
    ESP_ERROR_CHECK(ledc_timer_config(&t));

    ledc_channel_config_t ch[3];
    int pins[3] = {PIN_LED_R, PIN_LED_G, PIN_LED_B};
    for (int i = 0; i < 3; i++) {
        ch[i] = (ledc_channel_config_t){
            .gpio_num = pins[i], .speed_mode = LEDC_MODE,
            .channel = LEDC_CHANNEL_0 + i, .timer_sel = LEDC_TIMER,
            .duty = 0, .hpoint = 0,
        };
        ESP_ERROR_CHECK(ledc_channel_config(&ch[i]));
    }

#ifdef CONFIG_MIMI_LED_WS2812_ENABLE
    /* 板载 WS2812 RGB（DevKitC-1 = GPIO48） */
    esp_err_t wr = ws2812_init(PIN_LED_WS2812);
    if (wr != ESP_OK) {
        ESP_LOGW(TAG, "WS2812 init failed (%s), continuing with LEDC only",
                 esp_err_to_name(wr));
    } else {
        /* v6: 论文实验不需要板载 RGB——开机立即发熄灭帧，清掉灯珠锁存的旧颜色。
           之后只有 menuconfig 打开 MIMI_LED_WS2812_MIRROR 才跟随执行器。 */
        ws2812_off();
    }
#endif

    ESP_LOGI(TAG, "LED PWM ready: R(%d) G(%d) B(%d)", PIN_LED_R, PIN_LED_G, PIN_LED_B);
    return ESP_OK;
}

static void _set(ledc_channel_t ch, uint8_t pct) {
    if (pct > 100) pct = 100;
    uint32_t raw = (uint32_t)((pct * 8191) / 100);
    esp_err_t r1 = ledc_set_duty(LEDC_MODE, ch, raw);
    esp_err_t r2 = ledc_update_duty(LEDC_MODE, ch);
    if (r1 != ESP_OK || r2 != ESP_OK) {
        ESP_LOGE(TAG, "ledc ch%d set %d%%(%lu) failed: %s / %s",
                 (int)ch, (int)pct, (unsigned long)raw,
                 esp_err_to_name(r1), esp_err_to_name(r2));
    }
}

static uint8_t s_r = 0, s_g = 0, s_b = 0;
static bool s_save_enabled = false;  /* 自检期间不写 NVS，恢复流程启动后开启 */

/* ---- LED 状态 NVS 持久化 ----
 * 关闭 USB-JTAG 串口会触发芯片复位（USB_UART_CHIP_RESET），
 * 复位后自检结束灯会全灭。把亮度存 NVS，开机自检后恢复，
 * 保证"断开串口灯不灭"（v6 修复）。 */

void actuator_led_save_state(void)
{
    nvs_handle_t h;
    if (nvs_open("led_state", NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_u8(h, "r", s_r);
    nvs_set_u8(h, "g", s_g);
    nvs_set_u8(h, "b", s_b);
    nvs_commit(h);
    nvs_close(h);
}

void actuator_led_restore_state(void)
{
    s_save_enabled = true;
    nvs_handle_t h;
    if (nvs_open("led_state", NVS_READONLY, &h) != ESP_OK) return;
    uint8_t r = 0, g = 0, b = 0;
    nvs_get_u8(h, "r", &r);
    nvs_get_u8(h, "g", &g);
    nvs_get_u8(h, "b", &b);
    nvs_close(h);
    if (r || g || b) {
        ESP_LOGI(TAG, "Restore LED state: R=%d G=%d B=%d", r, g, b);
        actuator_led_r_set(r);
        actuator_led_g_set(g);
        actuator_led_b_set(b);
    }
}

static void _sync_ws2812(void)
{
#ifdef CONFIG_MIMI_LED_WS2812_MIRROR
    ws2812_set_rgb(s_r * 255 / 100, s_g * 255 / 100, s_b * 255 / 100);
#endif
}

/* v7: only save to NVS when value actually changed (no-op writes waste Flash erase cycles) */
void actuator_led_r_set(uint8_t p) { if (p == s_r) return; s_r = p; _set(LEDC_CHANNEL_0, p); _sync_ws2812(); if (s_save_enabled) actuator_led_save_state(); }
void actuator_led_g_set(uint8_t p) { if (p == s_g) return; s_g = p; _set(LEDC_CHANNEL_1, p); _sync_ws2812(); if (s_save_enabled) actuator_led_save_state(); }
void actuator_led_b_set(uint8_t p) { if (p == s_b) return; s_b = p; _set(LEDC_CHANNEL_2, p); _sync_ws2812(); if (s_save_enabled) actuator_led_save_state(); }

void actuator_get_rgb(uint8_t *r, uint8_t *g, uint8_t *b)
{
    if (r) *r = s_r;
    if (g) *g = s_g;
    if (b) *b = s_b;
}

void actuator_led_test_sequence(void)
{
    /* 上电自检：红→绿→蓝→白→灭，LEDC 与 WS2812 同时驱动，
       保证无论板载 RGB 还是外接 LED 至少一种可见。 */
    ESP_LOGI(TAG, "LED self-test: R->G->B->W->off");
    actuator_led_r_set(100); vTaskDelay(pdMS_TO_TICKS(250));
    actuator_led_r_set(0);   vTaskDelay(pdMS_TO_TICKS(100));
    actuator_led_g_set(100); vTaskDelay(pdMS_TO_TICKS(250));
    actuator_led_g_set(0);   vTaskDelay(pdMS_TO_TICKS(100));
    actuator_led_b_set(100); vTaskDelay(pdMS_TO_TICKS(250));
    actuator_led_b_set(0);   vTaskDelay(pdMS_TO_TICKS(100));
    actuator_led_r_set(100); actuator_led_g_set(100); actuator_led_b_set(100);
    vTaskDelay(pdMS_TO_TICKS(250));
    actuator_led_r_set(0); actuator_led_g_set(0); actuator_led_b_set(0);
    ESP_LOGI(TAG, "LED self-test done");
}

void actuator_blink_led(int idx, int on_ms, int off_ms) {
    ledc_channel_t ch = LEDC_CHANNEL_0 + idx;
    _set(ch, 100); vTaskDelay(pdMS_TO_TICKS(on_ms));
    _set(ch, 0);   vTaskDelay(pdMS_TO_TICKS(off_ms));
}

/* Debug: direct GPIO digital output (bypasses LEDC PWM).
 * Use gpio_config() with explicit mode — gpio_set_direction alone can
 * leave the output path disabled (OutputEn=0) if the pin was previously
 * held by a peripheral. gpio_config is the robust way. */
esp_err_t actuator_gpio_direct(int pin, bool level)
{
    gpio_reset_pin(pin);
    gpio_config_t cfg = {
        .pin_bit_mask = (1ULL << pin),
        .mode = GPIO_MODE_OUTPUT,           /* push-pull output */
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    esp_err_t r1 = gpio_config(&cfg);
    esp_err_t r2 = gpio_set_level(pin, level ? 1 : 0);
    ESP_LOGI(TAG, "gpio_direct: pin %d -> %s (cfg=%s lvl=%s)",
             pin, level ? "HIGH" : "LOW",
             esp_err_to_name(r1), esp_err_to_name(r2));
    return r1 != ESP_OK ? r1 : r2;
}

/* Breathing LED removed — AI controls LEDs via tool_calls at runtime. */
