/*
 * DistillToMCU —Serial CLI (v2: sensor inject + rules)
 * ======================================================
 * ESP-IDF v5.2 console REPL with sensor data injection support.
 *
 * Commands:
 *   led <r|g|b|off> [0-100]  —direct LED control
 *   say <message>             —trigger agent execution loop
 *   sensor <json>             —inject sensor data from PC
 *   stats                     —show execution counters
 *   rules                     —show active rules
 *   heap                      —PSRAM/heap usage
 *   reboot                    —reboot device
 */

#include "serial_cli.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_console.h"
#include <string.h>
#include <stdlib.h>
#include "executor.h"
#include "actuator.h"
#include "sensor.h"
#include "rule_engine.h"
#include "rule_store.h"
#include "ws2812.h"
#include "cJSON.h"

#define TAG "cli"

/* ---- led ---- */
static int cmd_led(int argc, char **argv) {
    if (argc < 2) { printf("Usage: led <r|g|b|off> [0-100]\n"); return 1; }
    const char *c = argv[1];
    int br = (argc >= 3) ? atoi(argv[2]) : 50;
    if (br < 0) br = 0;
    if (br > 100) br = 100;
    if (!strcmp(c, "r"))      actuator_led_r_set((uint8_t)br);
    else if (!strcmp(c, "g")) actuator_led_g_set((uint8_t)br);
    else if (!strcmp(c, "b")) actuator_led_b_set((uint8_t)br);
    else if (!strcmp(c, "off")) {
        actuator_led_r_set(0); actuator_led_g_set(0); actuator_led_b_set(0);
    }
    printf("LED %s -> %d%%\n", c, br);
    return 0;
}

/* ---- ledtest: LED 自检序列（红→绿→蓝→白→灭） ---- */
static int cmd_ledtest(int argc, char **argv) {
    printf("Running LED self-test (R->G->B->W->off)...\n");
    actuator_led_test_sequence();
    printf("LED self-test done. If you saw colors, LED is working.\n");
    return 0;
}

/* ---- breathe: breathing LED ---- */


/* ---- ws: 直接控制板载 WS2812 RGB ---- */
static int cmd_ws(int argc, char **argv) {
    if (argc < 4) { printf("Usage: ws <r> <g> <b>  (0-255)\n"); return 1; }
    int r = atoi(argv[1]), g = atoi(argv[2]), b = atoi(argv[3]);
    if (r < 0 || r > 255 || g < 0 || g > 255 || b < 0 || b > 255) {
        printf("Invalid value (0-255)\n"); return 1;
    }
    ws2812_set_rgb((uint8_t)r, (uint8_t)g, (uint8_t)b);
    printf("WS2812 RGB set: (%d,%d,%d)\n", r, g, b);
    return 0;
}

/* ---- rule add <json>: 导入一条规则（v6 新增，打通 MCU 规则获取路径） ---- */
static int cmd_rule_add(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: rule add <json>\n");
        printf("Example: rule add {\"conditions\":[{\"sensor\":\"temperature\",\"op\":\"gt\",\"value\":30}],"
               "\"action\":{\"device\":\"fan\",\"command\":\"on\",\"params\":{\"speed\":2}}}\n");
        return 1;
    }
    if (strcmp(argv[1], "save") == 0) {
        rule_store_force_persist();
        printf("[OK] Rules persisted to flash\n");
        return 0;
    }
    if (strcmp(argv[1], "add") != 0) {
        printf("Unknown subcommand '%s'. Use: rule add <json> | rule save\n", argv[1]);
        return 1;
    }
    if (argc < 3) {
        printf("Usage: rule add <json>\n");
        return 1;
    }
    char json_str[1024] = {0};
    for (int i = 2; i < argc; i++) {
        if (i > 2) strncat(json_str, " ", sizeof(json_str) - strlen(json_str) - 1);
        strncat(json_str, argv[i], sizeof(json_str) - strlen(json_str) - 1);
    }
    cJSON *rule = cJSON_Parse(json_str);
    if (!rule) {
        printf("ERROR: Invalid JSON: %s\n",
               cJSON_GetErrorPtr() ? cJSON_GetErrorPtr() : "parse error");
        return 1;
    }
    cJSON *act = cJSON_GetObjectItem(rule, "action");
    cJSON *conds = cJSON_GetObjectItem(rule, "conditions");
    if (!cJSON_IsObject(act) || !cJSON_IsArray(conds) || cJSON_GetArraySize(conds) == 0) {
        printf("ERROR: rule needs non-empty \"conditions\" array and \"action\" object\n");
        cJSON_Delete(rule);
        return 1;
    }
    /* v7: validate action.device and action.command (WS path already does this,
       CLI didn't → crash on malformed rules from serial / boot-load) */
    cJSON *act_dev = cJSON_GetObjectItem(act, "device");
    cJSON *act_cmd = cJSON_GetObjectItem(act, "command");
    if (!cJSON_IsString(act_dev) || !act_dev->valuestring || !act_dev->valuestring[0]) {
        printf("ERROR: action.device must be a non-empty string\n");
        cJSON_Delete(rule);
        return 1;
    }
    if (!cJSON_IsString(act_cmd) || !act_cmd->valuestring || !act_cmd->valuestring[0]) {
        printf("ERROR: action.command must be a non-empty string\n");
        cJSON_Delete(rule);
        return 1;
    }
    /* 默认字段：id/state/safety_level/confidence */
    if (!cJSON_GetObjectItem(rule, "id")) {
        static int s_rule_seq = 0;
        char rid[24];
        snprintf(rid, sizeof(rid), "rule_%04d", ++s_rule_seq);
        cJSON_AddStringToObject(rule, "id", rid);
    }
    if (!cJSON_GetObjectItem(rule, "state"))
        cJSON_AddStringToObject(rule, "state", "candidate");
    if (!cJSON_GetObjectItem(rule, "safety_level"))
        cJSON_AddNumberToObject(rule, "safety_level", 1);
    if (!cJSON_GetObjectItem(rule, "confidence"))
        cJSON_AddNumberToObject(rule, "confidence", 0.7);

    rule_store_add(rule);
    rule_store_force_persist();
    const char *rid = cJSON_GetObjectItem(rule, "id")->valuestring;
    printf("[OK] Rule added: %s (%d conditions)\n",
           rid, cJSON_GetArraySize(conds));
    cJSON_Delete(rule);
    return 0;
}

/* ---- say: agent loop trigger ----
 * Runs executor_handle() in a dedicated task with a large stack,
 * because LLM calls + JSON parsing exceed the REPL task stack (4KB).
 */
static char s_say_msg[256];
static TaskHandle_t s_say_task = NULL;

static void say_worker(void *pv) {
    const char *msg = (const char *)pv;
    cJSON *res = executor_handle(msg);
    if (res) {
        printf("[%s] latency=%d us\n",
               cJSON_GetObjectItem(res, "mode")->valuestring,
               (int)cJSON_GetObjectItem(res, "latency_us")->valuedouble);
        if (!strcmp(cJSON_GetObjectItem(res, "mode")->valuestring, "local"))
            printf("  rule: %s\n", cJSON_GetObjectItem(res, "rule_id")->valuestring);
        cJSON *rp = cJSON_GetObjectItem(res, "reply");
        if (rp && cJSON_IsString(rp) && rp->valuestring)
            printf("  reply: %.200s\n", rp->valuestring);
        cJSON_Delete(res);
    } else {
        printf("[ERROR] executor returned NULL\n");
    }
    /* free stack, clear handle */
    s_say_task = NULL;
    vTaskDelete(NULL);
}

static int cmd_say(int argc, char **argv) {
    if (argc < 2) { printf("Usage: say <message>\n"); return 1; }
    if (s_say_task) { printf("[BUSY] previous say still running\n"); return 1; }
    s_say_msg[0] = 0;
    for (int i = 1; i < argc; i++) {
        if (i > 1) strncat(s_say_msg, " ", sizeof(s_say_msg) - strlen(s_say_msg) - 1);
        strncat(s_say_msg, argv[i], sizeof(s_say_msg) - strlen(s_say_msg) - 1);
    }
    printf("[OK] queued: %.60s\n", s_say_msg);
    xTaskCreate(say_worker, "say", 16 * 1024, s_say_msg, 5, &s_say_task);
    return 0;
}

/* ---- gpio: direct digital control for debugging ---- */
static int cmd_gpio(int argc, char **argv) {
    if (argc < 3) { printf("Usage: gpio <pin> <0|1>\n"); return 1; }
    int pin = atoi(argv[1]);
    int lvl = atoi(argv[2]);
    if (pin < 0 || pin > 48) { printf("Invalid pin %d\n", pin); return 1; }
    esp_err_t r = actuator_gpio_direct(pin, lvl != 0);
    printf("gpio %d -> %d (%s)\n", pin, lvl, esp_err_to_name(r));
    return 0;
}

/* ---- sensor: inject JSON from PC ---- */
static int cmd_sensor(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: sensor <json>\n");
        printf("Example: sensor {\"temperature\":23.5,\"humidity\":45,\"light\":300,\"co2\":800}\n");
        return 1;
    }
    /* Reassemble JSON string (may contain spaces, braces, etc.) */
    char json_str[512] = {0};
    for (int i = 1; i < argc; i++) {
        if (i > 1) strncat(json_str, " ", sizeof(json_str) - strlen(json_str) - 1);
        strncat(json_str, argv[i], sizeof(json_str) - strlen(json_str) - 1);
    }

    cJSON *data = cJSON_Parse(json_str);
    if (!data) {
        sensor_note_parse_fail();
        printf("ERROR: Invalid JSON: %s\n", cJSON_GetErrorPtr() ? cJSON_GetErrorPtr() : "parse error");
        return 1;
    }

    sensor_inject(data);  /* takes ownership */
    printf("[OK] Sensor data injected (%d fields)\n",
           cJSON_GetArraySize(data) > 0 ? cJSON_GetArraySize(data) : 0);
    return 0;
}

/* ---- stats ---- */
extern TaskHandle_t agent_task_handle(void);

static int cmd_stats(int argc, char **argv) {
    int t, l, c;
    executor_get_stats(&t, &l, &c);
    int ar = t > 0 ? (int)(l * 100.0 / t + 0.5) : 0;

    printf("{\"total\":%d,\"local\":%d,\"cloud\":%d,\"ar_pct\":%d", t, l, c, ar);

    /* v8: hardware metrics */
    printf(",\"free_sram\":%u,\"free_psram\":%u",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    /* rule store stats */
    cJSON *rules = rule_engine_get_all_rules();
    int n = cJSON_GetArraySize(rules);
    int active = 0, verified = 0, degraded = 0, retired = 0, cand = 0;
    for (int i = 0; i < n; i++) {
        cJSON *r = cJSON_GetArrayItem(rules, i);
        cJSON *st = cJSON_GetObjectItem(r, "state");
        const char *s = (st && cJSON_IsString(st)) ? st->valuestring : "";
        if (!strcmp(s, "active")) active++;
        else if (!strcmp(s, "verified")) verified++;
        else if (!strcmp(s, "degraded")) degraded++;
        else if (!strcmp(s, "retired")) retired++;
        else cand++;
    }
    printf(",\"rules_total\":%d,\"rules_active\":%d,\"rules_verified\":%d", n, active, verified);
    printf(",\"rules_candidate\":%d,\"rules_degraded\":%d", cand, degraded);

    /* agent loop stack watermark */
    extern TaskHandle_t agent_task_handle(void);
    TaskHandle_t ah = agent_task_handle();
    if (ah) printf(",\"agent_stack_free\":%u", (unsigned)uxTaskGetStackHighWaterMark(ah));

    /* v10: injection diagnostics */
    uint32_t inj_ok = 0, inj_drop = 0, inj_fail = 0;
    sensor_get_inject_stats(&inj_ok, &inj_drop, &inj_fail);
    printf(",\"inject_ok\":%u,\"inject_dropped\":%u,\"inject_parse_fail\":%u",
           (unsigned)inj_ok, (unsigned)inj_drop, (unsigned)inj_fail);

    /* v10.5f (D5): user override (rule veto) count */
    printf(",\"override_count\":%u",
           (unsigned)executor_get_override_count());

    printf("}\n");
    return 0;
}

/* ---- v10.5f: real match-latency distribution ---- */
static int cmd_latstats(int argc, char **argv) {
    int n;
    uint32_t mean, p50, p95, p99, mx;
    executor_latstats(&n, &mean, &p50, &p95, &p99, &mx);
    printf("{\"match_lat\":{\"n\":%d,\"mean_us\":%u,\"p50_us\":%u,"
           "\"p95_us\":%u,\"p99_us\":%u,\"max_us\":%u}}\n",
           n, (unsigned)mean, (unsigned)p50, (unsigned)p95,
           (unsigned)p99, (unsigned)mx);
    return 0;
}

/* ---- rules: show active rules ---- */
static int cmd_rules(int argc, char **argv) {
    cJSON *rules = rule_engine_get_all_rules();
    if (!rules) {
        printf("No rules loaded.\n");
        return 0;
    }
    int count = cJSON_GetArraySize(rules);
    printf("%d rules:\n", count);
    for (int i = 0; i < count && i < 20; i++) {
        cJSON *r = cJSON_GetArrayItem(rules, i);
        if (!r) continue;
        /* null-safe field access: a rule with missing state/action
           fields must not crash the CLI (BUG#1 fix) */
        cJSON *idj = cJSON_GetObjectItem(r, "id");
        cJSON *stj = cJSON_GetObjectItem(r, "state");
        cJSON *act = cJSON_GetObjectItem(r, "action");
        cJSON *dvj = (act && cJSON_IsObject(act)) ? cJSON_GetObjectItem(act, "device") : NULL;
        cJSON *cmj = (act && cJSON_IsObject(act)) ? cJSON_GetObjectItem(act, "command") : NULL;
        const char *id   = idj && cJSON_IsString(idj) ? idj->valuestring : "?";
        const char *state = stj && cJSON_IsString(stj) ? stj->valuestring : "?";
        const char *dev  = dvj && cJSON_IsString(dvj) ? dvj->valuestring : "?";
        const char *cmd  = cmj && cJSON_IsString(cmj) ? cmj->valuestring : "?";
        printf("  %s [%s] %s.%s\n", id, state, dev, cmd);
    }
    if (count > 20) printf("  ... and %d more\n", count - 20);
    return 0;
}

/* ---- heap ---- */
static int cmd_heap(int argc, char **argv) {
    printf("Free heap:  %d KB\n", (int)(esp_get_free_heap_size() / 1024));
    printf("Free PSRAM: %d KB\n", (int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    printf("Active injection: %s\n", sensor_has_injection() ? "YES" : "no");
    return 0;
}

/* ---- reboot ---- */
static int cmd_reboot(int argc, char **argv) {
    printf("Rebooting...\n"); esp_restart(); return 0;
}

void serial_cli_start(void) {
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t rc = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    rc.prompt = "d2mcu> ";
    rc.max_cmdline_length = 512;  /* increased for sensor JSON */

#if defined(CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG)
    esp_console_dev_usb_serial_jtag_config_t uc =
        ESP_CONSOLE_DEV_USB_SERIAL_JTAG_CONFIG_DEFAULT();
#else
    esp_console_dev_uart_config_t uc = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
#endif

    esp_console_cmd_t cmds[] = {
        {.command = "led",    .help = "Set LED <r|g|b|off> [0-100]",               .func = cmd_led},
        {.command = "ledtest", .help = "Run LED self-test sequence",               .func = cmd_ledtest},
                {.command = "ws",     .help = "Set WS2812 RGB <r> <g> <b> (0-255)",        .func = cmd_ws},
        {.command = "rule",   .help = "rule add <json> | rule save",               .func = cmd_rule_add},
        {.command = "gpio",   .help = "Debug: set GPIO <pin> <0|1> directly",     .func = cmd_gpio},
        {.command = "say",    .help = "Send message to agent (triggers full loop)",.func = cmd_say},
        {.command = "sensor", .help = "Inject sensor JSON from PC",                 .func = cmd_sensor},
        {.command = "stats",  .help = "Show execution stats",                       .func = cmd_stats},
        {.command = "latstats", .help = "Match-latency distribution (p50/p95/p99)", .func = cmd_latstats},
        {.command = "rules",  .help = "Show active rules",                          .func = cmd_rules},
        {.command = "heap",   .help = "Show heap/PSRAM usage",                      .func = cmd_heap},
        {.command = "reboot", .help = "Reboot device",                              .func = cmd_reboot},
    };
    for (int i = 0; i < sizeof(cmds) / sizeof(cmds[0]); i++)
        ESP_ERROR_CHECK(esp_console_cmd_register(&cmds[i]));

#if defined(CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG)
    ESP_ERROR_CHECK(esp_console_new_repl_usb_serial_jtag(&uc, &rc, &repl));
#else
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&uc, &rc, &repl));
#endif
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
    ESP_LOGI(TAG, "CLI ready. Type 'help'.");
}
