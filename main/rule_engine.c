#include "rule_engine.h"
#include "rule_store.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "mimi_config.h"

#define TAG "rule_engine"
#define MAX_ACT 8

/* ---- Online rule interval learning (v9) ----
 * Welford online mean/variance per (rule_id, sensor_name).
 * Each cloud LLM tool_call triggers sensor_snapshot → welford_insert.
 * When welford.count >= ONLINE_MIN_SAMPLES, recalculate interval
 * and update the rule's condition in-place. */
#define ONLINE_MAX_TRACKERS 64
#define ONLINE_MIN_SAMPLES  5

typedef struct {
    char rid[24];
    char sensor[16];
    int count;
    float mean;
    float m2;     /* sum of squared diffs from current mean */
} welf_t;

static welf_t s_welf[ONLINE_MAX_TRACKERS];
static int s_welf_n = 0;

static welf_t *welf_find(const char *rid, const char *sensor) {
    for (int i = 0; i < s_welf_n; i++)
        if (strcmp(s_welf[i].rid, rid) == 0 && strcmp(s_welf[i].sensor, sensor) == 0)
            return &s_welf[i];
    if (s_welf_n >= ONLINE_MAX_TRACKERS) return NULL;
    welf_t *w = &s_welf[s_welf_n++];
    strncpy(w->rid, rid, sizeof(w->rid)-1);
    strncpy(w->sensor, sensor, sizeof(w->sensor)-1);
    w->count = 0; w->mean = 0; w->m2 = 0;
    return w;
}

static void welf_insert(welf_t *w, float v) {
    w->count++;
    float delta = v - w->mean;
    w->mean += delta / (float)w->count;
    float delta2 = v - w->mean;
    w->m2 += delta * delta2;
}

static float welf_std(welf_t *w) {
    if (w->count < 2) return 1.0f;
    return sqrtf(w->m2 / (float)(w->count - 1));
}

/* ---- PMS state ---- */
#define PMS_MAX 500
typedef struct { char rid[24]; uint8_t alpha; uint8_t beta; } pms_t;
static pms_t s_pms[PMS_MAX];
static int s_pms_n = 0;

/* xorshift32 PRNG for PMS perturbation */
static uint32_t s_prng = 0xdeadbeef;

static inline uint32_t prng_next(void) {
    s_prng ^= s_prng << 13; s_prng ^= s_prng >> 17; s_prng ^= s_prng << 5;
    return s_prng;
}
static inline float prng_uniform(void) {
    return (float)prng_next() / (float)UINT32_MAX; /* [0,1) */
}

static pms_t *pms_find(const char *id) {
    for (int i = 0; i < s_pms_n; i++)
        if (strcmp(s_pms[i].rid, id) == 0) return &s_pms[i];
    if (s_pms_n >= PMS_MAX) return NULL;
    pms_t *a = &s_pms[s_pms_n++];
    strncpy(a->rid, id, sizeof(a->rid)-1);
    a->alpha = 1; a->beta = 1; /* Beta(1,1) prior */
    return a;
}

/* ---- actuator state ---- */
typedef struct { char *dev; char state[16]; int64_t last_us; } act_t;
static act_t s_act[MAX_ACT];
static int s_act_n = 0;

esp_err_t rule_engine_init(void) { ESP_LOGI(TAG, "Ready"); return ESP_OK; }

static cJSON *_find(const char *id) {
    cJSON *rules = rule_store_get_all();
    int n = cJSON_GetArraySize(rules);
    for (int i = 0; i < n; i++) {
        cJSON *r = cJSON_GetArrayItem(rules, i);
        cJSON *rid = cJSON_GetObjectItem(r, "id");
        if (rid && rid->valuestring && strcmp(rid->valuestring, id) == 0) return r;
    }
    return NULL;
}

static bool _chk(cJSON *c, cJSON *sns) {
    cJSON *se = cJSON_GetObjectItem(c, "sensor");
    cJSON *op = cJSON_GetObjectItem(c, "op");
    cJSON *vl = cJSON_GetObjectItem(c, "value");
    if (!se || !op || !vl) return true;
    cJSON *sv = cJSON_GetObjectItem(sns, se->valuestring);
    /* Missing sensor field → skip this condition (don't fail the rule).
       Injected sensors have 4-5 fields; learned rules may have 10+ conditions
       (temp_trend, hour, light_category...).  Missing = unknown = not a fail. */
    if (!sv) return true;
    double s = sv->valuedouble, v = vl->valuedouble;
    const char *o = op->valuestring;
    if (strcmp(o, "gt") == 0) return s > v;
    if (strcmp(o, "lt") == 0) return s < v;
    if (strcmp(o, "gte") == 0) return s >= v;
    if (strcmp(o, "lte") == 0) return s <= v;
    if (strcmp(o, "eq") == 0) return s == v;
    return false;
}

static bool _match(cJSON *r, cJSON *sns) {
    cJSON *st = cJSON_GetObjectItem(r, "state");
    if (!st) return false;
    const char *ss = st->valuestring;
    if (strcmp(ss, "candidate") && strcmp(ss, "verified") && strcmp(ss, "active")) return false;
    cJSON *sl = cJSON_GetObjectItem(r, "safety_level");
    if (sl && sl->valueint >= SAFETY_L3) return false;
    cJSON *cs = cJSON_GetObjectItem(r, "conditions");
    if (!cs || cJSON_GetArraySize(cs) == 0) return false;

    /* v9: sensor coverage check — at least max(2, n_conds/3) fields
       must be present in the sensor snapshot */
    int n_conds = cJSON_GetArraySize(cs);
    int present = 0;
    for (int i = 0; i < n_conds; i++) {
        cJSON *c = cJSON_GetArrayItem(cs, i);
        cJSON *se = cJSON_GetObjectItem(c, "sensor");
        if (se && se->valuestring) {
            cJSON *sv = cJSON_GetObjectItem(sns, se->valuestring);
            if (sv) present++;
        }
    }
    int min_needed = n_conds / 3;
    if (min_needed < 2) min_needed = 2;
    if (present < min_needed) return false;

    for (int i = 0; i < n_conds; i++)
        if (!_chk(cJSON_GetArrayItem(cs, i), sns)) return false;
    return true;
}

static int _spec(cJSON *r) {
    cJSON *cs = cJSON_GetObjectItem(r, "conditions");
    return cs ? cJSON_GetArraySize(cs) * 2 : 0;
}
static double _conf(cJSON *r) {
    cJSON *cf = cJSON_GetObjectItem(r, "confidence");
    return cf ? cf->valuedouble : 0.0;
}

cJSON *rule_engine_match(cJSON *sns) {
    cJSON *rules = rule_store_get_all(), *hits = cJSON_CreateArray();
    for (int i = 0; i < cJSON_GetArraySize(rules); i++) {
        cJSON *r = cJSON_GetArrayItem(rules, i);
        /* v6 crash fix: 必须深拷贝(recurse=1)，浅拷贝会丢失 action/conditions
           子节点，executor 访问 best->action->device 时空指针 panic
           (Guru Meditation LoadProhibited @ EXCVADDR 0x10)。 */
        if (_match(r, sns)) cJSON_AddItemToArray(hits, cJSON_Duplicate(r, 1));
    }
    return hits;
}

cJSON *rule_engine_resolve(cJSON *hits) {
    int n = cJSON_GetArraySize(hits);
    if (n == 0) return NULL;

    cJSON *best = NULL;
    float best_score = -1e9f;

    s_prng ^= (uint32_t)esp_timer_get_time();  /* perturb seed with wall clock */

    for (int i = 0; i < n; i++) {
        cJSON *r = cJSON_GetArrayItem(hits, i);
        cJSON *rid = cJSON_GetObjectItem(r, "id");
        pms_t *arm = (rid && rid->valuestring) ? pms_find(rid->valuestring) : NULL;

        if (arm && n > 1) {
            /* PMS: mean + uniform perturbation */
            int tot = arm->alpha + arm->beta;
            float mean_v = (float)arm->alpha / (float)tot;
            float delta = 1.0f / (float)(tot + 1);
            float score = mean_v + (prng_uniform() * 2.0f - 1.0f) * delta;
            if (score > best_score) { best_score = score; best = r; }
        } else {
            /* fallback: deterministic specificity + confidence */
            float score = (float)_spec(r) * 10.0f + _conf(r);
            if (score > best_score) { best_score = score; best = r; }
        }
    }

    if (!best) return NULL;

    /* Cooldown check: actuator mutex */
    cJSON *act = cJSON_GetObjectItem(best, "action");
    if (act) {
        cJSON *dev = cJSON_GetObjectItem(act, "device");
        if (dev && dev->valuestring) {
            int64_t now = esp_timer_get_time();
            for (int i = 0; i < s_act_n; i++)
                if (strcmp(s_act[i].dev, dev->valuestring) == 0) {
                    if ((now - s_act[i].last_us) / 1000 < RULE_MUTEX_COOLDOWN_MS) {
                        /* v10.2: idempotent action — actuator already in the
                           requested state (on/off). Repeating fan.on while the
                           fan is already on needs no physical actuation, so
                           it should NOT be blocked (nor fall back to cloud).
                           Only state *transitions* are cooldown-protected. */
                        cJSON *cmdj = cJSON_GetObjectItem(act, "command");
                        const char *cmd = cmdj && cmdj->valuestring
                                          ? cmdj->valuestring : "";
                        bool idempotent = false;
                        if (cmd && s_act[i].state[0]) {
                            bool want_on = (strcmp(cmd, "on") == 0
                                            || strcmp(cmd, "set") == 0);
                            bool is_on = (strcmp(s_act[i].state, "on") == 0
                                          || strcmp(s_act[i].state, "set") == 0);
                            if ((want_on && is_on)
                                || (strcmp(cmd, "off") == 0
                                    && strcmp(s_act[i].state, "off") == 0))
                                idempotent = true;
                        }
                        if (!idempotent) return NULL;
                    }
                }
        }
    }
    return best;
}

static double _wilson(int pos, int tot) {
    if (tot == 0) return 0.0;
    double p = (double)pos / tot, z = 1.96, z2 = z * z, dn = 1.0 + z2 / tot;
    double c = (p + z2 / (2.0 * tot)) / dn;
    double m = z * sqrt((p * (1.0 - p) / tot + z2 / (4.0 * tot * tot))) / dn;
    double v = c - m;
    return v < 0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

void rule_engine_update_on_exec(const char *id, const char *fb) {
    cJSON *r = _find(id); if (!r) return;
    cJSON *ec = cJSON_GetObjectItem(r, "evidence_count");
    cJSON *pf = cJSON_GetObjectItem(r, "positive_feedback");
    cJSON *nf = cJSON_GetObjectItem(r, "negative_feedback");
    if (!ec) { ec = cJSON_AddNumberToObject(r, "evidence_count", 0); }
    if (!pf) { pf = cJSON_AddNumberToObject(r, "positive_feedback", 0); }
    if (!nf) { nf = cJSON_AddNumberToObject(r, "negative_feedback", 0); }
    cJSON_SetNumberValue(ec, ec->valueint + 1);
    if (strcmp(fb, "accepted") == 0) cJSON_SetNumberValue(pf, pf->valueint + 1);
    else if (strcmp(fb, "corrected") == 0) cJSON_SetNumberValue(nf, nf->valueint + 1);
    double cf = _wilson(pf->valueint, ec->valueint);
    cJSON *cfj = cJSON_GetObjectItem(r, "confidence");
    if (cfj) cJSON_SetNumberValue(cfj, cf); else cJSON_AddNumberToObject(r, "confidence", cf);

    /* v8 PMS update */
    pms_t *arm = pms_find(id);
    if (arm) {
        if (strcmp(fb, "accepted") == 0 && arm->alpha < 255) arm->alpha++;
        else if (strcmp(fb, "corrected") == 0 && arm->beta < 255) arm->beta++;
    }

    /* Track last-triggered timestamp for freshness */
    cJSON *lt = cJSON_GetObjectItem(r, "last_triggered");
    if (lt) cJSON_SetNumberValue(lt, (double)esp_timer_get_time() / 1000000.0);
    else cJSON_AddNumberToObject(r, "last_triggered", (double)esp_timer_get_time() / 1000000.0);

    const char *st = cJSON_GetObjectItem(r, "state")->valuestring;
    if (!strcmp(st, "candidate") && ec->valueint >= 3 && cf >= 0.70)
        cJSON_ReplaceItemInObject(r, "state", cJSON_CreateString("verified"));
    else if (!strcmp(st, "verified") && cf >= 0.85 && nf->valueint == 0)
        cJSON_ReplaceItemInObject(r, "state", cJSON_CreateString("active"));
    if (nf->valueint > 0 && (!strcmp(st, "active") || !strcmp(st, "verified")))
        cJSON_ReplaceItemInObject(r, "state", cJSON_CreateString("degraded"));
    rule_store_mark_dirty();
}

/* ---- v8: Time-decaying freshness ---- */
void rule_engine_update_freshness(void) {
    int64_t now = esp_timer_get_time();
    cJSON *rules = rule_store_get_all();
    int n = cJSON_GetArraySize(rules);
    int degraded = 0;

    for (int i = 0; i < n; i++) {
        cJSON *r = cJSON_GetArrayItem(rules, i);
        cJSON *lt = cJSON_GetObjectItem(r, "last_triggered");
        cJSON *st = cJSON_GetObjectItem(r, "state");
        if (!st || !lt) continue;
        const char *ss = st->valuestring;
        if (!strcmp(ss, "retired")) continue;

        double last_ts = lt->valuedouble;
        double days = (now / 1000000.0 - last_ts) / 86400.0;
        if (days < 0) days = 0;
        cJSON *ec = cJSON_GetObjectItem(r, "evidence_count");
        double ev = ec ? (double)ec->valueint : 0.0;
        double tau = RULE_DECAY_TAU_BASE + ev * 0.15;  /* evidence越多衰减越慢 */
        double f = exp(-days / tau);

        cJSON *fr = cJSON_GetObjectItem(r, "freshness");
        if (fr) cJSON_SetNumberValue(fr, f);
        else cJSON_AddNumberToObject(r, "freshness", f);

        /* Degrade if stale */
        if (f < RULE_FRESHNESS_DEGRADE && !strcmp(ss, "active")) {
            cJSON_ReplaceItemInObject(r, "state", cJSON_CreateString("degraded"));
            degraded++;
        }
    }
    if (degraded) {
        ESP_LOGI(TAG, "Freshness: degraded %d stale rules", degraded);
        rule_store_mark_dirty();
    }
}

void rule_engine_actuator_state(const char *dev, const char *st) {
    for (int i = 0; i < s_act_n; i++)
        if (strcmp(s_act[i].dev, dev) == 0) {
            strncpy(s_act[i].state, st, 15);
            s_act[i].last_us = esp_timer_get_time(); return;
        }
    if (s_act_n < MAX_ACT) {
        s_act[s_act_n].dev = strdup(dev);
        strncpy(s_act[s_act_n].state, st, 15);
        s_act[s_act_n].last_us = esp_timer_get_time();
        s_act_n++;
    }
}

cJSON *rule_engine_get_all_rules(void) {
    return rule_store_get_all();
}

/* ---- v9: Online Welford feed + interval update ----
 * Called from executor after each cloud LLM tool_call:
 *   for each sensor in snapshot:
 *     welf_insert(tracker_for(rule_or_device, sensor), sensor_value)
 *   if count >= ONLINE_MIN_SAMPLES:
 *     update rule condition: lower = mean - 2*std, upper = mean + 2*std
 */
void rule_engine_welford_feed(const char *rule_or_dev, cJSON *sensors) {
    if (!sensors) return;
    cJSON *iter = sensors->child;
    while (iter) {
        if (cJSON_IsNumber(iter) && iter->string) {
            /* skip non-sensor fields */
            if (!strcmp(iter->string, "btn_accept") || !strcmp(iter->string, "btn_correct")) {
                iter = iter->next; continue;
            }
            welf_t *w = welf_find(rule_or_dev, iter->string);
            if (w) welf_insert(w, (float)iter->valuedouble);
        }
        iter = iter->next;
    }
}

void rule_engine_online_update_intervals(void) {
    int updated = 0;
    for (int i = 0; i < s_welf_n; i++) {
        welf_t *w = &s_welf[i];
        if (w->count < ONLINE_MIN_SAMPLES) continue;

        /* v9: two-phase interval estimation.
           count < 10: use observed min/max (no distribution assumption)
           count >= 10: mean +/- 2*sigma (statistical generalization) */
        float lo, hi;
        if (w->count < 10) {
            /* Use min/max approximation from mean+std: empirical range */
            float s = welf_std(w);
            lo = w->mean - 1.5f * s;  /* conservative for small N */
            hi = w->mean + 1.5f * s;
        } else {
            lo = w->mean - 2.0f * welf_std(w);
            hi = w->mean + 2.0f * welf_std(w);
        }

        /* Find the rule and update its condition */
        cJSON *rules = rule_store_get_all();
        int n = cJSON_GetArraySize(rules);
        for (int j = 0; j < n; j++) {
            cJSON *r = cJSON_GetArrayItem(rules, j);
            cJSON *rid = cJSON_GetObjectItem(r, "id");
            if (!rid || !rid->valuestring || strcmp(rid->valuestring, w->rid) != 0) continue;
            /* v10: only online-update candidate rules. Verified/active rules
               were distilled with conformal coverage guarantees; overwriting
               them with a small online sample (mean +/- 2sigma) drifts the
               interval and breaks matching (observed: UCI AR dropped from
               25% to 8% after the first 6-min online update). */
            cJSON *st = cJSON_GetObjectItem(r, "state");
            if (!st || !cJSON_IsString(st) ||
                strcmp(st->valuestring, "candidate") != 0) continue;
            cJSON *conds = cJSON_GetObjectItem(r, "conditions");
            if (!conds) continue;
            for (int k = 0; k < cJSON_GetArraySize(conds); k++) {
                cJSON *c = cJSON_GetArrayItem(conds, k);
                cJSON *sn = cJSON_GetObjectItem(c, "sensor");
                cJSON *op = cJSON_GetObjectItem(c, "op");
                if (!sn || !op || strcmp(sn->valuestring, w->sensor) != 0) continue;
                const char *op_s = op->valuestring;
                if (!strcmp(op_s, "gte")) {
                    cJSON_ReplaceItemInObject(c, "value", cJSON_CreateNumber(lo));
                } else if (!strcmp(op_s, "lte")) {
                    cJSON_ReplaceItemInObject(c, "value", cJSON_CreateNumber(hi));
                }
                updated++;
            }
        }
    }
    if (updated) {
        ESP_LOGI(TAG, "Online interval update: %d conditions adjusted", updated);
        rule_store_mark_dirty();
    }
}
