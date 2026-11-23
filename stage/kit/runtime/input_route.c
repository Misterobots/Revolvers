#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "input_route.h"
#include "cfg.h"
#include "host.h"
#include "platform/platform.h"

typedef struct { uint64_t tick, buttons; int player; int32_t axes[4]; } RouteEvent;
typedef struct { uint64_t buttons; int32_t axes[4]; } RouteState;
static PlatMutex route_lock = PLAT_MUTEX_INIT;
static const char *control;
static int initialized, playing;
static FILE *record;
static uint64_t epoch, end_tick, next_check, release_tick;
static RouteState last[2], playback[2];
static RouteEvent *events;
static size_t count, cursor;

static void status(const char *message) {
    char path[2048];
    snprintf(path, sizeof path, "%s.status", control);
    FILE *f = fopen(path, "w");
    if (f) { fprintf(f, "%s\n", message); fclose(f); }
    rt_log("input_route: %s", message);
}

static uint64_t tick_now(void) { return frame_step_ticks(); }

static void write_state(uint64_t tick, int p, const RouteState *s) {
    fprintf(record, "%llu %d %llx %d %d %d %d\n", (unsigned long long)tick, p + 1,
            (unsigned long long)s->buttons, s->axes[0], s->axes[1], s->axes[2], s->axes[3]);
}

static void stop_record(uint64_t now) {
    if (!record) return;
    RouteState zero = {0};
    uint64_t elapsed = now >= epoch ? now - epoch : 0;
    for (int p = 0; p < 2; p++) write_state(elapsed, p, &zero);
    fprintf(record, "%llu END\n", (unsigned long long)(elapsed + 1));
    fclose(record);
    record = NULL;
}

static int load_route(const char *path, uint64_t now) {
    FILE *f = fopen(path, "r");
    if (!f) { status("error: cannot open route"); return 0; }
    RouteEvent *loaded = NULL;
    size_t n = 0, cap = 0;
    uint64_t end = 0, previous = 0;
    int ended = 0, valid = 1;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        unsigned long long tick, buttons;
        int p; int32_t a[4]; char word[16];
        if (sscanf(line, "%llu %15s", &tick, word) == 2 && !strcmp(word, "END")) {
            if (ended || tick < previous) { valid = 0; break; }
            end = tick; ended = 1; continue;
        }
        if (ended || sscanf(line, "%llu %d %llx %d %d %d %d", &tick, &p, &buttons,
                            &a[0], &a[1], &a[2], &a[3]) != 7 || p < 1 || p > 2 || tick < previous) {
            valid = 0; break;
        }
        for (int i = 0; i < 4; i++) if (a[i] < -32767 || a[i] > 32767) valid = 0;
        if (!valid || n >= 1000000) { valid = 0; break; }
        if (n == cap) {
            cap = cap ? cap * 2 : 256;
            RouteEvent *grown = realloc(loaded, cap * sizeof *loaded);
            if (!grown) { valid = 0; break; }
            loaded = grown;
        }
        loaded[n] = (RouteEvent){.tick = tick, .buttons = buttons, .player = p - 1};
        memcpy(loaded[n++].axes, a, sizeof a);
        previous = tick;
    }
    fclose(f);
    if (!valid || !ended || !n) { free(loaded); status("error: incomplete or invalid route"); return 0; }
    stop_record(now);
    free(events); events = loaded; count = n; cursor = 0;
    memset(playback, 0, sizeof playback);
    epoch = now; end_tick = end; playing = 1; release_tick = 0;
    status("playing");
    return 1;
}

static void command_check(uint64_t now) {
    if (!initialized) {
        initialized = 1;
        control = cfg_get("NX2NAVI_ROUTE_CONTROL");
        if (control && *control) status("ready");
    }
    if (!control || !*control || plat_now_ms() < next_check) return;
    next_check = plat_now_ms() + 100;
    FILE *f = fopen(control, "r");
    if (!f) return;
    char line[2048] = {0};
    int got = fgets(line, sizeof line, f) != NULL;
    fclose(f);
    if (!got || !strchr(line, '\n')) return; // producer atomically publishes a complete line
    if (remove(control) != 0) return;
    line[strcspn(line, "\r\n")] = 0;
    if (!strcmp(line, "stop")) {
        stop_record(now); playing = 0; release_tick = now + 1;
        status("stopped");
    } else if (!strncmp(line, "record ", 7) && line[7]) {
        // Never truncate an existing recording. All session paths are unique.
        FILE *exists = fopen(line + 7, "r");
        if (exists) { fclose(exists); status("error: recording already exists"); return; }
        FILE *out = fopen(line + 7, "w");
        if (!out) { status("error: cannot create recording"); return; }
        stop_record(now); playing = 0; record = out; epoch = now;
        memset(last, 0, sizeof last);
        fprintf(record, "# Orbitals route v1: relative game tick, player, buttons(hex), lx ly rx ry\n");
        for (int p = 0; p < 2; p++) write_state(0, p, &last[p]);
        fflush(record); status("recording");
    } else if (!strncmp(line, "play ", 5) && line[5]) load_route(line + 5, now);
    else status("error: expected record <path>, play <path>, or stop");
}

int input_route_connected(void) {
    // Opt-in keeps player 2 available during playback, regardless of host devices.
    const char *c = cfg_get("NX2NAVI_ROUTE_CONTROL");
    return c && *c;
}

void input_route_poll(int player, uint64_t *buttons, int32_t axes[4], int blocked) {
    if (player < 0 || player > 1) return;
    plat_mutex_lock(&route_lock);
    uint64_t now = tick_now();
    command_check(now);
    if (playing) {
        uint64_t elapsed = now >= epoch ? now - epoch : 0;
        while (cursor < count && events[cursor].tick <= elapsed) {
            RouteEvent *e = &events[cursor++];
            playback[e->player].buttons = e->buttons;
            memcpy(playback[e->player].axes, e->axes, sizeof e->axes);
        }
        if (elapsed >= end_tick) {
            playing = 0; release_tick = now + 1;
            memset(playback, 0, sizeof playback); status("finished");
        }
        *buttons = playback[player].buttons;
        memcpy(axes, playback[player].axes, sizeof playback[player].axes);
        *buttons &= ~((uint64_t)255 << 16);
        for (int s = 0; s < 2; s++) {
            int bit = 16 + s * 4;
            if (axes[s * 2] <= -16384) *buttons |= 1ull << bit;
            if (axes[s * 2 + 1] >= 16384) *buttons |= 1ull << (bit + 1);
            if (axes[s * 2] >= 16384) *buttons |= 1ull << (bit + 2);
            if (axes[s * 2 + 1] <= -16384) *buttons |= 1ull << (bit + 3);
        }
    }
    if (blocked || (release_tick && now <= release_tick)) {
        *buttons = 0; memset(axes, 0, 4 * sizeof *axes);
    }
    if (record) {
        RouteState current = {.buttons = *buttons};
        memcpy(current.axes, axes, sizeof current.axes);
        if (current.buttons != last[player].buttons || memcmp(current.axes, last[player].axes, sizeof current.axes)) {
            write_state(now >= epoch ? now - epoch : 0, player, &current);
            last[player] = current; fflush(record);
        }
    }
    plat_mutex_unlock(&route_lock);
}
