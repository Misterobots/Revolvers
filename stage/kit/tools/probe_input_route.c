// Standalone checks against the actual route implementation. No game or desktop input.
#include <assert.h>
#include "../runtime/input_route.c"
static uint64_t test_tick, test_ms;
static const char *test_control;
uint64_t frame_step_ticks(void) { return test_tick; }
uint64_t plat_now_ns(void) { return test_ms * 1000000; }
const char *cfg_get(const char *name) { return !strcmp(name, "NX2NAVI_ROUTE_CONTROL") ? test_control : NULL; }
void rt_log(const char *fmt, ...) { (void)fmt; }
void plat_mutex_lock(PlatMutex *m) { (void)m; }
void plat_mutex_unlock(PlatMutex *m) { (void)m; }
static void cmd(const char *value) {
    FILE *f = fopen(test_control, "w"); assert(f);
    fprintf(f, "%s\n", value); fclose(f); test_ms += 101;
}
static void poll(int p, uint64_t *b, int32_t a[4]) { input_route_poll(p, b, a, 0); }
int main(int argc, char **argv) {
    assert(argc == 2); test_control = argv[1];
    char recording[2048], command[2200];
    snprintf(recording, sizeof recording, "%s.route", test_control);
    remove(recording);
    uint64_t b = 7; int32_t a[4] = {1,2,3,4};
    poll(0, &b, a); assert(b == 7 && a[0] == 1); // idle is transparent
    snprintf(command, sizeof command, "record %s", recording); cmd(command);
    test_tick = 100; b = 1; memset(a, 0, sizeof a); poll(0, &b, a);
    test_tick = 106; b = 2; a[0] = 32767; poll(1, &b, a);
    test_tick = 112; cmd("stop"); poll(0, &b, a);
    snprintf(command, sizeof command, "play %s", recording); cmd(command);
    test_tick = 200; b = 99; memset(a, 0, sizeof a); poll(0, &b, a); assert(b == 1);
    b = 99; poll(1, &b, a); assert(b == 0 && a[0] == 0);
    test_tick = 206; poll(1, &b, a); assert((b & 3) == 2 && a[0] == 32767 && (b & (1ull << 18)));
    test_tick = 208; input_route_poll(1, &b, a, 1); assert(b == 0 && a[0] == 0);
    test_tick = 213; poll(1, &b, a); assert(!playing && b == 0 && a[0] == 0);
    test_tick = 215; b = 42; a[0] = 10; poll(0, &b, a); assert(b == 42 && a[0] == 10);
    // Invalid and incomplete files never activate playback.
    FILE *f = fopen(recording, "w"); fprintf(f, "0 1 1 0 0 0 0\n"); fclose(f);
    cmd(command); poll(0, &b, a); assert(!playing);
    f = fopen(recording, "w"); fprintf(f, "2 1 1 0 0 0 0\n1 END\n"); fclose(f);
    cmd(command); poll(0, &b, a); assert(!playing);
    remove(recording); remove(test_control);
    printf("input_route: two-player recording/replay, game-time scheduling, neutral release, host-menu blocking and invalid-file rejection passed\n");
    return 0;
}
