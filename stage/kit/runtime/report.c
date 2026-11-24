// F8 bug reports (report.h). F8 on the render thread (flight.c trigger) calls report_begin: the report gets a folder
// and the next present is captured full size. Thirty presents on, flight.c's dump thread writes the last frames'
// thumbnails into <report>/frames and calls report_finish, which adds the screenshot, the log, system info and an
// issue template, packs it all into one zip beside the folder and writes a shortcut to a pre-filled GitHub issue.
//
// Nothing leaves the machine: the player attaches the zip themselves. The user folder path (C:\Users\<name>) in the
// log and the system info is replaced by %USERPROFILE%. GitHub takes attachments up to 25 MB: a zip over 24 MB is
// packed again with every other thumbnail, then with none.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "report.h"
#include "sysinfo.h"

#define REPORT_ZIP_CAP (24u << 20)
#define LOG_TAIL (6u << 20)
#define PREV_LOG_TAIL (1u << 20)
#define DEFAULT_ISSUES_URL "https://github.com/Misterobots/Revolvers/issues/new"

static char g_root[PLAT_PATH_MAX];   // <exe dir>/reports
static char g_name[64];              // orbitals-report-YYYYMMDD-HHMMSS
static char g_dir[PLAT_PATH_MAX];    // g_root/g_name
static volatile int g_busy;          // from report_begin until report_finish is done
static uint64_t g_frame;
static struct tm g_tm;

static volatile int g_shot_state;    // 0 none, 1 wanted, 2 taken
static uint8_t *g_shot;
static int g_shot_w, g_shot_h;

static char g_notice[2][300];
static volatile int g_notice_i;
static volatile uint64_t g_notice_until;

static void notice(double seconds, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void notice(double seconds, const char *fmt, ...) {
    int i = !__atomic_load_n(&g_notice_i, __ATOMIC_RELAXED);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_notice[i], sizeof g_notice[i], fmt, ap);
    va_end(ap);
    __atomic_store_n(&g_notice_i, i, __ATOMIC_RELEASE);
    __atomic_store_n(&g_notice_until, plat_now_ns() + (uint64_t)(seconds * 1e9), __ATOMIC_RELEASE);
}

int report_notice(char *out, size_t cap) {
    if (plat_now_ns() >= __atomic_load_n(&g_notice_until, __ATOMIC_ACQUIRE)) return 0;
    snprintf(out, cap, "%s", g_notice[__atomic_load_n(&g_notice_i, __ATOMIC_ACQUIRE)]);
    return 1;
}

int report_enabled(void) {
    static int on = -1;
    if (on < 0) on = strcmp(cfg_value("NX2NAVI_REPORT"), "0") != 0;
    return on;
}

int report_shot_wanted(void) { return __atomic_load_n(&g_shot_state, __ATOMIC_ACQUIRE) == 1; }

void report_shot_take(uint8_t *rgba, int w, int h) {
    if (__atomic_load_n(&g_shot_state, __ATOMIC_ACQUIRE) != 1) { free(rgba); return; }
    g_shot = rgba, g_shot_w = w, g_shot_h = h;
    __atomic_store_n(&g_shot_state, 2, __ATOMIC_RELEASE);
}

int report_begin(uint64_t frame) {
    if (__atomic_exchange_n(&g_busy, 1, __ATOMIC_ACQ_REL)) {
        notice(4, "A bug report is still being saved...");
        return 0;
    }
    if (!g_root[0]) {
        char tmp[PLAT_PATH_MAX];
        char *slash = plat_exe_path(tmp, sizeof tmp) ? plat_last_sep(tmp) : NULL;
        if (slash) snprintf(slash, sizeof tmp - (size_t)(slash - tmp), "/reports");
        else snprintf(tmp, sizeof tmp, "reports");
        if (!plat_full_path(tmp, g_root, sizeof g_root)) snprintf(g_root, sizeof g_root, "%s", tmp);
    }
    plat_local_time(&g_tm);
    snprintf(g_name, sizeof g_name, "orbitals-report-%04d%02d%02d-%02d%02d%02d", g_tm.tm_year + 1900, g_tm.tm_mon + 1,
             g_tm.tm_mday, g_tm.tm_hour, g_tm.tm_min, g_tm.tm_sec);
    snprintf(g_dir, sizeof g_dir, "%s/%s", g_root, g_name);
    plat_file_mkdir_p(g_dir);
    if (!plat_dir_exists(g_dir)) {
        rt_log("report: cannot create %s", g_dir);
        notice(6, "Could not save a bug report (no write access to %s)", g_root);
        __atomic_store_n(&g_busy, 0, __ATOMIC_RELEASE);
        return 0;
    }
    g_frame = frame;
    free(g_shot), g_shot = NULL;
    __atomic_store_n(&g_shot_state, 1, __ATOMIC_RELEASE);
    notice(30, "Saving bug report...");
    rt_log("report: F8 at frame %llu: %s", (unsigned long long)frame, g_dir);
    return 1;
}

void report_frames_dir(char *out, size_t cap) { snprintf(out, cap, "%s/frames", g_dir); }

// ---------------------------------------------------------------- text helpers
typedef struct { char *p; size_t n, cap; } Buf;

static void buf_put(Buf *b, const void *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        size_t c = b->cap ? b->cap : 4096;
        while (c < b->n + n + 1) c *= 2;
        char *q = realloc(b->p, c);
        if (!q) return;
        b->p = q, b->cap = c;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void buf_printf(Buf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void buf_printf(Buf *b, const char *fmt, ...) {
    char tmp[4096];
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (k > 0) buf_put(b, tmp, (size_t)k < sizeof tmp ? (size_t)k : sizeof tmp - 1);
}

static int lower(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

// The user folder (both slash styles, any case) becomes %USERPROFILE%: logs name files under it.
static void scrub(Buf *b) {
    const char *home = getenv("USERPROFILE");
    if (!home || !*home) home = getenv("HOME");
    size_t hn = home ? strlen(home) : 0;
    if (hn < 4 || !b->p) return;
    static const char rep[] = "%USERPROFILE%";
    Buf o = {0};
    size_t i = 0;
    while (i < b->n) {
        size_t k = 0;
        while (k < hn && i + k < b->n) {
            int a = lower((unsigned char)b->p[i + k]), c = lower((unsigned char)home[k]);
            if (a == '/') a = '\\';
            if (c == '/') c = '\\';
            if (a != c) break;
            k++;
        }
        if (k == hn) {
            buf_put(&o, rep, sizeof rep - 1);
            i += hn;
        } else {
            buf_put(&o, b->p + i, 1);
            i++;
        }
    }
    free(b->p);
    *b = o;
}

// The last `tail` bytes of a file, from a line start, scrubbed.
static int read_tail(const char *path, size_t tail, Buf *out) {
    uint64_t n = 0;
    uint8_t *d = plat_file_read(path, &n);
    if (!d) return 0;
    size_t from = 0;
    if (n > tail) {
        from = (size_t)n - tail;
        while (from < n && d[from - 1] != '\n') from++;
        buf_printf(out, "[report: the first %llu bytes of this log are left out]\n", (unsigned long long)from);
    }
    buf_put(out, d + from, (size_t)n - from);
    free(d);
    scrub(out);
    return 1;
}

static int write_buf(const char *dir, const char *name, const Buf *b) {
    char path[PLAT_PATH_MAX + 64];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    return plat_file_write(path, b->p ? b->p : "", b->n);
}

// ---------------------------------------------------------------- zip (deflate for text, stored for PNGs)
typedef struct { char name[96]; uint32_t crc, csize, usize, off; uint16_t method; } ZEnt;
typedef struct { FILE *f; ZEnt e[1024]; int n; uint32_t at; uint16_t dtime, ddate; } Zip;

static void le16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v, p[1] = (uint8_t)(v >> 8); }
static void le32(uint8_t *p, uint32_t v) { le16(p, v), le16(p + 2, v >> 16); }

static int zip_add(Zip *z, const char *name, const uint8_t *data, size_t n) {
    if (z->n >= (int)(sizeof z->e / sizeof *z->e)) return 0;
    size_t nl = strlen(name);
    if (nl >= sizeof z->e[0].name) return 0;
    const char *dot = strrchr(name, '.');
    int store = dot && !strcmp(dot, ".png");
    uint8_t *c = NULL;
    size_t cn = n;
    if (!store && n) {
        c = malloc(n * 9 / 8 + 64);
        cn = c ? flight_deflate_raw(data, n, c) : 0;
        if (!cn || cn >= n) free(c), c = NULL, cn = n, store = 1;
    } else {
        store = 1;
    }
    ZEnt *e = &z->e[z->n++];
    memcpy(e->name, name, nl + 1);
    e->crc = flight_crc32(data, n), e->usize = (uint32_t)n, e->csize = (uint32_t)cn, e->off = z->at;
    e->method = store ? 0 : 8;
    uint8_t h[30] = {'P', 'K', 3, 4};
    le16(h + 4, 20), le16(h + 6, 0x0800), le16(h + 8, e->method), le16(h + 10, z->dtime), le16(h + 12, z->ddate);
    le32(h + 14, e->crc), le32(h + 18, e->csize), le32(h + 22, e->usize), le16(h + 26, (uint32_t)nl), le16(h + 28, 0);
    fwrite(h, 1, 30, z->f);
    fwrite(name, 1, nl, z->f);
    fwrite(c ? c : data, 1, cn, z->f);
    z->at += 30 + (uint32_t)nl + (uint32_t)cn;
    free(c);
    return 1;
}

static int zip_close(Zip *z) {
    uint32_t cd = z->at, cdn = 0;
    for (int i = 0; i < z->n; i++) {
        ZEnt *e = &z->e[i];
        size_t nl = strlen(e->name);
        uint8_t h[46] = {'P', 'K', 1, 2};
        le16(h + 4, 20), le16(h + 6, 20), le16(h + 8, 0x0800), le16(h + 10, e->method), le16(h + 12, z->dtime);
        le16(h + 14, z->ddate), le32(h + 16, e->crc), le32(h + 20, e->csize), le32(h + 24, e->usize);
        le16(h + 28, (uint32_t)nl), le32(h + 42, e->off);
        fwrite(h, 1, 46, z->f);
        fwrite(e->name, 1, nl, z->f);
        cdn += 46 + (uint32_t)nl;
    }
    uint8_t t[22] = {'P', 'K', 5, 6};
    le16(t + 8, (uint32_t)z->n), le16(t + 10, (uint32_t)z->n), le32(t + 12, cdn), le32(t + 16, cd);
    fwrite(t, 1, 22, z->f);
    z->at += cdn + 22;
    return fclose(z->f) == 0;
}

static int zip_file(Zip *z, const char *path, const char *name) {
    uint64_t n = 0;
    uint8_t *d = plat_file_read(path, &n);
    if (!d) return 0;
    int ok = zip_add(z, name, d, (size_t)n);
    free(d);
    return ok;
}

typedef struct { char names[1024][32]; int n; } NameList;
static int collect(const char *name, void *ctx) {
    NameList *l = ctx;
    if (l->n < 1024 && strlen(name) < sizeof l->names[0]) snprintf(l->names[l->n++], sizeof l->names[0], "%s", name);
    return 1;
}
static int cmp_name(const void *a, const void *b) { return strcmp(a, b); }

// thumb_step: 1 every thumbnail, 2 every other one, 0 none. The zip's size in bytes, 0 on failure.
static uint32_t build_zip(const char *zip_path, const char *frames_dir, const NameList *thumbs, int thumb_step) {
    Zip *z = calloc(1, sizeof *z);
    if (!z || !(z->f = fopen(zip_path, "wb"))) return free(z), 0;
    z->dtime = (uint16_t)(g_tm.tm_hour << 11 | g_tm.tm_min << 5 | g_tm.tm_sec / 2);
    z->ddate = (uint16_t)((g_tm.tm_year - 80) << 9 | (g_tm.tm_mon + 1) << 5 | g_tm.tm_mday);
    char path[PLAT_PATH_MAX + 64], name[128];
    static const char *const top[] = {"issue.md", "sysinfo.txt", "screenshot.png", "log.txt", "previous-log.txt"};
    for (size_t i = 0; i < sizeof top / sizeof *top; i++) {
        snprintf(path, sizeof path, "%s/%s", g_dir, top[i]);
        if (plat_file_exists(path)) zip_file(z, path, top[i]);
    }
    snprintf(path, sizeof path, "%s/frames.tsv", frames_dir);
    if (plat_file_exists(path)) zip_file(z, path, "frames/frames.tsv");
    for (int i = 0; thumb_step && i < thumbs->n; i += thumb_step) {
        snprintf(path, sizeof path, "%s/%s", frames_dir, thumbs->names[i]);
        snprintf(name, sizeof name, "frames/%s", thumbs->names[i]);
        zip_file(z, path, name);
    }
    uint32_t size = zip_close(z) ? z->at : 0;
    free(z);
    return size;
}

static void url_put(Buf *b, const char *s) {
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || strchr("-_.~", c)) buf_put(b, s, 1);
        else buf_printf(b, "%%%02X", c);
    }
}

// The first line of sysinfo starting with key ("GPU: ..."), without the key.
static void info_line(const char *info, const char *key, char *out, size_t cap) {
    out[0] = 0;
    const char *p = strstr(info, key);
    if (!p) return;
    p += strlen(key);
    size_t n = strcspn(p, "\r\n");
    snprintf(out, cap, "%.*s", (int)(n < cap - 1 ? n : cap - 1), p);
}

// ---------------------------------------------------------------- the dump thread's part
void report_finish(const char *frames_dir, uint64_t frame) {
    uint64_t t0 = plat_now_ms();
    for (int i = 0; i < 300 && __atomic_load_n(&g_shot_state, __ATOMIC_ACQUIRE) != 2; i++) plat_sleep_ms(10);
    int have_shot = 0;
    char path[PLAT_PATH_MAX + 64];
    if (__atomic_load_n(&g_shot_state, __ATOMIC_ACQUIRE) == 2 && g_shot) {
        snprintf(path, sizeof path, "%s/screenshot.png", g_dir);
        have_shot = flight_png(path, g_shot, (uint32_t)g_shot_w, (uint32_t)g_shot_h);
    }
    __atomic_store_n(&g_shot_state, 0, __ATOMIC_RELEASE);
    free(g_shot), g_shot = NULL;

    char game[16];
    snprintf(game, sizeof game, "%d.%d.%d", rt_game_version / 100, rt_game_version / 10 % 10, rt_game_version % 10);
    Buf info = {0};
    buf_printf(&info, "Orbitals PC %s (game %s)\nReport: %s, F8 at frame %llu\n", rt_port_version, game, g_name,
               (unsigned long long)g_frame);
    {
        char *s = malloc(32768);
        if (s) {
            sysinfo_text(s, 32768);
            buf_put(&info, s, strlen(s));
            free(s);
        }
    }
    scrub(&info);
    write_buf(g_dir, "sysinfo.txt", &info);

    const char *log = rt_log_path();
    if (log) {
        Buf b = {0};
        if (read_tail(log, LOG_TAIL, &b)) write_buf(g_dir, "log.txt", &b);
        free(b.p);
        char prev[PLAT_PATH_MAX + 16];
        snprintf(prev, sizeof prev, "%s", log);
        char *dot = strrchr(prev, '.');
        if (dot && !strcmp(dot, ".log")) {
            strcpy(dot, ".1.log");
            Buf pb = {0};
            if (read_tail(prev, PREV_LOG_TAIL, &pb)) write_buf(g_dir, "previous-log.txt", &pb);
            free(pb.p);
        }
    }

    char gpu[200], drv[200], os[160];
    info_line(info.p ? info.p : "", "GPU: ", gpu, sizeof gpu);
    info_line(info.p ? info.p : "", "Driver: ", drv, sizeof drv);
    info_line(info.p ? info.p : "", "OS: ", os, sizeof os);
    Buf body = {0};
    buf_printf(&body,
               "**What happened?**\n<!-- e.g. rainbow edges on asteroids, the screen flashes, a crash. -->\n\n"
               "**Where in the game?**\n<!-- area, mission or menu, and what you were doing -->\n\n"
               "**Does it happen every time?**\n\n"
               "**Report file**\n<!-- Drag %s.zip from the game's reports folder into this box. -->\n\n"
               "---\nOrbitals PC %s, game %s\nGPU: %s\nDriver: %s\nOS: %s\n",
               g_name, rt_port_version, game, gpu, drv, os);
    write_buf(g_dir, "issue.md", &body);

    NameList *thumbs = calloc(1, sizeof *thumbs);
    if (thumbs) {
        plat_file_list(frames_dir, ".png", collect, thumbs);
        qsort(thumbs->names, (size_t)thumbs->n, sizeof thumbs->names[0], cmp_name);
    }
    char zip_path[PLAT_PATH_MAX + 80];
    snprintf(zip_path, sizeof zip_path, "%s/%s.zip", g_root, g_name);
    static const NameList none;
    uint32_t size = 0;
    int step = 1;
    for (; step <= 2; step++) {
        size = build_zip(zip_path, frames_dir, thumbs ? thumbs : &none, step);
        if (!size || size <= REPORT_ZIP_CAP) break;
    }
    if (size > REPORT_ZIP_CAP) size = build_zip(zip_path, frames_dir, &none, 0), step = 0;
    free(thumbs);

    // A shortcut beside the zip: a new GitHub issue with the template and the machine filled in
    const char *base = cfg_get("NX2NAVI_REPORT_URL");
    if (!base || !*base) base = DEFAULT_ISSUES_URL;
    Buf url = {0}, title = {0};
    buf_printf(&title, "[Report] <short description> (%s)", g_name);
    buf_printf(&url, "[InternetShortcut]\r\nURL=%s?title=", base);
    url_put(&url, title.p);
    buf_put(&url, "&body=", 6);
    url_put(&url, body.p ? body.p : "");
    buf_put(&url, "\r\n", 2);
    snprintf(path, sizeof path, "%s/%s - open a GitHub issue.url", g_root, g_name);
    int have_url = plat_file_write(path, url.p, url.n);
    free(url.p), free(title.p), free(body.p), free(info.p);

    if (size) {
        notice(12, "Bug report saved: reports\\%s.zip (%.1f MB). Attach it to a new issue on GitHub; the "
                   "\"open a GitHub issue\" shortcut beside it fills in the form.", g_name, size / 1048576.0);
        rt_log("report: %s.zip, %.1f MB (%s, thumbnails %s), %s, written in %llu ms (frames from F8 at %llu)", g_name,
               size / 1048576.0, have_shot ? "screenshot" : "no screenshot",
               step == 1 ? "all" : step == 2 ? "every other one" : "none", have_url ? "issue shortcut" : "no shortcut",
               (unsigned long long)(plat_now_ms() - t0), (unsigned long long)frame);
    } else {
        notice(8, "Could not write the bug report zip; the files are in reports\\%s", g_name);
        rt_log("report: cannot write %s", zip_path);
    }
    __atomic_store_n(&g_busy, 0, __ATOMIC_RELEASE);
}
