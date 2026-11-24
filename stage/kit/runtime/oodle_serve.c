// --oodle-serve: the game's own OodleLZ_Decompress for the offline shader tools (tools/ue_shaders.py).
//
// Unreal Engine 5 keeps each shader group of its IoStore container Oodle-compressed, and Oodle's decoder is not open
// source, so the shaders are decoded with the copy linked into the game's main, lifted with the rest of it. This is
// the Windows counterpart of nxnative's DM_OODLE_SERVE (dm/host/hle_oodle.c).
//
// NX2NAVI_OODLE_AT=<offset in main> names the function (tools/oodle_find.py finds it: the one function that takes
// the address of the string "OodleLZ_Decompress"). Requests on stdin until it ends: u64 compressed length, u64
// decompressed length, the compressed bytes. Answer on stdout: u64 bytes decoded (0 on failure), then those bytes.
// Everything else the runtime prints goes to stderr, so stdout carries only answers.
//
// Served on the guest main thread right after loading, before the SDK starts: Oodle takes its work memory from the
// caller (decoderMemory), so it needs nothing the SDK sets up.
#include "host.h"
#include "cpu.h"
#include "cfg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define fileno _fileno
#else
#include <unistd.h>
#endif

static uint64_t g_oodle;
static FILE *g_in, *g_out;

static int read_all(void *p, size_t n) { return fread(p, 1, n, g_in) == n; }
static void write_all(const void *p, size_t n) {
    if (fwrite(p, 1, n, g_out) != n) rt_fatal("oodle-serve: answer write failed");
}

// OodleLZ_Decompress(compBuf, compBufSize, rawBuf, rawLen, fuzzSafe, checkCRC, verbosity, decBufBase, decBufSize,
// fpCallback, callbackUserData, decoderMemory, decoderMemorySize, threadPhase): arguments 9-14 on the stack, one
// 8-byte slot each (AAPCS64). rt_call passes at most 8 and does not touch sp, so the slots are written here.
static int64_t decode(Cpu *c, const void *comp, int64_t cl, void *raw, int64_t rl, void *scratch, int64_t sn) {
    uint64_t sp = c->sp, nsp = (sp - 64) & ~15ull;
    uint64_t *slot = (uint64_t *)(uintptr_t)nsp;
    slot[0] = 0;                              // decBufSize
    slot[1] = 0;                              // fpCallback
    slot[2] = 0;                              // callbackUserData
    slot[3] = (uint64_t)(uintptr_t)scratch;   // decoderMemory
    slot[4] = (uint64_t)sn;                   // decoderMemorySize
    slot[5] = 3;                              // threadPhase: unthreaded
    c->sp = nsp;
    int64_t got = (int64_t)rt_call(c, g_oodle, 8, (uint64_t)(uintptr_t)comp, (uint64_t)cl, (uint64_t)(uintptr_t)raw,
                                   (uint64_t)rl, (uint64_t)1 /* fuzz safe */, (uint64_t)0 /* no CRC check */,
                                   (uint64_t)0 /* silent */, (uint64_t)0 /* decBufBase */);
    c->sp = sp;
    return got;
}

static void serve(Cpu *c) {
    enum { SLACK = 0x10000 };
    size_t sn = 16u << 20, comp_cap = 0, raw_cap = 0;
    uint8_t *scratch = malloc(sn), *comp = NULL, *raw = NULL;
    uint64_t n = 0, bad = 0;
    if (!scratch) rt_fatal("oodle-serve: decoder memory");
    char d[128];
    rt_logf("oodle-serve: OodleLZ_Decompress at %s, requests on stdin\n", ldr_describe(g_oodle, d, sizeof d));
    for (;;) {
        uint64_t hdr[2];
        if (!read_all(hdr, sizeof hdr)) break;
        if (hdr[0] > (1ull << 32) || hdr[1] > (1ull << 32))
            rt_fatal("oodle-serve: request too large (%#llx -> %#llx)", (unsigned long long)hdr[0],
                     (unsigned long long)hdr[1]);
        if (hdr[0] > comp_cap) {
            free(comp); comp_cap = hdr[0];
            if (!(comp = malloc(comp_cap))) rt_fatal("oodle-serve: input buffer");
        }
        if (hdr[1] + SLACK > raw_cap) {
            free(raw); raw_cap = hdr[1] + SLACK;
            if (!(raw = malloc(raw_cap))) rt_fatal("oodle-serve: output buffer");
        }
        if (hdr[0] && !read_all(comp, hdr[0])) rt_fatal("oodle-serve: request cut short");
        int64_t got = decode(c, comp, (int64_t)hdr[0], raw, (int64_t)hdr[1], scratch, (int64_t)sn);
        uint64_t out = got == (int64_t)hdr[1] ? (uint64_t)got : 0;
        if (!out && hdr[1] && bad++ < 8)
            rt_logf("oodle-serve: request %llu (%llu -> %llu bytes) failed (decoder returned %lld)\n",
                    (unsigned long long)n, (unsigned long long)hdr[0], (unsigned long long)hdr[1], (long long)got);
        write_all(&out, sizeof out);
        if (out) write_all(raw, out);
        fflush(g_out);
        n++;
    }
    rt_logf("oodle-serve: %llu requests, %llu failed\n", (unsigned long long)n, (unsigned long long)bad);
    fflush(stderr);
    exit(bad ? 1 : 0);
}

int oodle_serve(void) {
    const char *at = cfg_get("NX2NAVI_OODLE_AT");
    if (!at || !*at) { fprintf(stderr, "oodle-serve: set NX2NAVI_OODLE_AT (tools/oodle_find.py)\n"); return 2; }
    // stdout carries only answers: keep a private handle to it and send every other write to stderr
    int fd = dup(fileno(stdout));
    if (fd < 0 || dup2(fileno(stderr), fileno(stdout)) < 0) { fprintf(stderr, "oodle-serve: stdout\n"); return 2; }
#ifdef _WIN32
    _setmode(fd, _O_BINARY);
    _setmode(fileno(stdin), _O_BINARY);
#endif
    g_in = stdin;
    g_out = fdopen(fd, "wb");
    if (!g_out) { fprintf(stderr, "oodle-serve: answer stream\n"); return 2; }
    ldr_load(rt_config()->exefs);
    g_oodle = ldr_module(0)->base + strtoull(at, NULL, 0);
    hostfn_seal();
    kern_run_on_guest_main(serve);
    return 0;
}
