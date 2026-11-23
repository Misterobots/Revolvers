// Transfers of the Vulkan op-stream replay: buffer and texture copies, texture
// clears, event signals, counter reports and conditional rendering. vk_exec.c calls vk_exec_op_xfer for these ops.
//
// Buffers are guest memory: vk_mem_resolve maps an address to the VkBuffer behind its pool, textures are the vk_texture.c images
// (linear texel layout, GENERAL layout; block-linear storage is only converted when a mirrored texture is uploaded). So a
// copy is one Vulkan transfer command and the pitch of a buffer side comes from the payload (OP_COPY_STRIDE or the
// tight row). Every copy ends the open pass and starts with a full barrier. Destinations: an image is marked
// GPU-written at record time; a CPU-visible buffer range is reported to vk_gpu_written once the GPU is done (write
// watch does not see GPU writes, textures stored there re-upload).
//
// Events and counters are CPU-visible side effects. Their values go into guest memory on the completion thread through
// vk_completion_add, in submit order, as gfx_null.c does: event WRITE / ADD, counter {0, timestamp} for every type but
// samples passed. A timestamp report puts 0 in the counter word: NX2NAVI gates its clears on two such words being unequal.
//
// Samples passed (NVN_COUNTER_TYPE_SAMPLES_PASSED) is a running count on the queue: RESET_COUNTER zeroes it, every
// draw after it adds the samples that passed the depth and stencil tests, REPORT_COUNTER writes the running value.
// NX2NAVI resets and reports it 3 to 8 times a frame in the field and its CPU reads the results back (f_70010c3298); a 0
// there culls what the query tested. The replay counts with Vulkan occlusion queries ("occlusion counts" below).
//
// Conditional rendering is a CPU-side skip, not VK_EXT_conditional_rendering: the extension compares one 32-bit word
// with zero, NVN compares the counter words at addr and addr + 16. The condition is evaluated when the op is recorded
// from guest memory; a counter reported by an earlier submission that is not done yet counts as 0 if it is a timestamp,
// as unknown if it is samples passed. An unknown count renders: the report usually sits in the command buffer being
// recorded, so its value cannot be waited for, and the word in memory still holds an older query's count. Rendering
// costs at most the work the condition would have saved; in the field
// NX2NAVI conditions only on timestamp reports. The condition sets VkExec.render_disabled, which vk_exec.c honours for
// clears and vk_exec_draw.c for draws and dispatches. A condition that cannot be read (address outside every pool)
// renders.
// Copies and CLEAR_TEX are not conditional.
//
// Contains code Copyright (c) 2026 jhaizhou, Copyright (c) 2026 Peiikai (MIT; see THIRD_PARTY_NOTICES.md).
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"

enum { W_FORMAT, W_REGION, W_BUFFER, W_OVERLAP, W_COPY, W_CLEAR, W_QUERY, W_COUNT };

static void warn_once(int key, const char *fmt, ...) {
    static volatile int32_t seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

static uint32_t max1(uint32_t v) { return v ? v : 1; }

// ---- completion callbacks (completion thread)

typedef struct { uint64_t addr, size; } Range;

static void guest_written_cb(void *arg) {
    Range *r = arg;
    vk_gpu_written(r->addr, r->size);
    free(r);
}

// The GPU wrote [addr, addr+size) of CPU-visible memory: tell the texture cache once the fence is reached.
static void note_gpu_write(VkExec *ex, uint64_t addr, uint64_t size) {
    vk_usnap_drop(addr, size);
    if (nvn_gpu_only(addr)) return;
    Range *r = malloc(sizeof *r);
    if (!r) rt_fatal("vk: out of memory");
    *r = (Range){addr, size};
    vk_mem_gpu_pending(addr, size);
    vk_completion_add(ex->fence, guest_written_cb, r);
}

typedef struct { volatile uint32_t *val; uint32_t mode, value; } EventWrite;

static void event_cb(void *arg) {
    EventWrite *e = arg;
    if (e->mode == NVN_EVENT_SIGNAL_MODE_ADD) __atomic_fetch_add(e->val, e->value, __ATOMIC_RELEASE);
    else __atomic_store_n(e->val, e->value, __ATOMIC_RELEASE);
    free(e);
}

// NVNcounterData = {counter, timestamp}; timestamps are Tegra 614.4 MHz ticks (independent of GetSystemTick).
void oqw_report(uint64_t addr, uint64_t seq, uint64_t frame);   // oq_watch_win.c (NX2NAVI_OQ_READS)
void oqw_port_write(int begin);
void oqw_result(uint64_t addr, uint64_t seq, uint64_t count);
static void write_counter(uint64_t addr, uint64_t counter) {
    uint64_t rec[2] = {counter, (uint64_t)((unsigned __int128)kern_now_ns() * 384 / 625)};
    oqw_port_write(1);
    memcpy((void *)(uintptr_t)addr, rec, sizeof rec);
    oqw_port_write(0);
}

// Reports other than samples passed (timestamps; the statistics counters are not modelled) write a counter of 0. The
// counter word matters: Ocarina writes {1, 0} into a record from the CPU, reports TIMESTAMP_TOP into it and gates
// clears on it differing from a zero record (mode 1), clears that must not run.
static void pending_written(uint64_t addr, uint64_t seq);   // below, with the pending table

typedef struct { uint64_t addr, seq; } CounterWrite;

static void counter_cb(void *arg) {
    CounterWrite *w = arg;
    write_counter(w->addr, 0);
    pending_written(w->addr, w->seq);
    free(w);
}

// Reports queued by the replay that the GPU has not finished. The condition of RENDER_COND reads a timestamp one as 0
// and a samples-passed one as unknown.
// The newest report per address, in a table probed over PENDING_PROBE slots from the address's hash: Ocarina issues
// hundreds of occlusion reports a frame and conditions draws on them in the same frame, so a ring of the last 256
// reports had dropped the report a condition referred to, which then read an older query's count (often 0) and culled
// visible ground and trees. It also rotates through tens of thousands of record addresses, so an entry whose value is
// in memory (written >= seq: a condition reads memory anyway) gives its slot to a new address; a table that only grew
// filled up after a minute and conditions on the newer records read the CPU's 1 again.
#define PENDING (1u << 16)
#define PENDING_PROBE 32
// seq: the newest report to addr; written: the newest whose value the completion thread has put in guest memory. A
// report counts as pending until then, not until its fence passes: the callback runs after the fence, and in between
// the record still held what the CPU wrote (Ocarina re-arms its condition records with 1 for the next frame), so a
// condition read 1 where the GPU had written 0, a depth clear ran after the depth copy and frames showed only sky.
// frame: the present count (vk_exec_cond_frame_end, in replay order) when the report was replayed. The game arms its
// condition records from the CPU between frames, so within a frame a timestamp report decides the record's value even
// after it reached memory: Ocarina's CPU, running a frame ahead, had re-armed a record with 1 by the time a later depth
// clear of the frame read it, the clear ran after the depth copy and the frame showed only sky (white flashes).
typedef struct { uint64_t addr, seq, frame; volatile uint64_t written; int unknown, type; } Pending;
static Pending pending[PENDING];
static uint64_t report_seq;
static uint64_t cond_frame;

// Per-frame counts of conditional work (NX2NAVI_LOG_RCOND=1, vk_exec_cond_frame_end logs them): clears and
// draws issued under a render condition, run or skipped.
unsigned vk_cond_clears_run, vk_cond_clears_skip, vk_cond_draws_skip, vk_cond_depth_run;

void vk_exec_cond_frame_end(uint64_t frame) {
    static int on = -1;
    if (on < 0) on = rt_knob("LOG_RCOND", "0");
    if (on && (vk_cond_clears_run || vk_cond_clears_skip || vk_cond_draws_skip))
        rt_log("vk: frame %llu conditions: clears run %u skipped %u, draws skipped %u, depth clears run %u",
               (unsigned long long)frame, vk_cond_clears_run, vk_cond_clears_skip, vk_cond_draws_skip, vk_cond_depth_run);
    vk_cond_clears_run = vk_cond_clears_skip = vk_cond_draws_skip = vk_cond_depth_run = 0;
    if (on == 2) rt_log("SEQ present %llu", (unsigned long long)frame);
    cond_frame++;
}

static Pending *pending_slot(uint64_t addr, int insert);

// Completion thread: the value of report seq to addr is in guest memory.
static void pending_written(uint64_t addr, uint64_t seq) {
    Pending *e = pending_slot(addr, 0);
    if (!e) return;
    uint64_t w = __atomic_load_n(&e->written, __ATOMIC_RELAXED);
    while (w < seq && !__atomic_compare_exchange_n(&e->written, &w, seq, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {}
}

// Lookups run on the replay thread and the completion thread; only the replay thread inserts.
static Pending *pending_slot(uint64_t addr, int insert) {
    if (!addr) return NULL;
    uint32_t i = (uint32_t)((addr * 0x9E3779B97F4A7C15ull) >> 48) & (PENDING - 1);
    Pending *free_ = NULL, *victim = NULL;
    for (unsigned n = 0; n < PENDING_PROBE; n++, i = (i + 1) & (PENDING - 1)) {
        Pending *e = &pending[i];
        uint64_t a = __atomic_load_n(&e->addr, __ATOMIC_ACQUIRE);
        if (a == addr) return e;
        if (!insert) continue;
        if (!a) { if (!free_) free_ = e; continue; }
        if (__atomic_load_n(&e->written, __ATOMIC_ACQUIRE) >= e->seq && e->frame != cond_frame && (!victim || e->seq < victim->seq))
            victim = e;   // done, and not from this frame (whose reports still decide their records)
    }
    Pending *e = free_ ? free_ : victim;
    if (!e) return NULL;   // every probed entry still has a report in flight
    e->seq = 0, e->frame = 0, e->unknown = 0, e->type = -1;
    __atomic_store_n(&e->written, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&e->addr, addr, __ATOMIC_RELEASE);
    return e;
}

// ---- occlusion counts (samples passed)
//
// After a SAMPLES_PASSED reset, the first draw of a pass begins an occlusion query (precise when the device has
// occlusionQueryPrecise); the query ends with the pass, or at the next report or reset of the counter, and the next
// draw begins another. A Vulkan query begun inside a render pass must end inside it. NX2NAVI always resets, draws the
// tested proxies, reports, so counting stops at a report and restarts at the next reset: queries cover only the draws
// between the two, not every pass of the frame. Once the game reports twice without a reset in between, counting stays
// on for good (cumulative NVN semantics; that one report misses the draws since the previous one). The queries closed since the last
// report or reset form a segment, handed to the completion thread with that report or reset: in submit order it sums
// the segment's results into the running count (a reset drops the sum and zeroes the count) and writes the count of a
// report into guest memory. Slots come from one ring; the completion thread reads and host-resets them
// (hostQueryReset, Vulkan 1.2), so the replay records no reset commands. A full ring (no slot read back yet) leaves the
// draw uncounted and the segment's count at least 1, as does a query still unavailable after its fence. Without hostQueryReset the counts stay 0.
#define OQ_N 4096
#define LOG_ONCE(...) do { static volatile char once_; if (!__atomic_exchange_n(&once_, 1, __ATOMIC_RELAXED)) rt_log(__VA_ARGS__); } while (0)
static VkQueryPool oq_pool;
static int oq_state;                     // 0 not set up, 1 counting available, -1 unavailable
static VkQueryControlFlags oq_flags;
static volatile uint8_t oq_busy[OQ_N];   // from begin until the completion thread read and reset the slot
uint32_t oq_slot_id[OQ_N], oq_slot_wh[OQ_N];   // (diagnostic) the first draw each query slot counted
// (diagnostic, NX2NAVI_LOG_OQ) the first counted draw's box constants as the draw was recorded: c3[0] (centre xyz,
// half-size w: the box VS d0345637 places corner*w + xyz) and c4 rows 0-3 (translated world -> clip), the frame
static float oq_slot_box[OQ_N][4], oq_slot_vp[OQ_N][16];
static uint64_t oq_slot_frame[OQ_N];
static float oq_slot_aabb[OQ_N][6];   // (diagnostic, NX2NAVI_OQ_BOXES) the counted box draw's vertices: min xyz, max xyz
static int oq_slot_nv[OQ_N];
extern int64_t vk_dump_drawargs[4], vk_dump_geom[4];
// (diagnostic) NX2NAVI_OQ_WIN=<frame> (NX2NAVI_OQ_WIN_N frames, default 3): every counted draw and every report in
// that window, unsampled: the draw's per-box transform c3[0] (the box VS b8dcfbc2 places pos*w + xyz, then c4) and the
// rect it gives, viewport, scissor, depth/stencil/cull state and render enable; each report's queries and result.
const VkPipeState *vk_oq_cur_s;
static long oqw_a = -2, oqw_n;
void vk_oq_win_set(uint64_t from, long n) {   // F9 with NX2NAVI_DUMP_HOT_N (vk_dump.c): the window follows the dumps
    oqw_a = (long)from, oqw_n = n;
    rt_log("vk: oqwin: frames %ld..%ld (F9)", oqw_a, oqw_a + oqw_n - 1);
}
int vk_oq_win(uint64_t f) {
    long a, n;
    if (oqw_a == -2) {
        const char *e = cfg_get("NX2NAVI_OQ_WIN"), *m = cfg_get("NX2NAVI_OQ_WIN_N");   // numbers: rt_knob only gives on/off
        oqw_a = e && *e ? atol(e) : 0, oqw_n = m && *m ? atol(m) : 3;
        if (oqw_a <= 0) oqw_a = -1; else rt_log("vk: oqwin: frames %ld..%ld", oqw_a, oqw_a + oqw_n - 1);
    }
    a = oqw_a, n = oqw_n;
    return a > 0 && f >= (uint64_t)a && f < (uint64_t)(a + n);
}
static void oq_proj(const float *bb, const float *b, const float *m, char *o, size_t cap) {
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f, d0 = 1e9f, d1 = -1e9f; int behind = 0;
    for (int k = 0; k < 8; k++) {
        float p[3] = {bb[k & 1 ? 3 : 0], bb[k & 2 ? 4 : 1], bb[k & 4 ? 5 : 2]}, c[4];
        for (int i = 0; i < 3; i++) p[i] = p[i] * b[3] + b[i];
        for (int r = 0; r < 4; r++) c[r] = p[0] * m[r] + p[1] * m[4 + r] + p[2] * m[8 + r] + m[12 + r];
        if (c[3] <= 0) { behind++; continue; }
        float sx = (c[0] / c[3] + 1) * 960, sy = (1 - c[1] / c[3]) * 540, z = c[2] / c[3];
        if (sx < x0) x0 = sx; if (sx > x1) x1 = sx; if (sy < y0) y0 = sy; if (sy > y1) y1 = sy;
        if (z < d0) d0 = z; if (z > d1) d1 = z;
    }
    snprintf(o, cap, "x %.0f..%.0f y %.0f..%.0f z %.6f..%.6f behind %d", x0, x1, y0, y1, d0, d1, behind);
}
static uint32_t oq_next;                 // replay: next ring slot
static int oq_counting;                  // replay: draws are counted (a reset since the last report, or oq_keep)
static int oq_reported;                  // replay: a samples-passed report since the last reset
static int oq_keep;                      // replay: the game reported twice without a reset: count every draw
static int oq_open = -1;                 // replay: the slot whose query is active in the open pass
static uint32_t *oq_seg;                 // replay: slots closed since the last report or reset
static uint32_t oq_nseg, oq_capseg;
static int oq_lost;                      // replay: the segment missed a draw (ring full)
static uint64_t oq_count;                // completion thread: samples since the last reset

typedef struct { uint64_t addr, seq; int reset, lost; uint32_t n; uint32_t q[]; } OqSegment;

static int oq_ready(void) {
    if (oq_state) return oq_state > 0;
    VkCtx *v = vk_ctx();
    // vk_device.c enables every feature the device reports, so supported means enabled.
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(v->phys, &f2);
    oq_state = -1;
    if (!f12.hostQueryReset) {
        rt_log("vk: no hostQueryReset: samples-passed counters report 0");
        return 0;
    }
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_OCCLUSION,
                                .queryCount = OQ_N};
    VkResult r = vkCreateQueryPool(v->device, &qi, NULL, &oq_pool);
    if (r != VK_SUCCESS) {
        rt_log("vk: occlusion query pool creation failed (%d): samples-passed counters report 0", (int)r);
        return 0;
    }
    vkResetQueryPool(v->device, oq_pool, 0, OQ_N);
    oq_flags = f2.features.occlusionQueryPrecise ? VK_QUERY_CONTROL_PRECISE_BIT : 0;
    oq_state = 1;
    return 1;
}

static void oq_close(VkExec *ex) {
    if (oq_open < 0) return;
    vkCmdEndQuery(ex->cb, oq_pool, (uint32_t)oq_open);
    ex->recorded++;
    if (oq_nseg == oq_capseg) {
        oq_capseg = oq_capseg ? oq_capseg * 2 : 64;
        oq_seg = realloc(oq_seg, sizeof *oq_seg * oq_capseg);
        if (!oq_seg) rt_fatal("vk: out of memory");
    }
    oq_seg[oq_nseg++] = (uint32_t)oq_open;
    oq_open = -1;
}

void vk_exec_oq_draw(VkExec *ex) {
    if (!oq_counting || oq_open >= 0) return;
    uint32_t q = oq_next % OQ_N;
    if (__atomic_load_n(&oq_busy[q], __ATOMIC_ACQUIRE)) {
        warn_once(W_QUERY, "occlusion query ring full: draws left uncounted");
        oq_lost = 1;
        return;
    }
    oq_busy[q] = 1;
    oq_next++;
    {   // (diagnostic, NX2NAVI_LOG_OQ) the first draw a query counts: its primitive id (stride-0 stream 1) and pass size
        extern uint32_t oq_slot_id[], oq_slot_wh[];
        uint32_t id = 0xffffffffu;
        VkMemSpan ms;
        if (ex->vbuf[1].address && vk_mem_resolve(ex->vbuf[1].address, 4, &ms)) id = *(const uint32_t *)(uintptr_t)ex->vbuf[1].address;
        oq_slot_id[q] = id, oq_slot_wh[q] = ex->pass_w << 16 | ex->pass_h;
        memset(oq_slot_box[q], 0, sizeof oq_slot_box[q]), memset(oq_slot_vp[q], 0, sizeof oq_slot_vp[q]);
        const p_bind_buf *c3 = &ex->ubo[0][0], *c4 = &ex->ubo[0][1];   // vertex stage c3, c4
        if (c3->addr && c3->size >= 16 && vk_mem_resolve(c3->addr, 16, &ms)) memcpy(oq_slot_box[q], (const void *)(uintptr_t)c3->addr, 16);
        if (c4->addr && c4->size >= 64 && vk_mem_resolve(c4->addr, 64, &ms)) memcpy(oq_slot_vp[q], (const void *)(uintptr_t)c4->addr, 64);
        oq_slot_frame[q] = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
        oq_slot_nv[q] = 0;
        static int boxes = -1;
        if (boxes < 0) boxes = rt_knob("OQ_BOXES", "0");
        if (boxes && vk_dump_geom[0] == 2 && vk_dump_drawargs[0] > 0) {   // indexed box draw: its vertices through its indices
            uint32_t va = ex->pipe.vattr[0], off = va >> 8 & 0x7ffff, stream = va >> 27, stride = ex->pipe.vstream[stream & 15].stride;
            uint64_t vb = ex->vbuf[stream & 15].address, ib = (uint64_t)vk_dump_geom[3];
            int it = (int)vk_dump_geom[2], isz = it == 0 ? 1 : it == 1 ? 2 : 4, n = (int)(vk_dump_drawargs[0] < 36 ? vk_dump_drawargs[0] : 36);
            float *bb = oq_slot_aabb[q];
            if (vb && ib && stride && vk_mem_resolve(ib, (uint64_t)n * isz, &ms)) {
                for (int k = 0; k < n; k++) {
                    uint32_t ix = isz == 1 ? ((const uint8_t *)(uintptr_t)ib)[k] : isz == 2 ? ((const uint16_t *)(uintptr_t)ib)[k] : ((const uint32_t *)(uintptr_t)ib)[k];
                    uint64_t a = vb + ((uint64_t)ix + (uint64_t)vk_dump_drawargs[2]) * stride + off;
                    if (!vk_mem_resolve(a, 12, &ms)) break;
                    const float *v = (const float *)(uintptr_t)a;
                    for (int c = 0; c < 3; c++) {
                        if (!oq_slot_nv[q] || v[c] < bb[c]) bb[c] = v[c];
                        if (!oq_slot_nv[q] || v[c] > bb[3 + c]) bb[3 + c] = v[c];
                    }
                    oq_slot_nv[q]++;
                }
            }
        }
    }
    if (vk_oq_win(oq_slot_frame[q])) {
        const VkPipeState *ps = vk_oq_cur_s;
        VkRect2D sc = vk_exec_scissor(ex);
        char pr[160] = "no box vertices";
        if (oq_slot_nv[q]) oq_proj(oq_slot_aabb[q], oq_slot_box[q], oq_slot_vp[q], pr, sizeof pr);
        rt_log("vk: oqwin draw frame %llu q %u: %lld idx, aabb %.1f %.1f %.1f / %.1f %.1f %.1f, c3 %g %g %g w %g -> %s; vp %.0f,%.0f %.0fx%.0f z %.4f..%.4f "
               "scissor %d,%d %ux%u; depth t%d w%d f%d, stencil t%d f%d/%d ref %u mask %x, cull %d front %d, rd %d, render off %d cond %d, pass %ux%u",
               (unsigned long long)oq_slot_frame[q], q, (long long)vk_dump_drawargs[0], oq_slot_aabb[q][0], oq_slot_aabb[q][1], oq_slot_aabb[q][2],
               oq_slot_aabb[q][3], oq_slot_aabb[q][4], oq_slot_aabb[q][5], oq_slot_box[q][0], oq_slot_box[q][1], oq_slot_box[q][2], oq_slot_box[q][3], pr,
               ex->vp[0][0], ex->vp[0][1], ex->vp[0][2], ex->vp[0][3], ex->vp[0][4], ex->vp[0][5], sc.offset.x, sc.offset.y, sc.extent.width, sc.extent.height,
               ps ? (int)ps->ds.dt : -1, ps ? (int)ps->ds.dw : -1, ps ? (int)ps->ds.df : -1, ps ? (int)ps->ds.st : -1, ps ? (int)ps->ds.sf_f : -1,
               ps ? (int)ps->ds.sf_b : -1, ex->st_ref[0], ex->st_vmask[0], ps ? (int)ps->poly.cull : -1, ps ? (int)ps->poly.front : -1,
               ps ? ps->raster_discard : -1, ex->render_disabled, ex->has_render_cond, ex->pass_w, ex->pass_h);
    }
    vkCmdBeginQuery(ex->cb, oq_pool, q, oq_flags);
    { extern FILE *vk_drawlog_file; if (vk_drawlog_file) fprintf(vk_drawlog_file, "OQ %u\n", q); }
    ex->recorded++;
    oq_open = (int)q;
}

void vk_exec_oq_pass_end(VkExec *ex) { oq_close(ex); }

void vk_exec_xfer_shutdown(void) {
    if (oq_state > 0) vkDestroyQueryPool(vk_ctx()->device, oq_pool, NULL);
    oq_pool = VK_NULL_HANDLE, oq_state = 0, oq_next = 0, oq_counting = oq_reported = oq_keep = 0, oq_open = -1;
    oq_nseg = 0, oq_lost = 0, oq_count = 0;
    memset((void *)oq_busy, 0, sizeof oq_busy);
}

static void oq_cb(void *arg) {
    OqSegment *s = arg;
    VkDevice dev = vk_ctx()->device;
    uint64_t sum = 0, v[64 * 2];
    for (uint32_t i = 0; i < s->n;) {
        uint32_t j = i + 1;
        while (j < s->n && j - i < 64 && s->q[j] == s->q[j - 1] + 1) j++;
        // The fence has passed, so a query that is still unavailable will not become available: waiting would block the
        // completion thread, and with it every later callback, for good. It counts as visible (one sample).
        VkResult r = vkGetQueryPoolResults(dev, oq_pool, s->q[i], j - i, sizeof v, v, 2 * sizeof v[0],
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (r != VK_SUCCESS && r != VK_NOT_READY) {
            vk_device_fault_report(r);
            rt_fatal("vk: occlusion query results failed (%d)", (int)r);
        }
        vkResetQueryPool(dev, oq_pool, s->q[i], j - i);
        for (uint32_t k = i; k < j; k++) {
            if (v[2 * (k - i) + 1]) sum += v[2 * (k - i)];
            else {
                sum += 1;
                LOG_ONCE("vk: occlusion query slot %u not available after its fence, counted as visible", s->q[k]);
            }
            __atomic_store_n(&oq_busy[s->q[k]], 0, __ATOMIC_RELEASE);
        }
        i = j;
    }
    if (s->reset) oq_count = 0;
    else {
        oq_count += sum;
        {   // (diagnostic) NX2NAVI_LOG_OQ=1: samples-passed reports a second, how many read 0
            static int on = -1; static unsigned n, zero, noaddr; static uint64_t t0;
            if (on < 0) on = rt_knob("LOG_OQ", "0");
            if (on) {
                n++, zero += oq_count == 0, noaddr += !s->addr;
                static unsigned hb[5]; static uint64_t ht0;   // count buckets: 0, <1k, <100k, <1M, >=1M
                hb[oq_count == 0 ? 0 : oq_count < 1000 ? 1 : oq_count < 100000 ? 2 : oq_count < 1000000 ? 3 : 4]++;
                uint64_t hn = kern_now_ns();
                if (hn - ht0 > 1000000000ull) {
                    rt_log("vk: oq counts: 0:%u <1k:%u <100k:%u <1M:%u >=1M:%u", hb[0], hb[1], hb[2], hb[3], hb[4]);
                    memset(hb, 0, sizeof hb), ht0 = hn;
                }
                {   // tally per first counted draw's primitive id: zero vs nonzero results (logged with the histogram)
                    static unsigned tz[8], tn[8]; static uint64_t tt0;
                    uint32_t fid = s->n ? oq_slot_id[s->q[0]] : 0xffffffffu, slot = fid < 7 ? fid : 7;
                    (oq_count == 0 ? tz : tn)[slot]++;
                    uint64_t now = kern_now_ns();
                    if (now - tt0 > 5000000000ull) {
                        rt_log("vk: oq by first id (zero/nonzero): id0 %u/%u id1 %u/%u id2 %u/%u id3 %u/%u id4 %u/%u id5 %u/%u id6 %u/%u other %u/%u",
                               tz[0], tn[0], tz[1], tn[1], tz[2], tn[2], tz[3], tn[3], tz[4], tn[4], tz[5], tn[5], tz[6], tn[6], tz[7], tn[7]);
                        memset(tz, 0, sizeof tz), memset(tn, 0, sizeof tn), tt0 = now;
                    }
                }
                if (s->n ? vk_oq_win(oq_slot_frame[s->q[0]]) : 0) {
                    char qs[200]; int o = 0;
                    for (uint32_t i = 0; i < s->n && o < 180; i++) o += snprintf(qs + o, sizeof qs - o, " q%u", s->q[i]);
                    rt_log("vk: oqwin report frame %llu addr %#llx seq %llu: %u queries%s%s, reset %d, result available, %llu samples",
                           (unsigned long long)oq_slot_frame[s->q[0]], (unsigned long long)s->addr, (unsigned long long)s->seq, s->n, qs,
                           s->lost ? " (lost draws)" : "", s->reset, (unsigned long long)oq_count);
                }
                {   // NX2NAVI_OQ_BOXES=1: a sample of results (1 zero in 16, up to 300 lines; 1 nonzero in 64, up to 60) with
                    // the box (its vertices' bounds) projected by c4's first matrix: screen rect at 1920x1080 and depth range
                    static int boxes = -1; static unsigned zc, nc, zl, nl;
                    if (boxes < 0) boxes = rt_knob("OQ_BOXES", "0");
                    uint32_t q = s->n ? s->q[0] : 0;
                    int take = boxes && s->n && (oq_count == 0 ? (zc++ % 16 == 0 && zl < 300) : (nc++ % 64 == 0 && nl < 60));
                    if (take) {
                        oq_count == 0 ? zl++ : nl++;
                        const float *bb = oq_slot_aabb[q], *m = oq_slot_vp[q];
                        float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f, dmin = 1e9f, dmax = -1e9f; int behind = 0;
                        for (int k = 0; k < 8 && oq_slot_nv[q]; k++) {
                            float p[3] = {bb[k & 1 ? 3 : 0], bb[k & 2 ? 4 : 1], bb[k & 4 ? 5 : 2]}, c[4];
                            for (int r = 0; r < 4; r++) c[r] = p[0] * m[0 + r] + p[1] * m[4 + r] + p[2] * m[8 + r] + m[12 + r];
                            if (c[3] <= 0) { behind++; continue; }
                            float sx = (c[0] / c[3] + 1) * 960, sy = (1 - c[1] / c[3]) * 540, nz = c[2] / c[3];
                            if (sx < x0) x0 = sx; if (sx > x1) x1 = sx; if (sy < y0) y0 = sy; if (sy > y1) y1 = sy;
                            if (nz < dmin) dmin = nz; if (nz > dmax) dmax = nz;
                        }
                        rt_log("vk: oq box %s frame %llu report %#llx queries %u: %d verts, min %.1f %.1f %.1f max %.1f %.1f %.1f -> screen x %.0f..%.0f y %.0f..%.0f depth %.7f..%.7f (%d corners behind), pass %ux%u, %llu samples",
                               oq_count ? "nonzero" : "ZERO", (unsigned long long)oq_slot_frame[q], (unsigned long long)s->addr, s->n,
                               oq_slot_nv[q], bb[0], bb[1], bb[2], bb[3], bb[4], bb[5], x0, x1, y0, y1, dmin, dmax, behind,
                               oq_slot_wh[q] >> 16, oq_slot_wh[q] & 0xffff, (unsigned long long)oq_count);
                    }
                }
                static unsigned zlog, zall;
                if (oq_count == 0 && zall++ >= 4000 && zlog < 40) {
                    zlog++;
                    char ids[160]; int o = 0;
                    for (uint32_t i = 0; i < s->n && o < 140; i++)
                        o += snprintf(ids + o, sizeof ids - o, " q%u:id%#x %ux%u", s->q[i], oq_slot_id[s->q[i]],
                                      oq_slot_wh[s->q[i]] >> 16, oq_slot_wh[s->q[i]] & 0xffff);
                    rt_log("vk: oq: zero result at %#llx from %u queries:%s%s", (unsigned long long)s->addr, s->n, s->n ? ids : " (no draw counted)",
                           s->lost ? " (lost)" : "");
                }
                uint64_t now = kern_now_ns();
                if (now - t0 > 1000000000ull) {
                    rt_log("vk: oq: %u samples reports (%u read 0, %u to GPU-only memory) in the last second; last %llu, segment %u queries",
                           n, zero, noaddr, (unsigned long long)oq_count, s->n);
                    n = zero = noaddr = 0, t0 = now;
                }
            }
        }
        {   // (diagnostic) with NX2NAVI_DRAWLOG: every samples-passed report, by the frame its first query was recorded in
            extern FILE *vk_drawlog_file;
            if (vk_drawlog_file && s->n)
                fprintf(vk_drawlog_file, "OQR %llu %#llx %llu %u\n", (unsigned long long)oq_slot_frame[s->q[0]],
                        (unsigned long long)s->addr, (unsigned long long)oq_count, s->n);
        }
        if (s->lost && !oq_count) oq_count = 1;
        {   static int force = -1;   // (diagnostic) NX2NAVI_OQ_FORCE=1: every samples-passed report reads visible
            if (force < 0) force = rt_knob("OQ_FORCE", "0");
            if (force && s->addr && oq_count < 100000) { write_counter(s->addr, 100000); pending_written(s->addr, s->seq); free(s); return; }
        }
        if (s->addr) write_counter(s->addr, oq_count), pending_written(s->addr, s->seq), oqw_result(s->addr, s->seq, oq_count);
    }
    free(s);
}

// Hands the segment to the completion thread with a report (addr; 0: no CPU-visible memory) or a reset.
static void oq_segment(VkExec *ex, uint64_t addr, int reset) {
    oq_close(ex);
    OqSegment *s = malloc(sizeof *s + sizeof *s->q * oq_nseg);
    if (!s) rt_fatal("vk: out of memory");
    s->addr = addr, s->seq = report_seq, s->reset = reset, s->lost = oq_lost, s->n = oq_nseg;
    memcpy(s->q, oq_seg, sizeof *s->q * oq_nseg);
    oq_nseg = 0, oq_lost = 0;
    vk_completion_add(ex->fence, oq_cb, s);
}

static void reset_counter(VkExec *ex, const p_reset_counter *p) {
    if (p->type != NVN_COUNTER_TYPE_SAMPLES_PASSED || !oq_ready()) return;
    oq_segment(ex, 0, 1);
    oq_counting = 1, oq_reported = 0;
}

// A samples-passed report: the segment goes with it (addr 0: no CPU-visible memory) and counting stops until the next
// reset, unless reports came without resets before.
static void oq_report(VkExec *ex, uint64_t addr) {
    oq_segment(ex, addr, 0);
    oqw_report(addr, report_seq, cond_frame);
    if (oq_reported && !oq_keep) {
        oq_keep = 1;
        rt_log("vk: samples passed reported twice without a reset: counting every draw from now on");
    }
    oq_reported = 1;
    if (!oq_keep) oq_counting = 0;
}

static void signal_event(VkExec *ex, const p_signal_event *p) {
    EventWrite *e = malloc(sizeof *e);
    if (!e) rt_fatal("vk: out of memory");
    *e = (EventWrite){p->val, p->mode, p->value};
    vk_completion_add(ex->fence, event_cb, e);
}

static void report_counter(VkExec *ex, const p_report_counter *p) {
    int samples = p->type == NVN_COUNTER_TYPE_SAMPLES_PASSED && oq_ready();
    {   // (diagnostic) NX2NAVI_LOG_OQ=1: report types a second
        static int on = -1; static unsigned by[32]; static uint64_t t0;
        if (on < 0) on = rt_knob("LOG_OQ", "0");
        if (on) {
            by[p->type & 31]++;
            uint64_t now = kern_now_ns();
            if (now - t0 > 1000000000ull) {
                char b[256]; int k = 0;
                for (int i = 0; i < 32; i++) if (by[i]) k += snprintf(b + k, sizeof b - k, " type%d:%u", i, by[i]), by[i] = 0;
                if (k) rt_log("vk: counter reports in the last second:%s", b);
                t0 = now;
            }
        }
    }
    report_seq++;   // before oq_report / counter_cb take it
    if (nvn_gpu_only(p->addr)) {   // CPU-invisible pool: no host memory behind it
        if (samples) oq_report(ex, 0);
        return;
    }
    VkMemSpan m;
    if (!vk_mem_resolve(p->addr, sizeof(NVNcounterData), &m))
        rt_fatal("vk: counter report at %#llx is in no memory pool", (unsigned long long)p->addr);
    vk_mem_count(m.cls, VK_MUSE_GPU_WRITE, sizeof(NVNcounterData));
    if (samples) oq_report(ex, p->addr);
    else {
        CounterWrite *a = malloc(sizeof *a);
        if (!a) rt_fatal("vk: out of memory");
        *a = (CounterWrite){p->addr, report_seq};
        vk_completion_add(ex->fence, counter_cb, a);
    }
    Pending *e = pending_slot(p->addr, 1);
    if (e) e->seq = report_seq, e->frame = cond_frame, e->unknown = samples, e->type = (int)p->type;
    else LOG_ONCE("vk: a counter report's table neighbourhood is all in flight: conditions on %#llx read memory",
                  (unsigned long long)p->addr);
}

// ---- conditional rendering

// 1: *v read, 0: not readable, 2: a samples-passed report not in memory yet (unknown)
static int counter_word(uint64_t addr, uint64_t *v) {
    const Pending *e = pending_slot(addr, 0);   // the last report to addr decides
    if (e && !e->unknown && e->frame == cond_frame) return *v = 0, 1;   // a timestamp/statistics report of this frame
    if (e && __atomic_load_n(&e->written, __ATOMIC_ACQUIRE) < e->seq) return *v = 0, e->unknown ? 2 : 1;
    VkMemSpan m;
    if (nvn_gpu_only(addr) || !vk_mem_resolve(addr, sizeof *v, &m)) return 0;
    memcpy(v, (const void *)(uintptr_t)addr, sizeof *v);
    vk_mem_count(m.cls, VK_MUSE_CPU_READ, sizeof *v);
    return 1;
}

// The counter type of the newest report to addr the ring still holds, -1 when there is none (NX2NAVI_LOG_RCOND).
static int last_type(uint64_t addr) {
    const Pending *e = pending_slot(addr, 0);
    return e ? e->type : -1;
}

// NX2NAVI_LOG_RCOND=ab (diagnosis): the words a condition compares and where each came from: "f" a report replayed
// this frame (0), "p" a pending report (0 or unknown), "m" memory, with the newest report's type and frame.
void vk_exec_cond_why(const p_render_cond *c, char *buf, size_t cap) {
    int o = snprintf(buf, cap, "%#llx mode %d:", (unsigned long long)c->addr, c->mode);
    for (int k = 0; k < 2 && o < (int)cap; k++) {
        uint64_t a = c->addr + 16 * (uint64_t)k;
        const Pending *e = pending_slot(a, 0);
        const char *src = !e ? "m" : !e->unknown && e->frame == cond_frame ? "f"
                        : __atomic_load_n(&e->written, __ATOMIC_ACQUIRE) < e->seq ? "p" : "m";
        o += snprintf(buf + o, cap - (size_t)o, " %llu %s(type %d, frame %lld of %llu)",
                      (unsigned long long)*(const uint64_t *)(uintptr_t)a, src, e ? e->type : -1,
                      e ? (long long)e->frame : -1LL, (unsigned long long)cond_frame);
    }
}

// (diagnostic, NX2NAVI_LOG_RC=1) render enable / condition ops every 5 s: counts, results, and per condition the
// counter types last reported at its two words and its mode
static int last_type(uint64_t addr);
static void rc_tally(int cond, int on, const p_render_cond *c) {
    static int log = -1; static unsigned n[2][2], types[4][8][8]; static uint64_t t0;
    if (log < 0) log = rt_knob("LOG_RC", "0");
    if (!log) return;
    n[cond][on != 0]++;
    if (cond && c) {
        int tx = last_type(c->addr), ty = last_type(c->addr + 16);
        types[c->mode & 3][(tx + 1) & 7][(ty + 1) & 7]++;
    }
    uint64_t now = kern_now_ns();
    if (now - t0 < 5000000000ull) return;
    t0 = now;
    char b[512]; int o = 0;
    for (int m = 0; m < 4; m++) for (int x = 0; x < 8; x++) for (int y = 0; y < 8; y++)
        if (types[m][x][y] && o < 440) o += snprintf(b + o, sizeof b - o, " mode%d type%d/type%d:%u", m, x - 1, y - 1, types[m][x][y]);
    rt_log("vk: render ops in 5 s: enable on %u off %u; condition on %u off %u;%s", n[0][1], n[0][0], n[1][1], n[1][0], o ? b : " (no conditions)");
    memset(n, 0, sizeof n), memset(types, 0, sizeof types);
}

// (A/B) NX2NAVI_RENDER_SHARED=1: the old behaviour, where the last of render enable and render condition decides alone
static int render_shared(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("RENDER_SHARED", "0");
    return on;
}

static int cond_enabled(const p_render_cond *c) {
    uint64_t x, y;
    static int all = -1;   // (diagnostic) NX2NAVI_RCOND_ALL=1: every conditional render draws
    if (all < 0) all = rt_knob("RCOND_ALL", "0");
    if (all) return 1;
    int rx = counter_word(c->addr, &x), ry = counter_word(c->addr + 16, &y);
    if (!rx || !ry) {
        warn_once(W_BUFFER, "conditional render counters at %#llx are not readable: rendering", (unsigned long long)c->addr);
        return 1;
    }
    if (rx == 2 || ry == 2) return 1;   // a samples-passed count not done yet
    int on = c->mode == 0 ? x == y : x != y;
    // NX2NAVI_LOG_RCOND=1: one condition in 499 (up to 300 lines), on or off, with the counter types last reported there
    static int log_on = -1, nlog;
    static unsigned ncond;
    if (log_on < 0) log_on = rt_knob("LOG_RCOND", "0");
    if (log_on && ncond++ % 499 == 0 && nlog < 300) {
        nlog++;
        // (pending: the report value has not reached guest memory yet)
        int px = 0, py = 0;   // the word came from a report not done yet (forced to 0) rather than memory
        const Pending *ex_ = pending_slot(c->addr, 0), *ey_ = pending_slot(c->addr + 16, 0);
        px = ex_ && ex_->written < ex_->seq, py = ey_ && ey_->written < ey_->seq;
        // the record's 32 bytes only when all of them resolve (counter_word checked +0 and +16 alone)
        uint64_t m[4] = {0};
        VkMemSpan ms;
        int mem_ok = vk_mem_resolve(c->addr, sizeof m, &ms) && ms.avail >= sizeof m;
        if (mem_ok) memcpy(m, (const void *)(uintptr_t)c->addr, sizeof m);
        rt_log("vk: render condition %s at %#llx mode %d: %llu (type %d%s) vs %llu (type %d%s); memory %s%llx %llx %llx %llx",
               on ? "on" : "off", (unsigned long long)c->addr, c->mode, (unsigned long long)x, last_type(c->addr), px ? ", pending" : "",
               (unsigned long long)y, last_type(c->addr + 16), py ? ", pending" : "", mem_ok ? "" : "(not readable) ",
               (unsigned long long)m[0], (unsigned long long)m[1], (unsigned long long)m[2], (unsigned long long)m[3]);
    }
    return on;
}

// ---- textures

typedef struct {
    const VkTexImage *img;
    VkImageAspectFlags aspect;
    uint32_t level, layer0;
    int bpb, bw, bh;
    int32_t x, y, z;           // region origin
    uint32_t w, h, nz;         // region extent; nz: slices (3D) or layers
} Sub;

// Resolves tex, its view and a copy region to image coordinates. 0 (warned) when the copy cannot be replayed.
static int sub_resolve(const tex_t *t, const view_t *v, int has_v, const NVNcopyRegion *r, int defaults_to_level,
                       Sub *s) {
    memset(s, 0, sizeof *s);
    if (!t) return 0;
    TexVkFormat f;
    if (!tex_vk_format(t->d.format, &f) || f.vk != f.ideal || f.expand_rgb || f.astc_bw) {
        warn_once(W_FORMAT, "copy of format %#x texture (decoded or expanded on upload): skipped", t->d.format);
        return 0;
    }
    s->bpb = nvn_fmt_bpb(t->d.format, &s->bw, &s->bh);
    s->img = vk_texture_get(t);
    if (!s->img || !s->bpb) {
        warn_once(W_FORMAT, "copy of format %#x texture without an image: skipped", t->d.format);
        return 0;
    }
    const VkTexImage *img = s->img;
    if (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) s->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    else if (img->aspect == VK_IMAGE_ASPECT_DEPTH_BIT) s->aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    else {
        warn_once(W_FORMAT, "copy of depth-stencil format %#x texture: skipped", t->d.format);
        return 0;
    }
    s->level = has_v && v->set_levels ? v->base_level : 0;
    s->layer0 = has_v && v->set_layers ? v->min_layer : 0;
    uint32_t lw = max1(img->width >> s->level), lh = max1(img->height >> s->level);
    int k = img->scale > 1 ? img->scale : 1;   // a larger shadow map: the game's region in its texels
    s->x = r->xoffset * k, s->y = r->yoffset * k, s->z = r->zoffset;
    s->nz = r->depth > 0 ? (uint32_t)r->depth : 1;
    s->w = r->width > 0 ? (uint32_t)r->width * k : defaults_to_level ? (s->x < (int32_t)lw ? lw - (uint32_t)s->x : 0) : 0;
    s->h = r->height > 0 ? (uint32_t)r->height * k : defaults_to_level ? (s->y < (int32_t)lh ? lh - (uint32_t)s->y : 0) : 0;
    int is3d = img->type == VK_IMAGE_TYPE_3D;
    uint32_t layers = is3d ? 1 : (uint32_t)s->z + s->nz, first = is3d ? 0 : s->layer0;
    if (s->level >= img->levels || s->x < 0 || s->y < 0 || s->z < 0 || !s->w || !s->h ||
        (uint64_t)s->x + s->w > lw || (uint64_t)s->y + s->h > lh || first + layers > img->layers ||
        (is3d && (uint64_t)s->z + s->nz > max1(img->depth >> s->level))) {
        warn_once(W_REGION, "copy region %d,%d,%d %ux%ux%u outside level %u of a %ux%u texture: skipped", s->x, s->y,
                  s->z, s->w, s->h, s->nz, s->level, img->width, img->height);
        return 0;
    }
    return 1;
}

static VkImageSubresourceLayers sub_layers(const Sub *s) {
    int is3d = s->img->type == VK_IMAGE_TYPE_3D;
    return (VkImageSubresourceLayers){s->aspect, s->level, is3d ? 0 : s->layer0 + (uint32_t)s->z, is3d ? 1 : s->nz};
}

static VkOffset3D sub_offset(const Sub *s) {
    return (VkOffset3D){s->x, s->y, s->img->type == VK_IMAGE_TYPE_3D ? s->z : 0};
}

static VkExtent3D sub_extent(const Sub *s) {
    return (VkExtent3D){s->w, s->h, s->img->type == VK_IMAGE_TYPE_3D ? s->nz : 1};
}

// Buffer side of a texture <-> buffer copy: bytes covered, pitches, Vulkan buffer span (a VRAM mirror when the pool
// has one).
static int buffer_side(const VkExec *ex, const Sub *s, uint64_t addr, uint64_t *size, VkBufferImageCopy *c,
                       VkMemSpan *span) {
    uint32_t wb = (s->w + (uint32_t)s->bw - 1) / (uint32_t)s->bw, hb = (s->h + (uint32_t)s->bh - 1) / (uint32_t)s->bh;
    uint64_t row = ex->copy_stride > 0 ? (uint64_t)ex->copy_stride : (uint64_t)wb * (uint64_t)s->bpb;
    if (row % (uint64_t)s->bpb || row < (uint64_t)wb * (uint64_t)s->bpb) {
        warn_once(W_REGION, "buffer copy row stride %llu does not fit %u blocks of %d bytes: skipped",
                  (unsigned long long)row, wb, s->bpb);
        return 0;
    }
    uint64_t img = row * hb;
    *size = img * (s->nz - 1) + row * (hb - 1) + (uint64_t)wb * (uint64_t)s->bpb;
    VkMemSpan m;
    if (!addr || !vk_mem_resolve(addr, *size, &m) || m.avail < *size) {
        warn_once(W_BUFFER, "texture copy buffer %#llx (%llu bytes) is not one backed span: skipped",
                  (unsigned long long)addr, (unsigned long long)*size);
        return 0;
    }
    VK_MEM_PICK(m);
    *span = m;
    *c = (VkBufferImageCopy){.bufferOffset = m.offset, .bufferRowLength = (uint32_t)(row / (uint64_t)s->bpb) * (uint32_t)s->bw,
                             .bufferImageHeight = hb * (uint32_t)s->bh, .imageSubresource = sub_layers(s),
                             .imageOffset = sub_offset(s), .imageExtent = sub_extent(s)};
    return 1;
}

static void copy_buffer_image(VkExec *ex, const p_copy_b2t *p, int to_image) {
    Sub s;
    if (!sub_resolve(p->tex, &p->view, p->has_view, &p->r, 1, &s)) return;
    if (s.img->scale > 1) {   // the buffer holds the game's texel count, the image more
        warn_once(W_REGION, "buffer copy of a shadow map enlarged by NX2NAVI_SHADOW_SCALE: skipped");
        return;
    }
    uint64_t size;
    VkBufferImageCopy c;
    VkMemSpan m;
    if (!buffer_side(ex, &s, p->src, &size, &c, &m)) return;
    vk_mem_count(m.cls, to_image ? VK_MUSE_COPY_SRC : VK_MUSE_GPU_WRITE, size);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    if (to_image) vkCmdCopyBufferToImage(ex->cb, m.buffer, s.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    else vkCmdCopyImageToBuffer(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, m.buffer, 1, &c);
    ex->recorded++;
    if (cfg_get("NX2NAVI_DUMP_RAW3D") && (s.img->type == VK_IMAGE_TYPE_3D || vk_tex3d_hit(s.img->addr, 1))) {   // diagnosis
        static volatile int32_t n3;
        if (__atomic_add_fetch(&n3, 1, __ATOMIC_RELAXED) <= 400 && s.img->width >= 128)
            rt_log("tex3d copy %s: vkimage %p tex %#llx %ux%ux%u level %u box %d,%d,%d %ux%ux%u buffer %#llx row %llu",
                   to_image ? "buffer->tex" : "tex->buffer", (void *)s.img->image, (unsigned long long)s.img->addr, s.img->width,
                   s.img->height, s.img->depth, s.level, s.x, s.y, s.z, s.w, s.h, s.nz, (unsigned long long)p->src,
                   (unsigned long long)ex->copy_stride);
    }
    if (to_image) vk_texture_copy_written(p->tex, s.level);
    else vk_mem_gpu_wrote(&m, size), note_gpu_write(ex, p->src, size);
}

static void copy_t2t(VkExec *ex, const p_copy_t2t *p) {
    Sub s, d;
    if (!sub_resolve(p->src, &p->sv, p->has_sv, &p->sr, 1, &s)) return;
    if (!sub_resolve(p->dst, &p->dv, p->has_dv, &p->dr, 1, &d)) return;
    d.nz = s.nz;
    int same_size = s.w == d.w && s.h == d.h;
    int same_format = s.img->format == d.img->format;
    if (s.img->samples != d.img->samples && !(s.img->samples > 1 && d.img->samples == 1 && same_size && same_format)) {
        warn_once(W_COPY, "texture copy between %u and %u samples: skipped", s.img->samples, d.img->samples);
        return;
    }
    if (s.aspect != d.aspect || (s.aspect != VK_IMAGE_ASPECT_COLOR_BIT && !same_format)) {
        warn_once(W_COPY, "texture copy between formats %#x and %#x: skipped", p->src->d.format, p->dst->d.format);
        return;
    }
    VkImageSubresourceLayers sl = sub_layers(&s), dl = sub_layers(&d);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    if (s.img->samples > 1 && d.img->samples == 1) {
        VkImageResolve rv = {sl, sub_offset(&s), dl, sub_offset(&d), sub_extent(&s)};
        vkCmdResolveImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rv);
    } else if (same_size && same_format) {
        VkImageCopy cp = {sl, sub_offset(&s), dl, sub_offset(&d), sub_extent(&s)};
        vkCmdCopyImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    } else {   // scaled or converting copy: nearest, or linear when asked for (colour only)
        int is3d = s.img->type == VK_IMAGE_TYPE_3D;
        VkImageBlit bl = {.srcSubresource = sl, .dstSubresource = dl};
        bl.srcOffsets[0] = sub_offset(&s);
        bl.srcOffsets[1] = (VkOffset3D){s.x + (int32_t)s.w, s.y + (int32_t)s.h, is3d ? s.z + (int32_t)s.nz : 1};
        bl.dstOffsets[0] = sub_offset(&d);
        bl.dstOffsets[1] = (VkOffset3D){d.x + (int32_t)d.w, d.y + (int32_t)d.h, is3d ? d.z + (int32_t)d.nz : 1};
        int linear = (p->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) && s.aspect == VK_IMAGE_ASPECT_COLOR_BIT;
        vkCmdBlitImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl,
                       linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    ex->recorded++;
    if (cfg_get("NX2NAVI_DUMP_RAW3D") && (vk_tex3d_hit(d.img->addr, 1) ||
        ((d.img->type == VK_IMAGE_TYPE_3D || s.img->type == VK_IMAGE_TYPE_3D) && d.img->width >= 128))) {   // diagnosis: who fills 3D textures
        static volatile int32_t nt;
        if (__atomic_add_fetch(&nt, 1, __ATOMIC_RELAXED) <= 200)
            rt_log("tex3d copy tex->tex: %p %#llx -> %p %#llx %ux%ux%u", (void *)s.img->image,
                   (unsigned long long)s.img->addr, (void *)d.img->image, (unsigned long long)d.img->addr, d.w, d.h, d.nz);
    }
    vk_texture_copy_written(p->dst, d.level);
    static int lg = -1;
    if (lg < 0) lg = rt_knob("LOG_RCOND", "0");
    if (lg == 2 && (d.img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && d.w >= 1600)
        rt_log("SEQ depth copy %#llx -> %#llx", (unsigned long long)s.img->addr, (unsigned long long)d.img->addr);
}

// ---- buffer to buffer

extern void vk_watch_copy(VkExec *ex, uint64_t src, uint64_t dst, uint64_t size, uint32_t value, int phase);
static void copy_b2b(VkExec *ex, const p_copy_b2b *p) {
    uint64_t src = p->src, dst = p->dst, left = p->size;
    int started = 0;
    vk_watch_copy(ex, p->src, p->dst, p->size, 0, 0);   // NX2NAVI_WATCH (diagnostic)
    if (cfg_get("NX2NAVI_DUMP_RAW3D") && vk_tex3d_hit(p->dst, p->size)) {   // diagnosis
        static volatile int32_t nb;
        if (__atomic_add_fetch(&nb, 1, __ATOMIC_RELAXED) <= 200)
            rt_log("tex3d hit: buffer copy %#llx -> %#llx (%llu bytes) into 3D texture #%d", (unsigned long long)p->src,
                   (unsigned long long)p->dst, (unsigned long long)p->size, vk_tex3d_hit(p->dst, p->size));
    }
    while (left) {
        VkMemSpan s, d;
        if (!vk_mem_resolve(src, 1, &s) || !vk_mem_resolve(dst, 1, &d)) {
            warn_once(W_BUFFER, "buffer copy %#llx -> %#llx (%llu bytes) leaves backed memory: rest skipped",
                      (unsigned long long)p->src, (unsigned long long)p->dst, (unsigned long long)p->size);
            break;
        }
        uint64_t n = left < s.avail ? left : s.avail;
        if (n > d.avail) n = d.avail;
        VK_MEM_PICK(s);
        VK_MEM_PICK(d);
        if (s.buffer == d.buffer && s.offset < d.offset + n && d.offset < s.offset + n) {
            warn_once(W_OVERLAP, "buffer copy %#llx -> %#llx overlaps within one buffer: skipped",
                      (unsigned long long)p->src, (unsigned long long)p->dst);
            break;
        }
        if (!started) vk_exec_pass_end(ex), vk_exec_barrier(ex), started = 1;
        VkBufferCopy bc = {s.offset, d.offset, n};
        vkCmdCopyBuffer(ex->cb, s.buffer, d.buffer, 1, &bc);
        vk_mem_gpu_wrote(&d, n);
        ex->recorded++;
        vk_mem_count(s.cls, VK_MUSE_COPY_SRC, n);
        vk_mem_count(d.cls, VK_MUSE_GPU_WRITE, n);
        src += n, dst += n, left -= n;
    }
    if (started) note_gpu_write(ex, p->dst, p->size - left);
    vk_watch_copy(ex, p->src, p->dst, p->size, 0, 1);
}

// ---- buffer fills (ClearBuffer, and ReportValue as a 4-byte one): vkCmdFillBuffer in GPU order after a full
// barrier, so the value lands after earlier work, as NVN's semaphore release does; CPU-visible ranges go to
// vk_guest_written at the fence, as copies do. Works in CPU-invisible pools.

static void clear_buf(VkExec *ex, const p_clear_buf *p) {
    if ((p->addr | p->size) & 3) {
        warn_once(W_BUFFER, "buffer fill at %#llx of %llu bytes is not 4-byte aligned: skipped", (unsigned long long)p->addr,
                  (unsigned long long)p->size);
        return;
    }
    uint64_t dst = p->addr, left = p->size;
    int started = 0;
    vk_watch_copy(ex, 0, p->addr, p->size, p->value, 0);   // NX2NAVI_WATCH (diagnostic)
    while (left) {
        VkMemSpan d;
        if (!vk_mem_resolve(dst, 1, &d) || d.offset % 4) {
            warn_once(W_BUFFER, "buffer fill at %#llx (%llu bytes) leaves backed memory: rest skipped",
                      (unsigned long long)p->addr, (unsigned long long)p->size);
            break;
        }
        uint64_t n = left < d.avail ? left : d.avail & ~3ull;
        if (!n) break;
        if (!started) vk_exec_pass_end(ex), vk_exec_barrier(ex), started = 1;
        VK_MEM_PICK(d);
        vkCmdFillBuffer(ex->cb, d.buffer, d.offset, n, p->value);
        vk_mem_gpu_wrote(&d, n);
        ex->recorded++;
        vk_mem_count(d.cls, VK_MUSE_GPU_WRITE, n);
        dst += n, left -= n;
    }
    if (started) note_gpu_write(ex, p->addr, p->size - left);
    vk_watch_copy(ex, 0, p->addr, p->size, p->value, 1);
}

// ---- texture clears

static int int_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
        return 1;
    default: return 0;
    }
}

// Whole levels only (vkCmdClear*Image has no rectangle); partial regions, channel masks and a view that reinterprets
// the format are skipped.
static void clear_tex(VkExec *ex, const p_clear_tex *p) {
    const tex_t *t = p->tex;
    if (!t) return;
    Sub s;
    if (!sub_resolve(t, &p->view, p->has_view, &p->r, 1, &s)) return;
    const VkTexImage *img = s.img;
    int is3d = img->type == VK_IMAGE_TYPE_3D;
    uint32_t lw = max1(img->width >> s.level), lh = max1(img->height >> s.level);
    int reformat = p->has_view && p->view.set_format && p->view.format != t->d.format;
    if (s.x || s.y || s.w != lw || s.h != lh || (p->mask & 0xf) != 0xf || reformat ||
        (is3d && (s.z || s.nz != max1(img->depth >> s.level)))) {
        warn_once(W_CLEAR, "texture clear of region %d,%d %ux%u mask %#x (level %ux%u, view format change %d): skipped",
                  s.x, s.y, s.w, s.h, p->mask, lw, lh, reformat);
        return;
    }
    VkImageSubresourceRange rng = {img->aspect, s.level, 1, is3d ? 0 : s.layer0 + (uint32_t)s.z, is3d ? 1 : s.nz};
    if (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
        rng.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        VkClearColorValue cv;
        memcpy(&cv, &p->c, sizeof cv);
        if (int_format(img->format) != (p->is_uint != 0)) {
            // A value of the other kind that is all zero bits (0.0f is) is the same clear either way: Ocarina clears its
            // rg16ui edge map (480x270) with a float 0 each frame, and skipped, the post chain's edge classification
            // read the previous frames' edges. Other values stay skipped.
            if (cv.uint32[0] | cv.uint32[1] | cv.uint32[2] | cv.uint32[3]) {
                warn_once(W_CLEAR, "texture clear value kind (uint %d) does not match format %d: skipped", p->is_uint,
                          (int)img->format);
                return;
            }
        }
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        vkCmdClearColorImage(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rng);
    } else {
        int both = (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
        VkClearDepthStencilValue dv = {0, 0};
        if (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) dv.depth = p->c.f[0];
        if (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) dv.stencil = both ? p->c.u[1] : p->c.u[0];
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        vkCmdClearDepthStencilImage(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &dv, 1, &rng);
    }
    ex->recorded++;
    vk_texture_gpu_written(t);
}

void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload) {
    if (op != OP_RENDER_ENABLE && op != OP_SIGNAL_EVENT) vk_contact_flush(ex);
    switch (op) {
    case OP_COPY_B2B: copy_b2b(ex, payload); break;
    case OP_COPY_B2T: copy_buffer_image(ex, payload, 1); break;
    case OP_COPY_T2B: copy_buffer_image(ex, payload, 0); break;   // p_copy_b2t: src holds the destination address
    case OP_COPY_T2T: copy_t2t(ex, payload); break;
    case OP_CLEAR_TEX: clear_tex(ex, payload); break;
    case OP_SIGNAL_EVENT: signal_event(ex, payload); break;
    case OP_REPORT_COUNTER: report_counter(ex, payload); break;
    case OP_RESET_COUNTER: reset_counter(ex, payload); break;
    case OP_CLEAR_BUF: clear_buf(ex, payload); break;
    // Explicit enable and the condition are separate controls that must both allow rendering (each op updates its own
    // half; render_disabled is their combination), so a condition set after a disable no longer re-enables drawing.
    case OP_RENDER_ENABLE:
        rc_tally(0, ((const p_flag *)payload)->value, NULL);
        ex->render_off_explicit = !((const p_flag *)payload)->value;
        ex->render_disabled = render_shared() ? ex->render_off_explicit : ex->render_off_explicit || ex->render_off_cond;
        break;
    case OP_RENDER_COND:
        ex->render_cond = *(const p_render_cond *)payload;
        ex->has_render_cond = 1;
        ex->render_off_cond = !cond_enabled(payload);
        rc_tally(1, !ex->render_off_cond, payload);
        ex->render_disabled = render_shared() ? ex->render_off_cond : ex->render_off_explicit || ex->render_off_cond;
        break;
    default: rt_fatal("vk: op %s is not a transfer", nvn_op_name(op));
    }
}
