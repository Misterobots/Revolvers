// (orbitals) NX2NAVI_RHI_LABELS=1: engine names for GPU memory, taken from the console RHI (UE 5.6, NVN2RHI).
//
// The game's renderer creates every buffer, uniform buffer and texture through FDynamicRHI virtual methods, and the
// engine passes a name with each: FRHIBufferCreateDesc / FRHITextureCreateDesc::DebugName, FRHIUniformBufferLayout::Name,
// RHIBindDebugLabelName(resource, name). Those calls are indirect (GDynamicRHI->...), so they go through rt_dispatch,
// where rt_dispatch_wrap puts a host wrapper in front of the lifted method. The wrapper reads the name, calls the
// method, and records which guest memory the new resource got:
//   - exactly, when the method initializes an NVN buffer or texture on the same guest thread (nvnBufferInitialize /
//     nvnTextureInitialize call rhi_label_nvn while the label is set);
//   - by search otherwise: the returned object (and objects it points to, one level) is scanned for 64-bit values
//     that are GPU memory (vk_mem_resolve), e.g. a sub-allocation from a pool the RHI manages itself.
// The table (guest address -> name) is written with every vk_dump frame as rhi_labels.txt, sorted by address, so any
// address in passes.txt or a vsb_* file name can be looked up (exact, or the nearest entry below it).
//
// Slots of the NVN2 FDynamicRHI vtable (main+0x802ebf0; order reconstructed from UE 5.5.4 / 5.8 DynamicRHI.h and
// checked against strings in the binary, see memory note 2026-11-16):
//   41 RHICreateUniformBuffer(this, const void *Contents, const FRHIUniformBufferLayout *Layout, Usage, Validation)
//   44 RHICreateBufferInitializer(this, FRHICommandListBase &, const FRHIBufferCreateDesc &)   (sret in x8)
//   54 RHICreateTextureInitializer(this, FRHICommandListBase &, const FRHITextureCreateDesc &) (sret in x8)
//   68 RHIBindDebugLabelName(this, FRHICommandListBase &, FRHIBuffer *, const TCHAR *)
//   69 RHIBindDebugLabelName(this, FRHICommandListBase &, FRHIUnorderedAccessView *, const TCHAR *)
// (67, the texture overload, is the shared empty default and is not wrapped.) Each slot's function address is read
// from the relocated vtable and must equal the one found offline in this build, else that slot is skipped.
// Sizes: FRHIBufferDesc::Size is the buffer desc's first uint32 (the name at qword 2 matches the 5.8 layout);
// FRHIUniformBufferLayout::ConstantBufferSize is at byte 132. Addresses: the search learns per kind which field of the
// returned object holds the GPU address and then reads only that field. Each address keeps its last two labels with
// the frame they were made in, so memory reused for something else shows as such.
// Names are found by shape, not by struct offset (the 5.6 layouts are not in any source on disk): the first 64-bit
// value in the struct that points to a NUL-terminated UTF-16 string of printable ASCII. The offset found is logged.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "cpu.h"
#include "dispatch.h"
#include "host.h"
#include "platform/platform.h"
#include "vk.h"
#include "nvn_api.h"   // gfx_counters (frame number)

#define VT_OFF 0x802ebf0ull
enum { K_UB, K_BUF, K_TEX, K_NAME_BUF, K_NAME_UAV, K_COUNT };
static const char *const kind_name[K_COUNT] = {"ub", "buf", "tex", "bind_buf", "bind_uav"};
static const struct { int slot; uint64_t fn_off; int kind; } WANT[] = {
    {41, 0x153eb10, K_UB}, {44, 0x1510340, K_BUF}, {54, 0x153bd90, K_TEX}, {68, 0x1510700, K_NAME_BUF}, {69, 0x153d310, K_NAME_UAV},
};
#define NWANT (int)(sizeof WANT / sizeof WANT[0])

static int on;
static uint64_t main_base;
static GuestFn orig[NWANT];
static PlatMutex lock;

// ---- names, interned
#define NAME_CAP 8192
static char *names[NAME_CAP];
static uint64_t name_calls[NAME_CAP][K_COUNT];
static int nnames;
static int name_off_seen[K_COUNT][16];   // histogram of the struct offset (in qwords) the name was found at

static int intern(const char *s) {   // under lock
    uint32_t h = 2166136261u;
    for (const char *p = s; *p; p++) h = (h ^ (uint8_t)*p) * 16777619u;
    static int slot_of[NAME_CAP * 2];   // 0 empty, else index + 1
    for (uint32_t i = h & (NAME_CAP * 2 - 1), n = 0; n < NAME_CAP * 2; i = (i + 1) & (NAME_CAP * 2 - 1), n++) {
        if (!slot_of[i]) {
            if (nnames >= NAME_CAP - 1) return 0;
            names[++nnames] = strdup(s);
            slot_of[i] = nnames + 1;
            return nnames;
        }
        if (!strcmp(names[slot_of[i] - 1], s)) return slot_of[i] - 1;
    }
    return 0;
}

static int guest_ptr(uint64_t v) { return v >= 0x10000 && v < RT_ADDRESS_LIMIT; }

// A NUL-terminated UTF-16 string of 2..127 printable ASCII characters at a; out gets it as ASCII.
static int utf16_at(uint64_t a, char *out, int max) {
    if (!guest_ptr(a) || (a & 1)) return 0;
    uint16_t w[128];
    uint64_t room = 4096 - (a & 4095);   // read within the page first, then the next page if needed
    size_t n = room >= sizeof w ? sizeof w : (size_t)room;
    if (!plat_mem_read(a, w, n)) return 0;
    if (n < sizeof w && !plat_mem_read(a + n, (uint8_t *)w + n, sizeof w - n)) memset((uint8_t *)w + n, 0xff, sizeof w - n);
    int i = 0;
    for (; i < 128 && i < max - 1; i++) {
        if (!w[i]) break;
        if (w[i] < 0x20 || w[i] > 0x7e) return 0;
        out[i] = (char)w[i];
    }
    if (i < 2 || i >= 128 || i >= max - 1) return 0;
    out[i] = 0;
    return 1;
}

// The first qword in [s, s + 8*nq) that points to a name; its index, or -1.
static int find_name(uint64_t s, int nq, char *out, int max) {
    uint64_t q[16];
    if (nq > 16) nq = 16;
    if (!guest_ptr(s) || !plat_mem_read(s, q, (size_t)nq * 8)) return -1;
    for (int i = 0; i < nq; i++)
        if (utf16_at(q[i], out, max)) return i;
    return -1;
}

// ---- the table: guest address -> (name, kind, size)
#define TAB_CAP (1 << 20)
typedef struct { uint64_t size; int name; uint8_t kind, how; uint32_t frame; } Lab;   // how: 0 nvn, 1 search, 2 obj
typedef struct { uint64_t addr; Lab cur, prev; } Ent;   // prev: the label before, for reused memory
static Ent *tab;
static int ntab;

static uint32_t frame_of(void) { return (uint32_t)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1; }

static void record(uint64_t addr, uint64_t size, int name, int kind, int how) {   // under lock
    if (!addr || !tab) return;
    uint64_t h = (addr >> 4) * 0x9E3779B97F4A7C15ull;
    for (uint32_t i = (uint32_t)(h >> 44) & (TAB_CAP - 1), n = 0; n < TAB_CAP; i = (i + 1) & (TAB_CAP - 1), n++) {
        if (tab[i].addr == addr || !tab[i].addr) {
            if (!tab[i].addr) {
                if (ntab >= TAB_CAP * 3 / 4) return;
                ntab++;
            }
            Lab nl = {size, name, (uint8_t)kind, (uint8_t)how, frame_of()};
            if (tab[i].addr == addr) {
                if (tab[i].cur.how == 0 && how != 0) return;   // an exact NVN entry beats a found one
                if (tab[i].cur.name != name || tab[i].cur.kind != kind) tab[i].prev = tab[i].cur;   // reused memory
            }
            tab[i].addr = addr, tab[i].cur = nl;
            return;
        }
    }
}

// ---- the label of a guest thread while it runs a wrapped method (keyed by its Cpu: fibers keep their own)
static struct { Cpu *c; int name; int kind; } cur[256];
static int cur_slot(Cpu *c) { return (int)(((uintptr_t)c >> 6) * 0x9E3779B9u >> 24) & 255; }
static void cur_set(Cpu *c, int name, int kind) {   // under lock
    for (int i = cur_slot(c), n = 0; n < 256; i = (i + 1) & 255, n++)
        if (!cur[i].c || cur[i].c == c) { cur[i].c = c, cur[i].name = name, cur[i].kind = kind; return; }
}
static void cur_clear(Cpu *c) {   // under lock
    for (int i = cur_slot(c), n = 0; n < 256; i = (i + 1) & 255, n++)
        if (cur[i].c == c) { cur[i].c = NULL; return; }
}

// nvnBufferInitialize / nvnTextureInitialize (nvn_api_*.c): the exact memory of a resource made inside a wrapped call.
void rhi_label_nvn(Cpu *c, uint64_t addr, uint64_t size, int is_tex) {
    if (!on || !addr) return;
    plat_mutex_lock(&lock);
    for (int i = cur_slot(c), n = 0; n < 256; i = (i + 1) & 255, n++) {
        if (!cur[i].c) break;
        if (cur[i].c == c) { record(addr, size, cur[i].name, cur[i].kind, 0); break; }
    }
    (void)is_tex;
    plat_mutex_unlock(&lock);
}

// GPU memory addresses held in the object at o (first 32 qwords), and in objects it points to (first 32 qwords each),
// up to max hits. Never under lock (vk_mem_resolve takes the pool lock).
// path[k]: where hit k was, as (outer qword + 1) * 64 + inner qword for a hit one object down, else the qword index.
static int search(uint64_t o, uint64_t *hit, int *path, int max, int depth, int outer) {
    uint64_t q[32];
    int n = 0;
    if (!guest_ptr(o) || !plat_mem_read(o, q, sizeof q)) return 0;
    for (int i = 0; i < 32 && n < max; i++) {
        VkMemSpan ms;
        if (!guest_ptr(q[i])) continue;
        if (vk_mem_resolve(q[i], 1, &ms)) {
            int dup = 0;
            for (int k = 0; k < n; k++) dup |= hit[k] == q[i];
            if (!dup) path[n] = outer >= 0 ? (outer + 1) * 64 + i : i, hit[n++] = q[i];
        } else if (depth > 0 && !(q[i] & 7) && (q[i] < main_base || q[i] >= main_base + 0x10000000ull)) {
            n += search(q[i], hit + n, path + n, max - n, depth - 1, i);
        }
    }
    return n;
}

// Learning which field holds the address: the path seen in the most of the first LEARN calls that found anything; it
// is fixed if it was in at least 90% of them, and from then on only that field is read ("obj" entries).
#define NPATH (33 * 64)
#define LEARN 4096
static uint32_t path_seen[K_COUNT][NPATH];   // under lock
static int path_calls[K_COUNT], path_fixed[K_COUNT];   // path_fixed: learned path + 1; 0 while learning or unsettled

static uint64_t at_path(uint64_t o, int p) {   // the GPU address at a learned path, 0 if none
    uint64_t v;
    VkMemSpan ms;
    if (p >= 64) {
        if (!plat_mem_read(o + 8ull * (p / 64 - 1), &o, 8) || !guest_ptr(o)) return 0;
        p &= 63;
    }
    if (!plat_mem_read(o + 8ull * p, &v, 8) || !guest_ptr(v) || !vk_mem_resolve(v, 1, &ms)) return 0;
    return v;
}

static void learn(int kind) {   // under lock, at the LEARN-th call
    int top[4];
    for (int t = 0; t < 4; t++) {   // the four most seen paths
        int best = -1;
        for (int p = 0; p < NPATH; p++) {
            int taken = 0;
            for (int u = 0; u < t; u++) taken |= top[u] == p;
            if (!taken && (best < 0 || path_seen[kind][p] > path_seen[kind][best])) best = p;
        }
        top[t] = best;
    }
    int ok = path_seen[kind][top[0]] * 10u >= LEARN * 9u;
    rt_log("rhi_label: %s address path %s: %d (%u of %d calls); next %d (%u), %d (%u), %d (%u)", kind_name[kind],
           ok ? "learned" : "NOT settled, searching on", top[0], path_seen[kind][top[0]], LEARN, top[1],
           path_seen[kind][top[1]], top[2], path_seen[kind][top[2]], top[3], path_seen[kind][top[3]]);
    if (ok) __atomic_store_n(&path_fixed[kind], top[0] + 1, __ATOMIC_RELEASE);
}

static void note(Cpu *c, int kind, const char *nm, int off, uint64_t obj, int depth, uint64_t size) {
    uint64_t hit[4];
    int path[4], n = 0, fixed = __atomic_load_n(&path_fixed[kind], __ATOMIC_ACQUIRE);
    if (obj && fixed) {
        if ((hit[0] = at_path(obj, fixed - 1))) n = 1;
    } else if (obj) {
        n = search(obj, hit, path, 4, depth, -1);
    }
    plat_mutex_lock(&lock);
    int id = intern(nm);
    name_calls[id][kind]++;
    if (off >= 0 && off < 16) name_off_seen[kind][off]++;
    if (fixed) {
        if (n) record(hit[0], size, id, kind, 2);
    } else {
        for (int i = 0; i < n; i++) record(hit[i], size, id, kind, 1), path_seen[kind][path[i]]++;
        if (n && ++path_calls[kind] == LEARN) learn(kind);
    }
    plat_mutex_unlock(&lock);
    (void)c;
}

static void begin(Cpu *c, int kind, const char *nm) {
    plat_mutex_lock(&lock);
    cur_set(c, intern(nm), kind);
    plat_mutex_unlock(&lock);
}
static void end(Cpu *c) {
    plat_mutex_lock(&lock);
    cur_clear(c);
    plat_mutex_unlock(&lock);
}

// The wrappers: arguments are copied before the call (the method clobbers x0-x18), the result looked at after.
static void w_ub(Cpu *c) {
    uint64_t layout = c->x[2], sret = c->x[8];
    char nm[128] = "?";
    int off = find_name(layout, 16, nm, sizeof nm);
    begin(c, K_UB, nm);
    orig[0](c);
    end(c);
    uint32_t sz = 0;   // FRHIUniformBufferLayout::ConstantBufferSize, byte 132 (5.5 and 5.8 layouts agree)
    if (guest_ptr(layout) && !plat_mem_read(layout + 132, &sz, 4)) sz = 0;
    note(c, K_UB, nm, off, guest_ptr(sret) ? LD64(sret) : 0, 0, sz <= 0x10000 ? sz : 0);
}
static void w_desc(Cpu *c, int w, int kind) {
    uint64_t desc = c->x[2], sret = c->x[8];
    char nm[128] = "?";
    int off = find_name(desc, 16, nm, sizeof nm);
    begin(c, kind, nm);
    orig[w](c);
    end(c);
    uint32_t sz = 0;   // FRHIBufferDesc::Size, the desc's first field
    if (kind == K_BUF && guest_ptr(desc) && !plat_mem_read(desc, &sz, 4)) sz = 0;
    note(c, kind, nm, off, guest_ptr(sret) ? sret : 0, 1, sz);   // the initializer struct, and the resource it holds
}
static void w_buf(Cpu *c) { w_desc(c, 1, K_BUF); }
static void w_tex(Cpu *c) { w_desc(c, 2, K_TEX); }
static void w_bind(Cpu *c, int w, int kind) {
    uint64_t res = c->x[2], np = c->x[3];
    char nm[128] = "?";
    utf16_at(np, nm, sizeof nm);
    note(c, kind, nm, -1, res, 0, 0);
    orig[w](c);
}
static void w_bind_buf(Cpu *c) { w_bind(c, 3, K_NAME_BUF); }
static void w_bind_uav(Cpu *c) { w_bind(c, 4, K_NAME_UAV); }
static const GuestFn WRAP[NWANT] = {w_ub, w_buf, w_tex, w_bind_buf, w_bind_uav};

// ---- NX2NAVI_RHI_STAGING=1: every GPU->CPU readback the engine reads, as it reads it. FRHIGPUBufferReadback::Lock
// (Niagara's GPU particle counts among others) maps a staging buffer with LockStagingBuffer_RenderThread (slot 32;
// RHILockStagingBuffer, slot 30, on other threads). Each lock goes to logs/rhi_staging.log: frame, slot, staging
// object, fence, offset, size, the returned address and up to 32 of its first uint32 values (what the engine sees).
// A count that reads 0 (or stale) in the frames where an effect vanishes shows the readback path is at fault.
static GuestFn stg_orig[2];
static _Thread_local int stg_in32;
static FILE *stg_f;
static uint64_t stg_lines;
static void stg_log(int slot, uint64_t staging, uint64_t fence, uint32_t off, uint32_t size, uint64_t ptr) {
    if (!stg_f || stg_lines > 2000000) return;
    // Every word of small readbacks (the Niagara GPU instance counts are ~1 KiB): a slot that reads 0 only in the
    // frames an effect vanishes names that effect's count. Large ones (a fixed 32764-byte list) keep 32 words.
    uint32_t v[1024];
    int n = size <= sizeof v ? (int)(size / 4) : 32;
    if (slot == 30 && stg_in32) n = 0;   // slot 32 calls slot 30: log the values once, on the slot-32 line
    if (!guest_ptr(ptr) || !plat_mem_read(ptr, v, (size_t)n * 4)) n = 0;
    plat_mutex_lock(&lock);
    fprintf(stg_f, "f%u s%d stg %llx fence %llx off %u size %u -> %llx:", frame_of(), slot, (unsigned long long)staging,
            (unsigned long long)fence, off, size, (unsigned long long)ptr);
    for (int i = 0; i < n; i++) fprintf(stg_f, " %u", v[i]);
    fprintf(stg_f, "\n");
    if (!(++stg_lines & 255)) fflush(stg_f);
    plat_mutex_unlock(&lock);
}
static void w_stg30(Cpu *c) {   // void *RHILockStagingBuffer(this, Staging, Fence, Offset, Size)
    uint64_t s = c->x[1], fe = c->x[2];
    uint32_t off = (uint32_t)c->x[3], size = (uint32_t)c->x[4];
    stg_orig[0](c);
    stg_log(30, s, fe, off, size, c->x[0]);
}
static void w_stg32(Cpu *c) {   // void *LockStagingBuffer_RenderThread(this, RHICmdList, Staging, Fence, Offset, Size)
    uint64_t s = c->x[2], fe = c->x[3];
    uint32_t off = (uint32_t)c->x[4], size = (uint32_t)c->x[5];
    stg_in32 = 1;
    stg_orig[1](c);
    stg_in32 = 0;
    stg_log(32, s, fe, off, size, c->x[0]);
}
static void stg_install(void) {
    const char *e = cfg_get("NX2NAVI_RHI_STAGING");
    if (!e || !*e || *e == '0') return;
    static const struct { int slot; uint64_t fn_off; } S[2] = {{30, 0x151b350}, {32, 0x1582090}};
    static const GuestFn W[2] = {w_stg30, w_stg32};
    stg_f = fopen("logs/rhi_staging.log", "w");
    if (!stg_f) { rt_log("rhi_label: cannot write logs/rhi_staging.log"); return; }
    for (int i = 0; i < 2; i++) {
        uint64_t fn = LD64(main_base + VT_OFF + 8ull * S[i].slot);
        if (fn != main_base + S[i].fn_off) {
            rt_log("rhi_label: staging slot %d holds %#llx, not main+%#llx: skipped", S[i].slot, (unsigned long long)fn,
                   (unsigned long long)S[i].fn_off);
            continue;
        }
        rt_dispatch_wrap(fn, W[i], &stg_orig[i]);
        rt_log("rhi_label: staging lock slot %d wrapped -> logs/rhi_staging.log", S[i].slot);
    }
}

void rhi_label_install(void) {
    const char *e = cfg_get("NX2NAVI_RHI_LABELS");
    if (!e || !*e || *e == '0') return;
    for (int i = 0; i < rt_layout_count; i++)
        if (!strcmp(rt_layout[i].name, "main")) main_base = rt_layout[i].base;
    if (!main_base) { rt_log("rhi_label: no main module"); return; }
    tab = calloc(TAB_CAP, sizeof *tab);
    if (!tab) { rt_log("rhi_label: no memory for the table"); return; }
    plat_mutex_init(&lock);
    int n = 0;
    for (int i = 0; i < NWANT; i++) {
        uint64_t fn = LD64(main_base + VT_OFF + 8ull * WANT[i].slot);
        if (fn != main_base + WANT[i].fn_off) {
            rt_log("rhi_label: vtable slot %d holds %#llx, not main+%#llx: skipped", WANT[i].slot, (unsigned long long)fn,
                   (unsigned long long)WANT[i].fn_off);
            continue;
        }
        rt_dispatch_wrap(fn, WRAP[i], &orig[i]);
        n++;
    }
    on = n > 0;
    stg_install();
    rt_log("rhi_label: %d of %d RHI methods wrapped (NVN2 FDynamicRHI vtable main+%#llx)", n, NWANT, VT_OFF);
}

// After rt_dispatch_init: a wrap whose address was no function start has no original; never call through NULL.
static void check_origs(void) {
    static int done;
    if (done) return;
    done = 1;
    for (int i = 0; i < NWANT; i++)
        if (!orig[i]) rt_log("rhi_label: slot %d has no lifted function (not wrapped)", WANT[i].slot);
}

static int by_addr(const void *a, const void *b) {
    uint64_t x = ((const Ent *)a)->addr, y = ((const Ent *)b)->addr;
    return x < y ? -1 : x > y;
}

// vk_dump: the table as <dir>/rhi_labels.txt, sorted by address, then the names with call counts per kind.
void rhi_label_write(const char *dir) {
    if (!on) return;
    check_origs();
    plat_mutex_lock(&lock);
    Ent *s = malloc((size_t)(ntab ? ntab : 1) * sizeof *s);
    int n = 0;
    if (s)
        for (int i = 0; i < TAB_CAP; i++)
            if (tab[i].addr) s[n++] = tab[i];
    char path[600];
    snprintf(path, sizeof path, "%s/rhi_labels.txt", dir);
    FILE *f = fopen(path, "w");
    if (f && s) {
        qsort(s, (size_t)n, sizeof *s, by_addr);
        static const char *const how_name[3] = {"nvn", "search", "obj"};
        fprintf(f, "# v2: guest address, size (0 unknown), kind, how (nvn: exact NVN init in the call; obj: the learned "
                   "field of the returned object; search: any GPU address found in it), frame made, name\n"
                   "#     then '| prev' with the label the address had before (memory reused for something else)\n");
        fprintf(f, "# dump frame %u; learned paths (-1 none):", frame_of());
        for (int k = 0; k < K_COUNT; k++) fprintf(f, " %s=%d", kind_name[k], path_fixed[k] - 1);
        fprintf(f, "\n");
        for (int i = 0; i < n; i++) {
            const Lab *a = &s[i].cur, *b = &s[i].prev;
            fprintf(f, "%012llx %8llx %-8s %-6s %6u %s", (unsigned long long)s[i].addr, (unsigned long long)a->size,
                    kind_name[a->kind], how_name[a->how], a->frame, names[a->name] ? names[a->name] : "?");
            if (b->frame)
                fprintf(f, " | prev %llx %s %s %u %s", (unsigned long long)b->size, kind_name[b->kind], how_name[b->how],
                        b->frame, names[b->name] ? names[b->name] : "?");
            fprintf(f, "\n");
        }
        fprintf(f, "\n# name offsets found (qword index in the desc/layout: count)\n");
        for (int k = 0; k < K_COUNT; k++) {
            fprintf(f, "# %s:", kind_name[k]);
            for (int q = 0; q < 16; q++) if (name_off_seen[k][q]) fprintf(f, " %d:%d", q, name_off_seen[k][q]);
            fprintf(f, "\n");
        }
        fprintf(f, "\n# calls per name: ub buf tex bind_buf bind_uav\n");
        for (int i = 1; i <= nnames; i++)
            fprintf(f, "# %8llu %8llu %8llu %8llu %8llu  %s\n", (unsigned long long)name_calls[i][0],
                    (unsigned long long)name_calls[i][1], (unsigned long long)name_calls[i][2],
                    (unsigned long long)name_calls[i][3], (unsigned long long)name_calls[i][4], names[i]);
    }
    if (f) fclose(f);
    plat_mutex_unlock(&lock);
    free(s);
    rt_log("rhi_label: %d addresses, %d names -> %s", n, nnames, path);
}
