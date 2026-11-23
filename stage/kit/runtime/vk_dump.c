// NX2NAVI_DUMP=<frame>: a frame dump for the Vulkan back end, which RenderDoc cannot capture (it hides
// VK_EXT_external_memory_host). Between presents frame-1 and frame, the replay's command buffers also copy, after each
// render pass, its colour and depth targets and, after each dispatch, the storage images it writes into host buffers;
// the completion thread turns them into 8-bit PNGs once the submission is done. 3D textures (LUTs, volumes) sampled or
// written in the frame are copied once each, slices side by side. Files in NX2NAVI_DUMP_DIR (default "dump"):
//   pass_<seq>_t<i>_<w>x<h>_<vkformat>_<name>.png   colour target i (td: depth; img<slot>: storage image; lut<id>: 3D;
//                                                    a BC target drawn through a block view: its blocks as that view)
//   ..._tm.png                                       float formats: x/(1+x) of the same image (the plain one clamps);
//                                                    in both, a pixel with a NaN channel is magenta
//   passes.txt   per pass or dispatch: targets, draws, each distinct pipeline + texture set with its shaders' SPIR-V
//                hashes (sh_<hash>.spv written once) and the bound textures with their samplers
//   images.txt   per image: channel min/max, distinct texel values over the image and over its top quarter
// Sampled 8-bit 2D textures up to TEX2D_CAP texels are copied once each as pass_<seq>_tex<id>_... at the end of the pass
// that samples them; texture lines carry the storage address, mirror state, upload count and the image's and slot's
// swizzle. Copies and texture clears are logged as "copy/clear after pass N". NX2NAVI_DUMP_LOG=<from> also logs the
// frames from..frame-1 to passes.txt, with PNGs only of targets up to 64K texels. NX2NAVI_DUMP_BUF=<addr>:<size>[:...]
// snapshots one GPU buffer range after every logged pass (buf_<pass>.bin and bufs.txt, see buf_snap).
// PNGs are opaque (alpha dropped). Integer formats are scaled by the image maximum, depth by its min..max range, SNORM
// as x/2+1/2, UNORM and sRGB bytes written as stored. Caps: MAX_PASSES passes copied, PNG_CAP bytes of PNGs.
// NX2NAVI_DUMP_TEX=<id>|<w>x<h>[,...]: those texture slots, or 2D textures of that size, are copied where first sampled,
// whatever their format.
// NX2NAVI_DUMP_UBO=<n>: each listed pipeline also prints the first n vec4 of its user uniform blocks (c3 and up; 1 means 16,
// the cap is 4096), read from guest memory when the entry is listed. A block the submission rewrites with UpdateUniformBuffer shows its value from
// before the submission (vk_exec_draw.c upd_note).
// NX2NAVI_DUMP_SKIP_RUNS=1: a render pass with the same targets as the render pass before it is logged but not copied
// (the terrain decals draw one pass each into the G-buffer and would use up both caps).
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"

#include "vk_exec.h"
#include "flight.h"
uint64_t vk_exec_cs_storage_addr(const VkExec *ex, const GfxShaderRes *r);
uint64_t vk_exec_gfx_storage(VkExec *ex, int st, const GfxShaderRes *r, uint64_t *size);
uint64_t vk_exec_ubo_tex_handle(const VkExec *ex, int st, const GfxShaderRes *r);   // vk_exec_draw.c
int vk_exec_raw_9e5(VkImage img, VkFormat f);                               // vk_exec_draw.c
#include "platform/platform.h"

#define MAX_PASSES 4000
#define PNG_CAP (10ull << 30)
#define INFLIGHT_CAP (8ull << 30)   // host buffers waiting for their submission
#define MAX_ENTS 48                 // distinct pipeline + texture sets listed per pass
#define MAX_LUTS 256                // 3D textures and small sampled 8-bit 2D textures dumped per frame
#define TEX2D_CAP (1u << 20)        // texels of a sampled 2D texture that is dumped

int vk_dump_armed;
static uint64_t dump_frame, log_from;   // log_from: NX2NAVI_DUMP_LOG, first frame logged to passes.txt without images
static char dir[400];
static char base_dir[400];             // NX2NAVI_DUMP_DIR; with NX2NAVI_DUMP_WHITE each attempt goes to base_dir/f<frame>
static uint64_t white_last;            // NX2NAVI_DUMP_WHITE: the frame dumped last (0: none), checked 3 presents on
static int white_tries, white_caught;
static uint64_t dump_after;             // NX2NAVI_DUMP_AFTER
static FILE *passes_f, *images_f;
static int seq, finished, copied;      // copied: passes and dispatches of the dump frame whose images were copied
static void vsb_note_ssbo(uint64_t addr, int idx);   // NX2NAVI_DUMP_VSBUF, below
static void vsb_note_cs_ssbo(uint64_t addr, int idx);
static int skip_runs, dump_ubo;        // NX2NAVI_DUMP_SKIP_RUNS, NX2NAVI_DUMP_UBO (vec4 per block, 0 for none)
static uint32_t want_tex[16][2];       // NX2NAVI_DUMP_TEX: slot id and 0, or width and height
static int nwant_tex;
static uint64_t last_targets;          // targets_sig of the last render pass; 0 after a dispatch
static const char *pass_skip;          // why the current pass copies nothing, or NULL
static volatile int64_t png_bytes, inflight;
static PlatMutex rec_lock = PLAT_MUTEX_INIT, img_lock = PLAT_MUTEX_INIT;

// ---- formats

enum { K_UNORM, K_SNORM, K_UINT, K_SINT, K_FLOAT, K_RGB10A2, K_BGR10A2, K_RG11B10, K_E5, K_R5G6B5, K_D16, K_D24, K_D32F };
enum { M_UNORM, M_SNORM, M_INT, M_FLOAT, M_DEPTH };
typedef struct { VkFormat vk; const char *name; uint8_t bytes, nc, bits, kind, bgr; } Fmt;

#define F(v, n, by, nc, bi, k) {VK_FORMAT_##v, n, by, nc, bi, k, 0}
static const Fmt fmts[] = {
    F(R8_UNORM, "r8", 1, 1, 8, K_UNORM), F(R8_SNORM, "r8sn", 1, 1, 8, K_SNORM), F(R8_UINT, "r8ui", 1, 1, 8, K_UINT),
    F(R8_SINT, "r8i", 1, 1, 8, K_SINT), F(R8_SRGB, "r8srgb", 1, 1, 8, K_UNORM),
    F(R8G8_UNORM, "rg8", 2, 2, 8, K_UNORM), F(R8G8_SNORM, "rg8sn", 2, 2, 8, K_SNORM),
    F(R8G8_UINT, "rg8ui", 2, 2, 8, K_UINT), F(R8G8_SINT, "rg8i", 2, 2, 8, K_SINT),
    F(R8G8B8A8_UNORM, "rgba8", 4, 4, 8, K_UNORM), F(R8G8B8A8_SNORM, "rgba8sn", 4, 4, 8, K_SNORM),
    F(R8G8B8A8_UINT, "rgba8ui", 4, 4, 8, K_UINT), F(R8G8B8A8_SINT, "rgba8i", 4, 4, 8, K_SINT),
    F(R8G8B8A8_SRGB, "rgba8srgb", 4, 4, 8, K_UNORM),
    {VK_FORMAT_B8G8R8A8_UNORM, "bgra8", 4, 4, 8, K_UNORM, 1}, {VK_FORMAT_B8G8R8A8_SRGB, "bgra8srgb", 4, 4, 8, K_UNORM, 1},
    F(A2B10G10R10_UNORM_PACK32, "rgb10a2", 4, 4, 10, K_RGB10A2), F(A2R10G10B10_UNORM_PACK32, "bgr10a2", 4, 4, 10, K_BGR10A2),
    F(R5G6B5_UNORM_PACK16, "r5g6b5", 2, 3, 5, K_R5G6B5),
    F(R16_UNORM, "r16", 2, 1, 16, K_UNORM), F(R16_SNORM, "r16sn", 2, 1, 16, K_SNORM), F(R16_UINT, "r16ui", 2, 1, 16, K_UINT),
    F(R16_SINT, "r16i", 2, 1, 16, K_SINT), F(R16_SFLOAT, "r16f", 2, 1, 16, K_FLOAT),
    F(R16G16_UNORM, "rg16", 4, 2, 16, K_UNORM), F(R16G16_SNORM, "rg16sn", 4, 2, 16, K_SNORM),
    F(R16G16_UINT, "rg16ui", 4, 2, 16, K_UINT), F(R16G16_SINT, "rg16i", 4, 2, 16, K_SINT),
    F(R16G16_SFLOAT, "rg16f", 4, 2, 16, K_FLOAT),
    F(R16G16B16A16_UNORM, "rgba16", 8, 4, 16, K_UNORM), F(R16G16B16A16_SNORM, "rgba16sn", 8, 4, 16, K_SNORM),
    F(R16G16B16A16_UINT, "rgba16ui", 8, 4, 16, K_UINT), F(R16G16B16A16_SINT, "rgba16i", 8, 4, 16, K_SINT),
    F(R16G16B16A16_SFLOAT, "rgba16f", 8, 4, 16, K_FLOAT),
    F(R32_UINT, "r32ui", 4, 1, 32, K_UINT), F(R32_SINT, "r32i", 4, 1, 32, K_SINT), F(R32_SFLOAT, "r32f", 4, 1, 32, K_FLOAT),
    F(R32G32_UINT, "rg32ui", 8, 2, 32, K_UINT), F(R32G32_SINT, "rg32i", 8, 2, 32, K_SINT),
    F(R32G32_SFLOAT, "rg32f", 8, 2, 32, K_FLOAT),
    F(R32G32B32A32_UINT, "rgba32ui", 16, 4, 32, K_UINT), F(R32G32B32A32_SINT, "rgba32i", 16, 4, 32, K_SINT),
    F(R32G32B32A32_SFLOAT, "rgba32f", 16, 4, 32, K_FLOAT),
    F(B10G11R11_UFLOAT_PACK32, "r11g11b10f", 4, 3, 11, K_RG11B10), F(E5B9G9R9_UFLOAT_PACK32, "rgb9e5", 4, 3, 9, K_E5),
    // depth aspect of depth formats, as vkCmdCopyImageToBuffer writes it
    F(D16_UNORM, "d16", 2, 1, 16, K_D16), F(D16_UNORM_S8_UINT, "d16s8", 2, 1, 16, K_D16),
    F(X8_D24_UNORM_PACK32, "d24", 4, 1, 24, K_D24), F(D24_UNORM_S8_UINT, "d24s8", 4, 1, 24, K_D24),
    F(D32_SFLOAT, "d32f", 4, 1, 32, K_D32F), F(D32_SFLOAT_S8_UINT, "d32fs8", 4, 1, 32, K_D32F),
};
#undef F

static const Fmt *fmt_of(VkFormat f) {
    for (size_t i = 0; i < sizeof fmts / sizeof fmts[0]; i++)
        if (fmts[i].vk == f) return &fmts[i];
    return NULL;
}

static const char *fmt_name(VkFormat f, char tmp[16]) {
    const Fmt *d = fmt_of(f);
    if (d) return d->name;
    snprintf(tmp, 16, "vk%d", (int)f);
    return tmp;
}

static int mode_of(const Fmt *f) {
    switch (f->kind) {
    case K_SNORM: return M_SNORM;
    case K_UINT: case K_SINT: return M_INT;
    case K_FLOAT: case K_RG11B10: case K_E5: return M_FLOAT;
    case K_D16: case K_D24: case K_D32F: return M_DEPTH;
    default: return M_UNORM;
    }
}

// Unsigned small float with a 5-bit exponent and mbits of mantissa (half: 10, R11G11B10: 6 and 5).
static float ufloat(uint32_t v, int mbits) {
    uint32_t e = v >> mbits, m = v & ((1u << mbits) - 1);
    if (e == 0) return ldexpf((float)m, -14 - mbits);
    if (e == 31) return m ? NAN : INFINITY;
    return ldexpf((float)(m | 1u << mbits), (int)e - 15 - mbits);
}

static int32_t sext(uint32_t v, int bits) { return bits >= 32 ? (int32_t)v : (int32_t)(v << (32 - bits)) >> (32 - bits); }

static void decode(const Fmt *f, const uint8_t *p, float o[4]) {
    uint32_t u = 0;
    o[0] = o[1] = o[2] = 0, o[3] = 1;
    switch (f->kind) {
    case K_RGB10A2: case K_BGR10A2:
        memcpy(&u, p, 4);
        o[0] = (u & 1023) / 1023.f, o[1] = (u >> 10 & 1023) / 1023.f, o[2] = (u >> 20 & 1023) / 1023.f, o[3] = (u >> 30) / 3.f;
        if (f->kind == K_BGR10A2) { float t = o[0]; o[0] = o[2], o[2] = t; }
        return;
    case K_RG11B10:
        memcpy(&u, p, 4);
        o[0] = ufloat(u & 2047, 6), o[1] = ufloat(u >> 11 & 2047, 6), o[2] = ufloat(u >> 22 & 1023, 5);
        return;
    case K_E5: {
        memcpy(&u, p, 4);
        float s = ldexpf(1.f, (int)(u >> 27) - 24);
        o[0] = (u & 511) * s, o[1] = (u >> 9 & 511) * s, o[2] = (u >> 18 & 511) * s;
        return;
    }
    case K_R5G6B5:
        memcpy(&u, p, 2);
        o[0] = (u >> 11 & 31) / 31.f, o[1] = (u >> 5 & 63) / 63.f, o[2] = (u & 31) / 31.f;
        return;
    case K_D16: memcpy(&u, p, 2); o[0] = u / 65535.f; return;
    case K_D24: memcpy(&u, p, 4); o[0] = (u & 0xffffff) / 16777215.f; return;
    case K_D32F: memcpy(&o[0], p, 4); return;
    }
    int cb = f->bits / 8;
    for (int c = 0; c < f->nc; c++) {
        u = 0;
        memcpy(&u, p + c * cb, (size_t)cb);
        float x;
        switch (f->kind) {
        case K_UNORM: x = (float)u / (float)((1ull << f->bits) - 1); break;
        case K_SNORM: x = fmaxf((float)sext(u, f->bits) / (float)((1u << (f->bits - 1)) - 1), -1.f); break;
        case K_UINT: x = (float)u; break;
        case K_SINT: x = (float)sext(u, f->bits); break;
        default:
            if (f->bits == 16) x = (u & 0x8000 ? -1.f : 1.f) * ufloat(u & 0x7fff, 10);
            else memcpy(&x, &u, 4);
        }
        o[c] = x;
    }
    if (f->bgr) { float t = o[0]; o[0] = o[2], o[2] = t; }
}

// ---- conversion on the completion thread

typedef struct {
    VkBuffer buf;
    VmaAllocation alloc;
    const uint8_t *map;
    uint64_t bytes;
    const Fmt *f;
    uint32_t w, h, d;   // d: 3D slices, laid side by side in the PNG
    char name[160];
} Job;

// Distinct raw texel values among rows [y0, y1) of every slice.
static uint64_t distinct(const Job *j, uint32_t y0, uint32_t y1) {
    uint64_t n = (uint64_t)j->w * (y1 - y0) * j->d, cap = 1024;
    while (cap < 2 * n) cap *= 2;
    uint64_t *set = calloc(cap, sizeof *set), count = 0;
    if (!set) return 0;
    for (uint32_t z = 0; z < j->d; z++)
        for (uint32_t y = y0; y < y1; y++)
            for (uint32_t x = 0; x < j->w; x++) {
                const uint8_t *p = j->map + (((uint64_t)z * j->h + y) * j->w + x) * j->f->bytes;
                uint64_t a = 0, b = 0;
                memcpy(&a, p, j->f->bytes < 8 ? j->f->bytes : 8);
                if (j->f->bytes > 8) memcpy(&b, p + 8, j->f->bytes - 8u);
                uint64_t k = (a ^ (b * 0x9E3779B97F4A7C15ull)) + 1;
                if (!k) k = 1;
                uint64_t i = (k * 0xff51afd7ed558ccdull) >> 17;
                for (;; i++) {
                    uint64_t *s = &set[i & (cap - 1)];
                    if (*s == k) break;
                    if (!*s) { *s = k, count++; break; }
                }
            }
    free(set);
    return count;
}

static uint8_t u8(float x) { return x > 0 ? x < 1 ? (uint8_t)(x * 255.f + .5f) : 255 : 0; }

static void write_images(const Job *j) {
    uint32_t pw = j->w * j->d, ph = j->h;
    uint64_t n = (uint64_t)pw * ph;
    float *px = malloc(n * 16);
    uint8_t *out = malloc(n * 4), *isnan_ = calloc(n, 1);   // isnan_: a pixel with a NaN channel, drawn magenta
    if (!px || !out || !isnan_) rt_fatal("vk_dump: out of memory for %s", j->name);
    float mn[4] = {INFINITY, INFINITY, INFINITY, INFINITY}, mx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint64_t nans = 0;
    for (uint32_t z = 0; z < j->d; z++)
        for (uint32_t y = 0; y < j->h; y++)
            for (uint32_t x = 0; x < j->w; x++) {
                float *o = px + 4 * ((uint64_t)y * pw + (uint64_t)z * j->w + x);
                decode(j->f, j->map + (((uint64_t)z * j->h + y) * j->w + x) * j->f->bytes, o);
                for (int c = 0; c < 4; c++) {
                    if (isnan(o[c])) { nans++, o[c] = 0, isnan_[(uint64_t)y * pw + (uint64_t)z * j->w + x] = 1; continue; }
                    if (o[c] < mn[c]) mn[c] = o[c];
                    if (o[c] > mx[c]) mx[c] = o[c];
                }
            }
    int mode = mode_of(j->f), grey = j->f->nc == 1;
    float scale = 1, lo = mn[0], range = mx[0] - mn[0];
    if (mode == M_INT) {
        float m = 1;
        for (int c = 0; c < j->f->nc && c < 3; c++) m = fmaxf(m, fmaxf(fabsf(mx[c]), fabsf(mn[c])));
        scale = 1.f / m;
    }
    for (int pass = 0; pass < (mode == M_FLOAT ? 2 : 1); pass++) {
        for (uint64_t i = 0; i < n; i++) {
            const float *o = px + 4 * i;
            for (int c = 0; c < 3; c++) {
                float x = o[grey ? 0 : c];
                if (mode == M_SNORM) x = x * .5f + .5f;
                else if (mode == M_INT) x *= scale;
                else if (mode == M_DEPTH) x = range > 0 ? (x - lo) / range : 0;
                else if (mode == M_FLOAT && pass) x = x > 0 ? x / (1 + x) : 0;
                out[4 * i + c] = u8(x);
            }
            out[4 * i + 3] = 255;
            if (isnan_[i]) out[4 * i] = 255, out[4 * i + 1] = 0, out[4 * i + 2] = 255;
        }
        char path[600];
        snprintf(path, sizeof path, "%s/%s%s.png", dir, j->name, pass ? "_tm" : "");
        rt_write_png(path, out, pw, ph);
    }
    uint64_t all = distinct(j, 0, j->h), top = distinct(j, 0, j->h / 4 ? j->h / 4 : 1);
    plat_mutex_lock(&img_lock);
    if (images_f) {
        if (strstr(j->name, "_td_") && j->f->bytes >= 4 && j->d == 1) {   // depth images: the raw floats, w*h, for sampling
            char rp[600];
            snprintf(rp, sizeof rp, "%s/%s.f32", dir, j->name);
            float *dv = malloc((size_t)j->w * j->h * 4);
            if (dv) {
                for (uint64_t i = 0; i < (uint64_t)j->w * j->h; i++) memcpy(&dv[i], j->map + i * j->f->bytes, 4);
                plat_file_write(rp, dv, (size_t)j->w * j->h * 4);
                free(dv);
            }
        }
        {   // raw first word (as a float: a depth) at the top centre and the centre
            float t = 0, c = 0;
            if (j->w && j->h && j->f->bytes >= 4) {
                memcpy(&t, j->map + ((uint64_t)(j->h / 8) * j->w + j->w / 2) * j->f->bytes, 4);
                memcpy(&c, j->map + ((uint64_t)(j->h / 2) * j->w + j->w / 2) * j->f->bytes, 4);
            }
            fprintf(images_f, "%s raw top-centre %g centre %g\n", j->name, t, c);
        }
        fprintf(images_f, "%s %ux%ux%u min %g %g %g %g max %g %g %g %g distinct %llu top-quarter %llu nan %llu\n", j->name,
                j->w, j->h, j->d, mn[0], mn[1], mn[2], mn[3], mx[0], mx[1], mx[2], mx[3], (unsigned long long)all,
                (unsigned long long)top, (unsigned long long)nans);
        fflush(images_f);
    }
    plat_mutex_unlock(&img_lock);
    free(out);
    free(isnan_);
    free(px);
}

static void job_done(void *arg) {
    Job *j = arg;
    VmaAllocator vma = vk_ctx()->vma;
    vmaInvalidateAllocation(vma, j->alloc, 0, VK_WHOLE_SIZE);
    write_images(j);
    // A BC texture's raw blocks also go to <name>.bin, packed rows of blocks, so a script can decode them: the PNG
    // shows the block words through the integer mapping, which loses their bits.
    size_t nl = strlen(j->name);
    if (nl > 7 && !strcmp(j->name + nl - 7, "_blocks")) {
        char path[600];
        snprintf(path, sizeof path, "%s/%s.bin", dir, j->name);
        FILE *f = fopen(path, "wb");
        if (f) fwrite(j->map, 1, j->bytes, f), fclose(f);
    }
    vmaDestroyBuffer(vma, j->buf, j->alloc);
    plat_add64(&inflight, -(int64_t)j->bytes);
    free(j);
}

// ---- recording

// Records the copy of one subresource into a host buffer and queues its PNGs. Returns NULL, or why it was skipped.
// The buffer is w x h x d texels of vf; the image region is `ext` image texels at depth plane z (differs for the
// blocks of a BC image read as an uncompressed format of the block size, and for a 3D slice).
static int32_t roi_x, roi_y;   // image offset of the next copy_region (NX2NAVI_DUMP_SPLIT_ROI); 0 otherwise
// NX2NAVI_DUMP_SPLIT_THUMB=1: a split pass's target is scaled on the GPU to THUMB_W x THUMB_H (nearest) and only that
// is read back (about 0.5 MB per draw instead of 8), so every draw of every pass can be seen in one frame.
#define THUMB_W 320
#define THUMB_H 180
static int thumb_next;   // set around target_line for split passes
static const char *copy_region(VkExec *ex, VkImage image, VkFormat vf, VkImageAspectFlags aspect, uint32_t level,
                               uint32_t layer, uint32_t z, VkExtent3D ext, uint32_t w, uint32_t h, uint32_t d,
                               const char *name) {
    const Fmt *f = fmt_of(vf);
    if (!f) return "format not decoded";
    if (thumb_next && aspect == VK_IMAGE_ASPECT_COLOR_BIT && d == 1) {   // NX2NAVI_DUMP_SPLIT_THUMB: a small copy only
        if (mode_of(f) == M_INT) return "per-draw thumbnail: integer format not scaled";
        static VkImage timg;
        static VmaAllocation talloc;
        static int tbad;
        if (!timg && !tbad) {
            VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                    .format = VK_FORMAT_R16G16B16A16_SFLOAT, .extent = {THUMB_W, THUMB_H, 1}, .mipLevels = 1,
                                    .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                    .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
            if (vmaCreateImage(vk_ctx()->vma, &ci, &ai, &timg, &talloc, NULL) != VK_SUCCESS) {
                tbad = 1, timg = VK_NULL_HANDLE;
                rt_log("vk_dump: per-draw thumbnail image could not be created; split passes saved at full size");
            } else {
                VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                            .srcStageMask = VK_PIPELINE_STAGE_2_NONE, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
                                            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .image = timg, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
                VkDependencyInfo id = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
                vkCmdPipelineBarrier2(ex->cb, &id);
            }
        }
        if (timg) {
            VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                   .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
            VkDependencyInfo md = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
            vkCmdPipelineBarrier2(ex->cb, &md);
            VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1},
                              .srcOffsets = {{0, 0, 0}, {(int32_t)ext.width, (int32_t)ext.height, 1}},
                              .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .dstOffsets = {{0, 0, 0}, {THUMB_W, THUMB_H, 1}}};
            vkCmdBlitImage(ex->cb, image, VK_IMAGE_LAYOUT_GENERAL, timg, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_NEAREST);
            ex->recorded += 2;
            image = timg, vf = VK_FORMAT_R16G16B16A16_SFLOAT, f = fmt_of(vf), level = layer = z = 0;
            ext = (VkExtent3D){THUMB_W, THUMB_H, 1}, w = THUMB_W, h = THUMB_H;
            roi_x = roi_y = 0;
        }
    }
    if (__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1 != dump_frame && (uint64_t)w * h * d > 65536)
        return "logged frame, large image";
    if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    else if (aspect != VK_IMAGE_ASPECT_COLOR_BIT) return "stencil only";
    uint64_t bytes = (uint64_t)w * h * d * f->bytes;
    uint64_t est = ((uint64_t)h * (1 + 4ull * w * d) + 128) * (mode_of(f) == M_FLOAT ? 2 : 1);
    if ((uint64_t)png_bytes + est > PNG_CAP) return "PNG cap reached";
    if ((uint64_t)inflight + bytes > INFLIGHT_CAP) return "too many copies in flight";
    Job *j = calloc(1, sizeof *j);
    if (!j) rt_fatal("vk_dump: out of memory");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info) != VK_SUCCESS || !info.pMappedData) {
        free(j);
        return "host buffer allocation failed";
    }
    j->map = info.pMappedData, j->bytes = bytes, j->f = f, j->w = w, j->h = h, j->d = d;
    snprintf(j->name, sizeof j->name, "%s", name);
    VkMemoryBarrier2 pre = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
    VkMemoryBarrier2 post = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                             .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                             .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                             .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &pre};
    vkCmdPipelineBarrier2(ex->cb, &di);
    VkBufferImageCopy r = {.imageSubresource = {aspect, level, layer, 1}, .imageOffset = {roi_x, roi_y, (int32_t)z},
                           .imageExtent = ext};   // packed rows: a BC row of ext.width texels is w blocks
    vkCmdCopyImageToBuffer(ex->cb, image, VK_IMAGE_LAYOUT_GENERAL, j->buf, 1, &r);
    di.pMemoryBarriers = &post;
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded += 3;
    plat_add64(&png_bytes, (int64_t)est);
    plat_add64(&inflight, (int64_t)bytes);
    vk_completion_add(ex->fence, job_done, j);
    return NULL;
}

static const char *copy_out(VkExec *ex, VkImage image, VkFormat vf, VkImageAspectFlags aspect, uint32_t level,
                            uint32_t layer, uint32_t w, uint32_t h, uint32_t d, const char *name) {
    return copy_region(ex, image, vf, aspect, level, layer, 0, (VkExtent3D){w, h, d}, w, h, d, name);
}

// ---- NX2NAVI_DUMP_BUF=<addr>:<size>[:<off>:<stride>:<w>:<h>]: after every pass and dispatch that is logged, the GPU
// buffer range is copied to a host buffer; when it differs from the previous copy it goes to buf_<pass>.bin, and
// bufs.txt gets the changed bytes' extent plus, with the grid arguments, one map per dword of the w x h records of
// `stride` bytes at `off` (each record a letter: its value's index in the table under the map).

static uint64_t bd_addr, bd_size, bd_off, bd_stride;
static uint32_t bd_w, bd_h;
static uint8_t *bd_prev;
static int bd_auto;
static FILE *bufs_f;

typedef struct { VkBuffer buf; VmaAllocation alloc; const uint8_t *map; int pass; } BufJob;

static void buf_grid(const uint8_t *m) {
    for (uint32_t dw = 0; dw < bd_stride / 4; dw++) {
        uint32_t vals[52];
        int nv = 0, over = 0;
        fprintf(bufs_f, "  dword %u:\n", dw);
        for (uint32_t y = 0; y < bd_h; y++) {
            char line[260];
            uint32_t x;
            for (x = 0; x < bd_w && x < 256; x++) {
                uint64_t o = bd_off + ((uint64_t)y * bd_w + x) * bd_stride + dw * 4;
                if (o + 4 > bd_size) { line[x] = ' '; continue; }
                uint32_t v;
                memcpy(&v, m + o, 4);
                int k = 0;
                while (k < nv && vals[k] != v) k++;
                if (k == nv) {
                    if (nv < 52) vals[nv++] = v;
                    else { over = 1; line[x] = '?'; continue; }
                }
                line[x] = (char)(k < 26 ? 'a' + k : 'A' + k - 26);
            }
            line[x] = 0;
            fprintf(bufs_f, "    %2u %s\n", y, line);
        }
        for (int k = 0; k < nv; k++) fprintf(bufs_f, "%s%c=%08x", k % 8 ? " " : "    ", k < 26 ? 'a' + k : 'A' + k - 26, vals[k]);
        fprintf(bufs_f, "%s\n", over ? " (more: ?)" : "");
    }
}

static void buf_done(void *arg) {
    BufJob *j = arg;
    vmaInvalidateAllocation(vk_ctx()->vma, j->alloc, 0, VK_WHOLE_SIZE);
    plat_mutex_lock(&img_lock);
    if (bufs_f) {
        uint64_t lo = bd_size, hi = 0;
        for (uint64_t i = 0; i < bd_size; i++)
            if (!bd_prev || j->map[i] != bd_prev[i]) { if (i < lo) lo = i; hi = i + 1; }
        if (hi) {
            char path[600];
            snprintf(path, sizeof path, "%s/buf_%04d.bin", dir, j->pass);
            FILE *f = fopen(path, "wb");
            if (f) fwrite(j->map, 1, bd_size, f), fclose(f);
            fprintf(bufs_f, "after pass %04d: bytes %#llx..%#llx changed\n", j->pass, (unsigned long long)lo,
                    (unsigned long long)hi);
            if (bd_stride) buf_grid(j->map);
            fflush(bufs_f);
            if (!bd_prev) bd_prev = malloc(bd_size);
            if (bd_prev) memcpy(bd_prev, j->map, bd_size);
        }
    }
    plat_mutex_unlock(&img_lock);
    vmaDestroyBuffer(vk_ctx()->vma, j->buf, j->alloc);
    free(j);
}

static void buf_snap(VkExec *ex, int pass) {
    if (!bd_size) return;
    VkMemSpan s;
    if (!vk_mem_resolve(bd_addr, bd_size, &s) || s.avail < bd_size) return;
    VK_MEM_PICK(s);   // the GPU's view of the buffer
    BufJob *j = calloc(1, sizeof *j);
    if (!j) rt_fatal("vk_dump: out of memory");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bd_size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info) != VK_SUCCESS || !info.pMappedData) {
        free(j);
        return;
    }
    j->map = info.pMappedData, j->pass = pass;
    VkMemoryBarrier2 mb[2] = {
        {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
         .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT},
        {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
         .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb[0]};
    vkCmdPipelineBarrier2(ex->cb, &di);
    VkBufferCopy c = {s.offset, 0, bd_size};
    vkCmdCopyBuffer(ex->cb, s.buffer, j->buf, 1, &c);
    di.pMemoryBarriers = &mb[1];
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded += 3;
    vk_completion_add(ex->fence, buf_done, j);
}

// (orbitals, diagnostic) the backdrop quad's GPU inputs: at its draw the stream-1 word address and the c3+0 list
// address are noted; at the end of that pass both ranges are copied in GPU order to qsnap_<frame>_{w,list}.bin
static uint64_t qs_w, qs_list, qs_frame, qs_b1, qs_b2, qs_c5;
typedef struct { VkBuffer buf; VmaAllocation alloc; const uint8_t *map; uint64_t frame, size, addr; char tag[24]; int log; } QsJob;
static void qs_done(void *arg) {
    QsJob *j = arg;
    vmaInvalidateAllocation(vk_ctx()->vma, j->alloc, 0, VK_WHOLE_SIZE);
    if (j->log) {   // NX2NAVI_WATCH: the GPU-ordered bytes go to the log
        char hex[3 * 64 + 1];
        int o = 0;
        for (uint64_t i = 0; i < j->size && i < 64; i++) o += snprintf(hex + o, sizeof hex - o, "%02x%s", j->map[i], (i & 1) ? " " : "");
        rt_log("watch: GPU %s %#llx (frame %llu): %s", j->tag, (unsigned long long)j->addr, (unsigned long long)j->frame, hex);
        vmaDestroyBuffer(vk_ctx()->vma, j->buf, j->alloc);
        free(j);
        return;
    }
    char path[600];
    snprintf(path, sizeof path, "%s/qsnap_%llu_%s.bin", dir, (unsigned long long)j->frame, j->tag);
    FILE *f = fopen(path, "wb");
    if (f) fwrite(j->map, 1, j->size, f), fclose(f);
    vmaDestroyBuffer(vk_ctx()->vma, j->buf, j->alloc);
    free(j);
}
static uint64_t frame_now(void);
static void qs_copy_ex(VkExec *ex, uint64_t addr, uint64_t size, const char *tag, int log) {
    VkMemSpan s;
    if (!addr || !vk_mem_resolve(addr, size, &s) || s.avail < size) return;
    VK_MEM_PICK(s);
    QsJob *j = calloc(1, sizeof *j);
    if (!j) return;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info) != VK_SUCCESS || !info.pMappedData) { free(j); return; }
    j->map = info.pMappedData, j->frame = log ? frame_now() : qs_frame, j->size = size, j->addr = addr, j->log = log;
    snprintf(j->tag, sizeof j->tag, "%s", tag);
    VkMemoryBarrier2 mb[2] = {
        {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
         .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT},
        {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
         .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb[0]};
    vkCmdPipelineBarrier2(ex->cb, &di);
    VkBufferCopy c = {s.offset, 0, size};
    vkCmdCopyBuffer(ex->cb, s.buffer, j->buf, 1, &c);
    di.pMemoryBarriers = &mb[1];
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded += 3;
    vk_completion_add(ex->fence, qs_done, j);
}

static void qs_copy(VkExec *ex, uint64_t addr, uint64_t size, const char *tag) { qs_copy_ex(ex, addr, size, tag, 0); }

// ---- (orbitals, diagnostic) NX2NAVI_WATCH=<addr>:<len>[,...]: provenance of a few guest bytes (bad index buffers).
// Logs ("watch:" lines) the bytes at pool registration, every change seen before a submission (CPU writes, or GPU
// writes copied back to guest memory), every buffer copy/fill overlapping them (source, destination, size, the
// source bytes in CPU view at record time and GPU-ordered source-before / destination-before / destination-after
// readbacks), and indexed draws reading them whenever their bytes differ from the last such draw's.
static struct { uint64_t a, n; uint8_t last[64]; int have; uint8_t drawn[64]; int drawn_have; } W[8];
static int nw = -1;
static uint64_t w_submits;
static void w_init(void) {
    if (nw >= 0) return;
    nw = 0;
    const char *e = cfg_get("NX2NAVI_WATCH");
    while (e && *e && nw < 8) {
        char *end;
        uint64_t a = strtoull(e, &end, 16);
        uint64_t n = *end == ':' ? strtoull(end + 1, &end, 0) : 24;
        if (n > 64) n = 64;
        if (a) W[nw].a = a, W[nw].n = n, nw++;
        e = *end == ',' ? end + 1 : NULL;
    }
    if (nw) rt_log("watch: %d range(s)", nw);
}
int vk_watch_active(void) { w_init(); return nw > 0; }
static void w_hex(char *out, size_t cap, const uint8_t *b, uint64_t n) {
    size_t o = 0;
    out[0] = 0;
    for (uint64_t i = 0; i < n && o + 4 < cap; i++) o += (size_t)snprintf(out + o, cap - o, "%02x%s", b[i], (i & 1) ? " " : "");
}
static int w_read(uint64_t a, uint64_t n, uint8_t *out) {
    VkMemSpan s;
    if (!vk_mem_resolve(a, n, &s) || s.avail < n) return 0;
    memcpy(out, (const void *)(uintptr_t)a, n);
    return 1;
}
void vk_watch_pool(uint64_t mem, uint64_t size, const char *what) {
    if (!vk_watch_active()) return;
    for (int i = 0; i < nw; i++) {
        if (W[i].a + W[i].n <= mem || W[i].a >= mem + size) continue;
        char hex[200];
        w_hex(hex, sizeof hex, (const uint8_t *)(uintptr_t)W[i].a, W[i].n);
        rt_log("watch: %#llx %s pool %#llx+%#llx (frame %llu): %s", (unsigned long long)W[i].a, what, (unsigned long long)mem,
               (unsigned long long)size, (unsigned long long)frame_now(), hex);
        memcpy(W[i].last, (const void *)(uintptr_t)W[i].a, W[i].n), W[i].have = 1;
    }
}
void vk_watch_poll(const char *where) {
    if (!vk_watch_active()) return;
    if (!strcmp(where, "submission")) w_submits++;
    for (int i = 0; i < nw; i++) {
        uint8_t b[64];
        if (!w_read(W[i].a, W[i].n, b)) continue;
        if (W[i].have && !memcmp(b, W[i].last, W[i].n)) continue;
        char hex[200];
        w_hex(hex, sizeof hex, b, W[i].n);
        rt_log("watch: %#llx %s before %s %llu (frame %llu): %s", (unsigned long long)W[i].a,
               W[i].have ? "changed" : "first seen", where, (unsigned long long)w_submits, (unsigned long long)frame_now(), hex);
        memcpy(W[i].last, b, W[i].n), W[i].have = 1;
    }
}
// A copy (src != 0) or fill (src == 0, value) writing dst..dst+size; phase 0 before recording it, 1 after.
void vk_watch_copy(VkExec *ex, uint64_t src, uint64_t dst, uint64_t size, uint32_t value, int phase) {
    if (!vk_watch_active()) return;
    for (int i = 0; i < nw; i++) {
        if (W[i].a + W[i].n <= dst || W[i].a >= dst + size) continue;
        uint64_t off = W[i].a - dst;   // where the watched bytes come from in the source
        if (phase == 0) {
            char hex[200] = "(not readable)";
            uint8_t b[64];
            if (src && w_read(src + off, W[i].n, b)) w_hex(hex, sizeof hex, b, W[i].n);
            if (src)
                rt_log("watch: %#llx copy %#llx -> %#llx size %#llx in submission %llu (frame %llu); its source %#llx, CPU view at record: %s",
                       (unsigned long long)W[i].a, (unsigned long long)src, (unsigned long long)dst, (unsigned long long)size,
                       (unsigned long long)w_submits, (unsigned long long)frame_now(), (unsigned long long)(src + off), hex);
            else
                rt_log("watch: %#llx fill %#llx size %#llx value %#x in submission %llu (frame %llu)", (unsigned long long)W[i].a,
                       (unsigned long long)dst, (unsigned long long)size, value, (unsigned long long)w_submits,
                       (unsigned long long)frame_now());
            vk_exec_pass_end(ex);
            if (src) qs_copy_ex(ex, src + off, W[i].n, "source before copy", 1);
            qs_copy_ex(ex, W[i].a, W[i].n, "destination before", 1);
        } else qs_copy_ex(ex, W[i].a, W[i].n, "destination after", 1);
    }
}
static uint64_t shader_hash(const GfxShader *s);
void vk_watch_bind(uint64_t addr, uint64_t n, uint64_t size, unsigned long long prog) {
    if (!vk_watch_active()) return;
    static int logged;
    for (int i = 0; i < nw; i++) {
        if (W[i].a + W[i].n <= addr || W[i].a >= addr + n || logged > 400) continue;
        logged++;
        extern const void *vk_cur_cs;
        rt_log("watch: %#llx inside a shader-written binding %#llx+%#llx (game size %#llx%s), program %llu (cs %016llx), submission %llu (frame %llu)",
               (unsigned long long)W[i].a, (unsigned long long)addr, (unsigned long long)n, (unsigned long long)size,
               size ? "" : ": none, bound to the end of the pool", prog,
               (unsigned long long)(vk_cur_cs ? shader_hash((const GfxShader *)vk_cur_cs) : 0), (unsigned long long)w_submits,
               (unsigned long long)frame_now());
    }
}
void vk_watch_draw(uint64_t ibuf, uint64_t bytes) {
    if (!vk_watch_active()) return;
    for (int i = 0; i < nw; i++) {
        if (W[i].a + W[i].n <= ibuf || W[i].a >= ibuf + bytes) continue;
        uint8_t b[64];
        if (!w_read(W[i].a, W[i].n, b) || (W[i].drawn_have && !memcmp(b, W[i].drawn, W[i].n))) continue;
        char hex[200];
        w_hex(hex, sizeof hex, b, W[i].n);
        rt_log("watch: %#llx drawn as indices %#llx+%llu in submission %llu (frame %llu), CPU view: %s", (unsigned long long)W[i].a,
               (unsigned long long)ibuf, (unsigned long long)bytes, (unsigned long long)w_submits, (unsigned long long)frame_now(), hex);
        memcpy(W[i].drawn, b, W[i].n), W[i].drawn_have = 1;
    }
}

// (orbitals, diagnostic) the build-up quads (vs d25b4eb1): vertex stream 0, the c3 block, the storage buffers it names
// (c3+0x30..0x60, first 512 KiB) and the whole View block (c5), copied in GPU order at the end of their pass
static uint64_t qd_v, qd_c3, qd_s[4], qd_sn[4], qd_c5, qd_c5n, qd_t[2], qd_tn[2], qd_c6, qd_c7, qd_list_last;
static char qd_pfx[12];   // "d<draw>_" (the draw's number in the frame), so every armed draw keeps its own files
static void qd_flush(VkExec *ex) {
    if (!qd_c3) return;
    static const char *const st[4] = {"s30", "s40", "s50", "s60"};
    char t[24];
#define QD_TAG(n) (snprintf(t, sizeof t, "%s%s", qd_pfx, n), t)
    qs_copy(ex, qd_v, 256, QD_TAG("qv"));
    qs_copy(ex, qd_c3, 0x70, QD_TAG("qc3"));
    for (int i = 0; i < 4; i++) qs_copy(ex, qd_s[i], qd_sn[i], QD_TAG(st[i]));
    qs_copy(ex, qd_c5, qd_c5n, QD_TAG("qc5"));
    qs_copy(ex, qd_t[0], qd_tn[0], QD_TAG("t0"));
    qs_copy(ex, qd_t[1], qd_tn[1], QD_TAG("t2"));
    qs_copy(ex, qd_c6, 0x160, QD_TAG("qc6"));
    qs_copy(ex, qd_c7, 0x40, QD_TAG("qc7"));
#undef QD_TAG
    qd_v = qd_c3 = qd_c5 = qd_c5n = 0;
    memset(qd_t, 0, sizeof qd_t);
    memset(qd_s, 0, sizeof qd_s);
}
static void qs_flush(VkExec *ex) {
    qd_flush(ex);
    if (!qs_w && !qs_list) return;
    qs_copy(ex, qs_w, 64, "w");
    qs_copy(ex, qs_list, 4096, "list");
    qs_copy(ex, qs_b1, 65536, "b10");
    qs_copy(ex, qs_b2, 65536, "b20");
    qs_copy(ex, qs_c5, 1024, "c5");
    qs_w = qs_list = qs_b1 = qs_b2 = qs_c5 = 0;
}

static int active(void) {
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    return !finished && (f == dump_frame || (log_from && f >= log_from && f < dump_frame));
}

// (orbitals, diagnostic) Buffer-texture bindings as bound: texels.txt, one line per binding of each draw's descriptor
// write in the dumped frame (pass seq and draw number tie it to passes.txt), and tb_<addr>_<size>.bin with the guest bytes
// the view covers (once per address and size, at most 4 MiB each, 512 files), so any fetched element can be checked offline.
static int nents, ents_over, pass_draws;   // (tentative: defined below)
int vk_dump_recording(void) { return vk_dump_armed && active(); }
static FILE *texels_f;
void vk_dump_texel(int st, int binding, int slot, uint64_t handle, int img, int id, uint64_t addr, uint64_t size, int format,
                   int vkfmt, uint32_t bpe, uint64_t buf, uint64_t off, uint64_t range, int scratch, int zero, const char *why,
                   uint64_t prog) {
    static uint64_t saved[512][2];
    static int nsaved;
    plat_mutex_lock(&rec_lock);
    if (!active()) { plat_mutex_unlock(&rec_lock); return; }
    if (!texels_f) {
        char p[480];
        snprintf(p, sizeof p, "%s/texels.txt", dir);
        texels_f = fopen(p, "w");
    }
    if (texels_f) {
        fprintf(texels_f, "pass %04d draw %d prog %llu st %d binding %d slot %d %s handle %#llx id %d: guest addr %#llx size %#llx "
                "nvn fmt %#x bpe %u -> vk fmt %d view buf %#llx off %#llx range %#llx (%llu elements)%s%s; %s\n",
                seq, pass_draws + 1, (unsigned long long)prog, st, binding, slot, img ? "img" : "tex", (unsigned long long)handle, id,
                (unsigned long long)addr, (unsigned long long)size, format, bpe, vkfmt, (unsigned long long)buf,
                (unsigned long long)off, (unsigned long long)range, bpe ? (unsigned long long)(range / bpe) : 0ull,
                scratch ? " scratch copy" : "", zero ? " ZERO VIEW" : "", why ? why : "");
    }
    extern void vk_dump_vsb_texel(uint64_t addr, uint64_t size, int binding);
    vk_dump_vsb_texel(addr, size, binding);
    int have = 0;
    for (int i = 0; i < nsaved; i++) if (saved[i][0] == addr && saved[i][1] == size) have = 1;
    VkMemSpan ms;
    if (!have && addr && size && nsaved < 512 && vk_mem_resolve(addr, 1, &ms)) {
        uint64_t n = size < ms.avail ? size : ms.avail;
        if (n > (4u << 20)) n = 4u << 20;
        char p[480];
        snprintf(p, sizeof p, "%s/tb_%llx_%llx.bin", dir, (unsigned long long)addr, (unsigned long long)size);
        FILE *f = fopen(p, "wb");
        if (f) fwrite((const void *)(uintptr_t)addr, 1, n, f), fclose(f);
        saved[nsaved][0] = addr, saved[nsaved][1] = size, nsaved++;
    }
    plat_mutex_unlock(&rec_lock);
}
static uint64_t frame_now(void) { return __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1; }

// (orbitals) NX2NAVI_DUMP_SPLIT=1: in the dumped frame every draw ends its pass, so each pass line and target image
// shows the picture after one draw (the pass resumes with its contents loaded)
// NX2NAVI_DUMP_SPLIT_ADDR=<hex>: split only passes whose colour target 0 is that guest address (one pass's draws),
// or, for a value up to 0xffff, has that nvn format with more targets bound (addresses move between runs).
int vk_dump_split(VkExec *ex) {
    static int on = -1;
    static uint64_t addr;
    if (on < 0) {
        const char *a = cfg_get("NX2NAVI_DUMP_SPLIT_ADDR");
        addr = a && *a ? strtoull(a, NULL, 16) : 0;
        on = rt_knob("DUMP_SPLIT", "0") != 0 || addr;
        if (addr) rt_log("vk_dump: splitting passes into colour target %#llx per draw", (unsigned long long)addr);
    }
    if (!on || !active()) return 0;
    if (!addr) return 1;
    if (addr <= 0xffff) {   // a format: NX2NAVI_DUMP_SPLIT_ADDR=29 splits passes whose colour target 0 has nvn format 0x29
        if (!ex->color[0].tex || ex->color[0].tex->d.format != addr) return 0;
        static int single = -1;   // NX2NAVI_DUMP_SPLIT_SINGLE=1: single-target passes of that format too (translucency)
        if (single < 0) single = rt_knob("DUMP_SPLIT_SINGLE", "0") != 0;
        if (single) return 1;
        for (int i = 1; i < 8; i++) if (ex->color[i].tex) return 1;   // a multi-target (G-buffer) pass
        return 0;
    }
    return ex->color[0].tex && (uint64_t)(uintptr_t)ex->color[0].tex->mem == addr;
}

typedef struct { char *s; size_t n, cap; } Str;

__attribute__((format(printf, 2, 3))) static void sp(Str *b, const char *fmt, ...) {
    va_list a;
    va_start(a, fmt);
    int k = vsnprintf(NULL, 0, fmt, a);
    va_end(a);
    if (k < 0) return;
    if (b->n + (size_t)k + 1 > b->cap) {
        b->cap = (b->n + (size_t)k + 1) * 2;
        b->s = realloc(b->s, b->cap);
        if (!b->s) rt_fatal("vk_dump: out of memory");
    }
    va_start(a, fmt);
    vsnprintf(b->s + b->n, (size_t)k + 1, fmt, a);
    va_end(a);
    b->n += (size_t)k;
}

// SPIR-V hash of a shader; the first time, its SPIR-V goes to sh_<hash>.spv.
static uint64_t shader_hash(const GfxShader *s) {
    static struct { const GfxShader *s; uint64_t h; } tab[1024];
    static int n;
    if (!s) return 0;
    for (int i = 0; i < n; i++)
        if (tab[i].s == s) return tab[i].h;
    uint64_t words = 0, h = 0xcbf29ce484222325ull;
    const uint32_t *w = gfx_shader_spirv(s, &words);
    const uint8_t *b = (const uint8_t *)w;
    for (uint64_t i = 0; i < words * 4; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    if (dir[0]) {   // no dump directory (NX2NAVI_WATCH alone): the hash only
        char path[600];
        snprintf(path, sizeof path, "%s/sh_%016llx.spv", dir, (unsigned long long)h);
        FILE *f = fopen(path, "wb");
        if (f) fwrite(w, 4, (size_t)words, f), fclose(f);
    }
    if (n < 1024) tab[n].s = s, tab[n].h = h, n++;
    return h;
}
const void *vk_cur_vs;   // (diagnostic) the vertex shader of the draw being recorded
const void *vk_cur_cs;   // (diagnostic) the compute shader of the dispatch being recorded
uint64_t vk_dump_shader_hash(const void *s) { return shader_hash((const GfxShader *)s); }

// Per pass: distinct pipeline + texture sets, draws, 3D textures to copy at its end.
typedef struct { VkPipeline p; uint64_t th; int draws; char *text; } Ent;
static Ent ents[MAX_ENTS];
static int nents, ents_over, pass_draws;
typedef struct { VkImage image; VkFormat format; uint32_t w, h, d; int id, is2d; uint32_t levels; } Lut;
static Lut luts[MAX_LUTS];          // queued for the next pass end or dispatch
static int nluts;
static VkImage luts_done[MAX_LUTS];
static int nluts_done;

static void queue_lut(const VkTexImage *t, int id) {
    if (frame_now() != dump_frame) return;
    for (int i = 0; i < nluts_done; i++)
        if (luts_done[i] == t->image) return;
    if (nluts_done == MAX_LUTS) return;
    luts_done[nluts_done++] = t->image;
    luts[nluts++] = (Lut){t->image, t->format, t->width, t->height, t->type == VK_IMAGE_TYPE_3D ? t->depth : 1, id,
                          t->type != VK_IMAGE_TYPE_3D, t->levels};
}

// The uncompressed format of a BC format's block size, UNDEFINED for other formats: a BC texture is dumped as its raw
// blocks (one texel per block), which shows at least whether it was ever written.
static VkFormat block_format(VkFormat f) {
    if (f == VK_FORMAT_BC4_UNORM_BLOCK || f == VK_FORMAT_BC4_SNORM_BLOCK ||
        (f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_BC1_RGBA_SRGB_BLOCK))
        return VK_FORMAT_R32G32_UINT;
    if (f >= VK_FORMAT_BC2_UNORM_BLOCK && f <= VK_FORMAT_BC7_SRGB_BLOCK) return VK_FORMAT_R32G32B32A32_UINT;
    return VK_FORMAT_UNDEFINED;
}

static void flush_luts(VkExec *ex, int n) {
    char fn[16];
    for (int i = 0; i < nluts; i++) {
        const Lut *l = &luts[i];
        char name[160];
        VkFormat bf = block_format(l->format);
        snprintf(name, sizeof name, "pass_%04d_%s%d_%ux%ux%u_%d_%s%s", n, l->is2d ? "tex" : "lut", l->id, l->w, l->h, l->d,
                 (int)l->format, fmt_name(l->format, fn), bf ? "_blocks" : "");
        // a sampled depth texture copies its depth aspect (the colour aspect of a depth image read as zeros)
        VkImageAspectFlags asp = l->format >= VK_FORMAT_D16_UNORM && l->format <= VK_FORMAT_D32_SFLOAT_S8_UINT
                                     ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        const char *why = bf ? copy_region(ex, l->image, bf, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0,
                                           (VkExtent3D){l->w, l->h, l->d}, (l->w + 3) / 4, (l->h + 3) / 4, l->d, name)
                             : copy_out(ex, l->image, l->format, asp, 0, 0, l->w, l->h, l->d, name);
        fprintf(passes_f, "  %s tex %d -> %s.png%s%s\n", l->is2d ? "2D" : "3D", l->id, name, why ? " skipped: " : "",
                why ? why : "");
        // (orbitals) a wanted 2D BC texture (NX2NAVI_DUMP_TEX): every further mip level's raw blocks too, <name>_lv<n>
        for (uint32_t lv = 1; bf && l->is2d && lv < l->levels && lv < 16; lv++) {
            uint32_t w = l->w >> lv ? l->w >> lv : 1, h = l->h >> lv ? l->h >> lv : 1;
            char nl[180];   // ..._lv<n>_blocks: job_done writes the raw .bin only for names ending in _blocks
            size_t nn = strlen(name);
            snprintf(nl, sizeof nl, "%.*s_lv%u_blocks", (int)(nn > 7 ? nn - 7 : nn), name, lv);
            const char *wl = copy_region(ex, l->image, bf, VK_IMAGE_ASPECT_COLOR_BIT, lv, 0, 0, (VkExtent3D){w, h, 1},
                                         (w + 3) / 4, (h + 3) / 4, 1, nl);
            fprintf(passes_f, "  2D tex %d level %u -> %s.png%s%s\n", l->id, lv, nl, wl ? " skipped: " : "", wl ? wl : "");
        }
    }
    nluts = 0;
}

static int nvn_stage(int gfx_stage) { return gfx_stage == GFX_STAGE_VS ? 0 : gfx_stage == GFX_STAGE_FS ? 1 : 5; }

static const char *decl_type(uint16_t t, char tmp[24]) {
    static const char *base[] = {"?", "1D", "buffer", "2D", "3D", "cube"};
    snprintf(tmp, 24, "%s%s%s%s", base[(t & 0xff) < 6 ? t & 0xff : 0], t & 0x100 ? "Array" : "", t & 0x200 ? "MS" : "",
             t & 0x400 ? "Shadow" : "");
    return tmp;
}

static const char *wrap_name(int w) {
    static const char *n[] = {"clamp", "repeat", "mirror_clamp", "mirror_clamp_edge", "mirror_clamp_border",
                              "clamp_border", "mirrored_repeat", "clamp_edge"};
    return w >= 0 && w < 8 ? n[w] : "?";
}

static uint64_t handle_of(const VkExec *ex, int st, int slot, int img) {
    if (img) return slot >= 0 && slot < 32 ? ex->img[st][slot] : 0;
    return slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
}

// The textures, images and samplers of info's bindings; 3D textures are queued for a copy.
static void describe_bindings(Str *b, const VkExec *ex, const VkPipeInfo *info) {
    static const char *stn[6] = {"vs", "fs", "?", "?", "?", "cs"};
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *bd = &info->bindings[i];
        const GfxShaderRes *r = bd->res;
        if (r && bd->kind == GFX_RES_SSBO) {   // where the address lives (c0: the stage's NVN slot table) and the stage's slots
            int st = nvn_stage(bd->stage);
            sp(b, "    %s ssbo at c%d+%#x%s, bound:", stn[st], r->cbuf_slot, r->handle * 4, r->flags & 1 ? " written" : "");
            if (r->ptr) sp(b, " (address at +%#x of that buffer)", r->ptr - 1u);
            if (st == 5 && r->cbuf_slot == 0 && r->handle * 4 >= 0x10 && r->handle * 4 + 12 <= 0x10 + (int)sizeof ex->cs_push) {
                uint64_t pa; uint32_t ps;   // an NVN2 compute push-constant pointer {address, size}
                memcpy(&pa, ex->cs_push + r->handle * 4 - 0x10, 8), memcpy(&ps, ex->cs_push + r->handle * 4 - 0x10 + 8, 4);
                sp(b, " (push pointer %#llx+%#x)", (unsigned long long)pa, ps);
            }
            if (st != 5) {   // (orbitals) a draw stage's: what it resolves to, and its first words
                uint64_t gs = 0, ga = vk_exec_gfx_storage(ex, st, r, &gs);
                sp(b, " -> %#llx size %#llx", (unsigned long long)ga, (unsigned long long)gs);
                VkMemSpan ms;
                if (ga && st == 0) vsb_note_ssbo(ga, r->handle);
                if (ga && st == 5) vsb_note_cs_ssbo(ga, r->handle);
                if (ga && vk_mem_resolve(ga, 1, &ms)) {
                    sp(b, " (span %#llx)", (unsigned long long)ms.avail);
                    const uint32_t *gw = (const uint32_t *)(uintptr_t)ga;
                    sp(b, " words %08x %08x %08x %08x", gw[0], gw[1], gw[2], gw[3]);
                }
            }
            if (st == 5) {   // the address and how far its backing span reaches (what a size-0 binding covers)
                uint64_t ca = vk_exec_cs_storage_addr(ex, r);
                VkMemSpan ms;
                sp(b, " -> %#llx", (unsigned long long)ca);
                if (ca && vk_mem_resolve(ca, 1, &ms)) sp(b, " (span %#llx)", (unsigned long long)ms.avail);
            }
            for (int k = 0; k < 16; k++)
                if (ex->ssbo[st][k].addr) {
                    VkDescriptorBufferInfo cb;   // "(copy)": a binding that only reads it uses the submit-time copy (vk_usnap.c)
                    int copy = vk_usnap_find(ex->ssbo[st][k].addr, ex->ssbo[st][k].size, 1, &cb);
                    sp(b, " %d=%#llx+%#llx%s", k, (unsigned long long)ex->ssbo[st][k].addr, (unsigned long long)ex->ssbo[st][k].size,
                       copy ? "(copy)" : "");
                }
            sp(b, "\n");
            // NX2NAVI_DUMP_UBO: also the small storage buffers' guest bytes (GPU results as of the last write-back); per-view
            // constants (exposure and such) live in such buffers.
            for (int k = 0; dump_ubo && k < 16; k++) {
                const p_bind_buf *u = &ex->ssbo[st][k];
                VkMemSpan ms;
                if (!u->addr || !u->size || u->size > 0x100 || !vk_mem_resolve(u->addr, 1, &ms)) continue;
                const float *f = (const float *)(uintptr_t)u->addr;
                sp(b, "      %s ssbo %d %#llx:", stn[st], k, (unsigned long long)u->addr);
                for (uint64_t q = 0; q * 16 < u->size && q < (uint64_t)dump_ubo; q++)
                    sp(b, " [%llu] %g %g %g %g", (unsigned long long)q, f[4 * q], f[4 * q + 1], f[4 * q + 2], f[4 * q + 3]);
                sp(b, "\n");
            }
            continue;
        }
        if (r && bd->kind == GFX_RES_UBO && dump_ubo && r->slot >= 3 && r->slot - 3 < 18) {
            int st = nvn_stage(bd->stage);
            const p_bind_buf *u = &ex->ubo[st][r->slot - 3];
            VkMemSpan ms;
            sp(b, "    %s ubo c%d addr %#llx size %#llx", stn[st], r->slot, (unsigned long long)u->addr, (unsigned long long)u->size);
            if (u->addr && vk_mem_resolve(u->addr, 1, &ms)) {
                uint64_t n = u->size && u->size < ms.avail ? u->size : ms.avail;
                const float *f = (const float *)(uintptr_t)u->addr;
                for (uint64_t k = 0; k < (uint64_t)dump_ubo && (k + 1) * 16 <= n; k++)
                    sp(b, "%s[%llu] %g %g %g %g", k % 4 ? " " : "\n      ", (unsigned long long)k, f[4 * k], f[4 * k + 1],
                       f[4 * k + 2], f[4 * k + 3]);
            } else {
                sp(b, " (not mapped)");
            }
            sp(b, "\n");
            continue;
        }
        if (!r || (bd->kind != GFX_RES_TEX && bd->kind != GFX_RES_IMG)) continue;
        int st = nvn_stage(bd->stage), img = bd->kind == GFX_RES_IMG;
        char dt[24], fn[16];
        if (bd->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || bd->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
            sp(b, "    %s %s slot %d binding %u: buffer texture (the view each draw got: texels.txt)\n", stn[st], img ? "img" : "tex", r->slot, (unsigned)bd->binding);
            continue;
        }
        for (uint32_t k = 0; k < (bd->count ? bd->count : 1u); k++) {
            int slot = r->slot + (int)k;
            uint64_t h = handle_of(ex, st, slot, img);
            if (!h && slot < 0) h = vk_exec_ubo_tex_handle(ex, st, r);   // (orbitals) a handle in a uniform buffer
            sp(b, "    %s %s slot %d decl %s%s: ", stn[st], img ? "img" : "tex", slot, decl_type(r->type, dt),
               img && (r->flags & 4) ? " store" : "");
            if (!h) { sp(b, "no handle (dummy)\n"); continue; }
            view_t v;
            int hv = 0;
            const VkTexImage *t = vk_texture_by_id(NVN_HANDLE_TEX(h), &v, &hv);
            if (!t) {
                extern int vk_texture_slot_describe(int, char *, size_t);
                char why[300];
                vk_texture_slot_describe(NVN_HANDLE_TEX(h), why, sizeof why);
                sp(b, "tex %d unresolved (dummy): %s; handle %#llx\n", NVN_HANDLE_TEX(h), why, (unsigned long long)h);
                continue;
            }
            static const char *vt[] = {"1D", "2D", "3D", "cube", "1DArray", "2DArray", "cubeArray"};
            sp(b, "tex %d %s(%d) %ux%ux%u levels %u layers %u image %s vkimage %p", NVN_HANDLE_TEX(h), fmt_name(t->format, fn),
               (int)t->format, t->width, t->height, t->depth, t->levels, t->layers,
               t->view_type <= 6 ? vt[t->view_type] : "?", (void *)t->image);
            {   // (orbitals) the descriptor's NVN format and the view's effective component mapping (vk_exec_draw.c view
                // creation: the view's or texture's selectors, then alpha_one / chan); 2..5 = R..A, 0 = zero, 1 = one
                uint8_t sw[4];
                memcpy(sw, hv && v.set_swizzle ? v.swz : t->swz, 4);
                for (int i = 0; i < 4; i++) {
                    if (t->alpha_one && sw[i] == NVN_TEXTURE_SWIZZLE_A) sw[i] = NVN_TEXTURE_SWIZZLE_ONE;
                    else if (t->chan[0] && sw[i] >= NVN_TEXTURE_SWIZZLE_R && sw[i] <= NVN_TEXTURE_SWIZZLE_A)
                        sw[i] = t->chan[sw[i] - NVN_TEXTURE_SWIZZLE_R];
                }
                sp(b, " nvn-fmt %#x%s view-swz %u%u%u%u", t->nvn_fmt, t->nvn_fmt >= 0x79 && t->nvn_fmt <= 0x94 ? " (ASTC)" : "",
                   sw[0], sw[1], sw[2], sw[3]);
            }
            if (hv && v.set_format) sp(b, " view-nvn-fmt %#x", v.format);
            if (hv && v.set_target) sp(b, " view-target %u", v.target);
            if (hv && v.set_levels) sp(b, " view-levels %u+%u", v.base_level, v.nlevels);
            if (hv && v.set_layers) sp(b, " view-layers %u+%u", v.min_layer, v.nlayers);
            samp_par s;
            if (img) {
                // storage image: no sampler
            } else if (r->separate) {
                sp(b, " | separate sampler (not replayed: default sampler)");
            } else if (vk_sampler_params(NVN_HANDLE_SAMP(h), &s)) {
                static const char *minn[] = {"N", "L", "N_mipN", "L_mipN", "N_mipL", "L_mipL"};
                int mf = s.minf;
                sp(b, " | samp %d: mag %s min %s (vk min %s mip %s) wrap %s/%s/%s aniso %g lod %g..%g bias %g%s",
                   NVN_HANDLE_SAMP(h), s.magf == 1 ? "L" : "N", mf < 6 ? minn[mf] : "?",
                   mf == 1 || mf == 3 || mf == 5 ? "L" : "N", mf <= 1 ? "none" : mf >= 4 ? "L" : "N",
                   wrap_name(s.ws), wrap_name(s.wt), wrap_name(s.wr), s.aniso, s.lod_min, s.lod_max, s.lod_bias,
                   s.cmp_mode ? " compare" : "");
            } else {
                sp(b, " | samp %d empty (default sampler)", NVN_HANDLE_SAMP(h));
            }
            uint32_t ups;
            int last, wr;
            vk_texture_debug(t, &ups, &last, &wr);
            sp(b, " | addr %#llx mirrored %d gpu_owned %d bound %d uploads %u last-check %d written-since %d",
               (unsigned long long)t->addr, t->mirrored, t->gpu_owned, t->bound, ups, last, wr);
            if (t->depth == 64 && t->width == 144 && cfg_get("NX2NAVI_DUMP_RAW3D")) {   // diagnosis: storage now
                static uint64_t saved;
                uint64_t n = 144ull * 128 * 64 * 4;
                if (saved != t->addr && mem_read_hold(t->addr, n)) {
                    char path[600];
                    snprintf(path, sizeof path, "%s/now_%llx.bin", cfg_get("NX2NAVI_DUMP_RAW3D"), (unsigned long long)t->addr);
                    FILE *f = fopen(path, "wb");
                    if (f) fwrite((const void *)(uintptr_t)t->addr, 1, n, f), fclose(f);
                    mem_read_release();
                    saved = t->addr;
                }
            }
            uint8_t ssw[4] = {0};
            vk_texture_slot_swz(NVN_HANDLE_TEX(h), ssw);
            sp(b, " | swz image %u%u%u%u slot %u%u%u%u", t->swz[0], t->swz[1], t->swz[2], t->swz[3], ssw[0], ssw[1], ssw[2],
               ssw[3]);
            if (hv && v.set_swizzle) sp(b, " view %u%u%u%u", v.swz[0], v.swz[1], v.swz[2], v.swz[3]);
            sp(b, "\n");
            int wanted = 0;
            for (int k = 0; k < nwant_tex; k++)
                wanted |= want_tex[k][1] ? want_tex[k][0] == t->width && want_tex[k][1] == t->height
                                         : want_tex[k][0] == (uint32_t)NVN_HANDLE_TEX(h);
            if (t->type == VK_IMAGE_TYPE_3D) queue_lut(t, NVN_HANDLE_TEX(h));
            else if (wanted && t->type == VK_IMAGE_TYPE_2D && t->samples == 1) {
                int first = 1;   // (orbitals) the guest storage behind it too, once per image (vk_texture_dump_storage)
                for (int i = 0; i < nluts_done; i++) first &= luts_done[i] != t->image;
                queue_lut(t, NVN_HANDLE_TEX(h));
                if (first && frame_now() == dump_frame) {
                    char pre[600];
                    snprintf(pre, sizeof pre, "%s/stor_tex%d_%ux%u_nvn%x", dir, NVN_HANDLE_TEX(h), t->width, t->height,
                             t->nvn_fmt);
                    vk_texture_dump_storage(t, pre);
                }
            }
            else if (!img && t->type == VK_IMAGE_TYPE_2D && t->samples == 1 && (uint64_t)t->width * t->height <= TEX2D_CAP &&
                     (t->format == VK_FORMAT_R8_UNORM || t->format == VK_FORMAT_R8G8_UNORM ||
                      t->format == VK_FORMAT_R8G8B8A8_UNORM || t->format == VK_FORMAT_R8G8B8A8_SRGB ||
                      t->format == VK_FORMAT_B8G8R8A8_UNORM || t->format == VK_FORMAT_B8G8R8A8_SRGB))
                queue_lut(t, NVN_HANDLE_TEX(h));
        }
    }
}

static uint64_t bindings_hash(const VkExec *ex, const VkPipeInfo *info) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *bd = &info->bindings[i];
        if (bd->res && bd->kind == GFX_RES_SSBO) {
            int st = nvn_stage(bd->stage), slot = bd->res->slot;
            if (slot >= 0 && slot < 16) h = (h ^ ex->ssbo[st][slot].addr) * 0x100000001b3ull;
            continue;
        }
        if (!bd->res || (bd->kind != GFX_RES_TEX && bd->kind != GFX_RES_IMG)) continue;
        for (uint32_t k = 0; k < (bd->count ? bd->count : 1u); k++)
            h = (h ^ handle_of(ex, nvn_stage(bd->stage), bd->res->slot + (int)k, bd->kind == GFX_RES_IMG)) * 0x100000001b3ull;
    }
    return h;
}

extern int64_t vk_dump_drawargs[4];
// (diagnostic) NX2NAVI_DRAWLOG=<file>: every draw of every frame, one line: frame, pass size, vs, fs, primitive id
// (the first stride-0 stream word), index count, instances. NX2NAVI_DRAWLOG_FROM=<frame> starts it later.
extern int64_t vk_dump_drawargs[4];
static uint32_t dl_hash(const void *p, size_t n) {   // FNV-1a, 32 bits
    const uint8_t *b = p; uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 16777619u;
    return h;
}
FILE *vk_drawlog_file;   // (diagnostic) also gets "OQ <slot>" after a draw that opens an occlusion query
void vk_drawlog(VkExec *ex, const VkPipeState *s) {
    static FILE *f; static int state; static uint64_t from;
    if (state < 0) return;
    if (!state) {
        const char *e = cfg_get("NX2NAVI_DRAWLOG");
        if (!e || !*e || !(f = fopen(e, "w"))) { state = -1; return; }
        { extern FILE *vk_drawlog_file; vk_drawlog_file = f; }
        const char *fr = cfg_get("NX2NAVI_DRAWLOG_FROM");
        from = fr ? strtoull(fr, 0, 0) : 0;
        state = 1;
    }
    uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    if (fr < from) return;
    uint32_t id = 0xffffffffu;
    for (int k = 0; k < 16; k++)
        if (ex->vbuf[k].address && !s->vstream[k].stride) {
            VkMemSpan ms;
            if (vk_mem_resolve(ex->vbuf[k].address, 4, &ms)) { id = *(const uint32_t *)(uintptr_t)ex->vbuf[k].address; break; }
        }
    // Scene membership: once a frame, at the depth-only draw of VS 07845b9ba606c7fd (its instance buffer address at
    // c3+0x10), the instance slot -> primitive table; a line per slot that changed since the last frame logged.
    if (shader_hash(s->vs) == 0x07845b9ba606c7fdull && ex->ubo[0][0].addr) {
        static uint32_t prev[256]; static int have; static uint64_t last_fr;
        VkMemSpan ms;
        uint64_t gsb = *(const uint64_t *)(uintptr_t)(ex->ubo[0][0].addr + 0x10) & ~15ull;
        if (fr != last_fr && gsb && vk_mem_resolve(gsb, 256 * 16, &ms)) {
            last_fr = fr;
            char line[4096]; int o = 0, nchg = 0;
            for (int q = 0; q < 256 && o < (int)sizeof line - 40; q++) {
                uint32_t p = *(const uint32_t *)(uintptr_t)(gsb + q * 16) & 0xfffff;
                if (!have || p != prev[q]) {
                    if (have) o += snprintf(line + o, sizeof line - o, " %d:%u->%u", q, prev[q], p), nchg++;
                    prev[q] = p;
                }
            }
            if (!have) {
                o = 0;
                for (int q = 0; q < 256 && o < (int)sizeof line - 20; q++)
                    if (prev[q] != 0xfffff) o += snprintf(line + o, sizeof line - o, " %d:%u", q, prev[q]);
                fprintf(f, "MAPINIT %llu%s\n", (unsigned long long)fr, line);
            } else if (nchg) fprintf(f, "MAP %llu %d changes:%s\n", (unsigned long long)fr, nchg, line);
            have = 1;
        }
    }
    // Backdrop inputs: at each draw of the full-screen depth quad (VS 0c0640a4), the instance parameters, the stride-0
    // stream words, crc32s of its storage buffers' first 64 KiB, the c3/c4 constants, and the raw rows the slot selects.
    if (shader_hash(s->vs) == 0x0c0640a47cb8383cull) {
        VkMemSpan ms;
        uint32_t w[4] = {0}, c3c = 0, c4c = 0, sb[4] = {0};
        for (int k = 1; k <= 2; k++)
            if (ex->vbuf[k].address && vk_mem_resolve(ex->vbuf[k].address, 8, &ms))
                memcpy(&w[(k - 1) * 2], (const void *)(uintptr_t)ex->vbuf[k].address, 8);
        const p_bind_buf *c3 = &ex->ubo[0][0], *c4 = &ex->ubo[0][1];
        if (c3->addr && c3->size && vk_mem_resolve(c3->addr, c3->size, &ms)) c3c = dl_hash((const void *)(uintptr_t)c3->addr, c3->size);
        if (c4->addr && c4->size && vk_mem_resolve(c4->addr, c4->size, &ms)) c4c = dl_hash((const void *)(uintptr_t)c4->addr, c4->size);
        uint64_t sa[4] = {0};
        for (int q = 0; q < 4 && c3->addr; q++)
            if (vk_mem_resolve(c3->addr + q * 16, 8, &ms)) {
                sa[q] = *(const uint64_t *)(uintptr_t)(c3->addr + q * 16) & ~15ull;
                if (sa[q] && vk_mem_resolve(sa[q], 65536, &ms)) sb[q] = dl_hash((const void *)(uintptr_t)sa[q], 65536);
            }
        char rows[600]; int o = 0;
        // c3+0: instance data, SoA rows of 256 slots; the slot is stream 1's word (+ instance)
        for (int inst = 0; inst < 2 && sa[0]; inst++) {
            uint32_t slot = (w[0] & 0xfffff) + (uint32_t)inst;
            for (int r = 0; r < 4 && o < 560; r++) {
                uint64_t a = sa[0] + ((uint64_t)r * 256 + slot) * 16;
                if (!vk_mem_resolve(a, 16, &ms)) break;
                const float *v = (const float *)(uintptr_t)a; const uint32_t *u = (const uint32_t *)(uintptr_t)a;
                o += snprintf(rows + o, sizeof rows - o, r ? " %.6g,%.6g,%.6g,%.6g" : " i%d[%#x %#x %#x %#x]", inst, u[0], u[1], u[2], u[3]);
                (void)v;
            }
        }
        rows[o] = 0;
        if (c4->addr && c4->size && vk_mem_resolve(c4->addr, c4->size, &ms)) {   // the whole view buffer, as floats
            fprintf(f, "C4 %llu", (unsigned long long)fr);
            const uint32_t *u = (const uint32_t *)(uintptr_t)c4->addr;
            for (uint32_t i = 0; i < c4->size / 4 && i < 4096; i++) fprintf(f, " %08x", u[i]);
            fprintf(f, "\n");
        }
        float cv[4] = {0};
        if (c4->addr && vk_mem_resolve(c4->addr + 84 * 16, 16, &ms)) memcpy(cv, (const void *)(uintptr_t)(c4->addr + 84 * 16), 16);
        fprintf(f, "BACKDROP %llu inst %lld base %lld words %#x %#x %#x %#x c3 %08x c4 %08x pvt %g,%g,%g sb %08x %08x %08x %08x%s\n",
                (unsigned long long)fr, (long long)vk_dump_drawargs[1], (long long)vk_dump_drawargs[3], w[0], w[1], w[2], w[3],
                c3c, c4c, cv[0], cv[1], cv[2], sb[0], sb[1], sb[2], sb[3], rows);
    }
    fprintf(f, "%llu %ux%u %016llx %016llx %#x %lld %lld\n", (unsigned long long)fr, ex->pass_w, ex->pass_h,
            (unsigned long long)shader_hash(s->vs), (unsigned long long)shader_hash(s->fs), id,
            (long long)vk_dump_drawargs[0], (long long)vk_dump_drawargs[1]);
}


// (orbitals, diagnostic) NX2NAVI_QD_VS=<hash hex>[,...]: vertex shaders whose draws get the GPU-ordered input snapshot
// (qd_flush) and a geometry line; default the build-up quads' d25b4eb1eca6c2c8
static int qd_draw_no;
static int qd_armed(uint64_t h) {
    static uint64_t list[8]; static int n = -1;
    if (n < 0) {
        n = 0;
        const char *e = cfg_get("NX2NAVI_QD_VS");
        if (!e || !*e) list[n++] = 0xd25b4eb1eca6c2c8ull;
        else for (const char *q = e; *q && n < 8;) {
            char *end; list[n++] = strtoull(q, &end, 16);
            q = *end == ',' ? end + 1 : end;
            if (end == q && *q != ',') break;
        }
    }
    for (int i = 0; i < n; i++) if (list[i] == h) return 1;
    return 0;
}
extern int64_t vk_dump_drawargs[4], vk_dump_geom[4];
// The draw's path (op_draw / op_draw_idx / indirect), NVN primitive, index type, the indices it fetches (read from
// guest memory; for a non-indexed draw the ones the runtime builds) and the stream-0 bytes of each distinct vertex.
static void qd_geom(const VkExec *ex) {
    if (!passes_f) return;
    static const char *const path[4] = {"?", "draw (non-indexed)", "indexed", "indexed indirect"};
    int64_t pth = vk_dump_geom[0], prim = vk_dump_geom[1], it = vk_dump_geom[2], cnt = vk_dump_drawargs[0];
    fprintf(passes_f, "    geom: %s prim %lld itype %lld count %lld instances %lld %s %lld base instance %lld",
            path[pth >= 0 && pth < 4 ? pth : 0], (long long)prim, (long long)it, (long long)cnt, (long long)vk_dump_drawargs[1],
            pth == 1 ? "first" : "base vertex", (long long)vk_dump_drawargs[2], (long long)vk_dump_drawargs[3]);
    uint32_t ix[12]; int nix = 0;
    VkMemSpan ms;
    if (pth == 2 && cnt > 0 && it >= 0 && it <= 2) {
        static const int isz[3] = {1, 2, 4};
        uint64_t a = (uint64_t)vk_dump_geom[3];
        nix = cnt < 12 ? (int)cnt : 12;
        fprintf(passes_f, " ibuf %#llx", (unsigned long long)a);
        if (a && vk_mem_resolve(a, (uint64_t)nix * isz[it], &ms))
            for (int i = 0; i < nix; i++)
                ix[i] = it == 0 ? ((const uint8_t *)(uintptr_t)a)[i] : it == 1 ? ((const uint16_t *)(uintptr_t)a)[i]
                                                                     : ((const uint32_t *)(uintptr_t)a)[i];
        else nix = 0, fprintf(passes_f, " (not readable)");
        for (int i = 0; i < nix; i++) ix[i] += (uint32_t)vk_dump_drawargs[2];   // base vertex
        // (orbitals, diagnostic) the whole index list: range, and its bytes to ib_<addr>_<count>.bin (once per list)
        if (nix && vk_mem_resolve(a, (uint64_t)cnt * isz[it], &ms)) {
            uint32_t lo = ~0u, hi = 0;
            for (int64_t i = 0; i < cnt; i++) {
                uint32_t v = it == 0 ? ((const uint8_t *)(uintptr_t)a)[i] : it == 1 ? ((const uint16_t *)(uintptr_t)a)[i]
                                                                       : ((const uint32_t *)(uintptr_t)a)[i];
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
            fprintf(passes_f, " index range %u..%u", lo, hi);
            char ip[640];
            snprintf(ip, sizeof ip, "%s/ib_%llx_%lld.bin", dir, (unsigned long long)a, (long long)cnt);
            FILE *ef = fopen(ip, "rb");
            if (ef) fclose(ef);
            else if ((ef = fopen(ip, "wb"))) fwrite((const void *)(uintptr_t)a, isz[it], (size_t)cnt, ef), fclose(ef);
        }
    } else if (pth == 1 && cnt > 0) {
        uint32_t first = (uint32_t)vk_dump_geom[3];
        if (prim == NVN_DRAW_PRIMITIVE_QUADS) {   // op_draw's rewrite: 0 1 2 0 2 3 per quad
            static const uint32_t k[6] = {0, 1, 2, 0, 2, 3};
            for (int i = 0; i < 12 && i / 6 * 4 + 3 < cnt; i++) ix[nix++] = first + i / 6 * 4 + k[i % 6];
        } else for (int i = 0; i < cnt && i < 12; i++) ix[nix++] = first + (uint32_t)i;
    }
    fprintf(passes_f, " vertices:");
    for (int i = 0; i < nix; i++) fprintf(passes_f, " %u", ix[i]);
    fprintf(passes_f, "\n");
    uint64_t va = ex->vbuf[0].address, vsz = ex->vbuf[0].size, str = ex->pipe.vstream[0].stride;
    fprintf(passes_f, "    geom: stream 0 %#llx size %llu stride %llu\n", (unsigned long long)va, (unsigned long long)vsz,
            (unsigned long long)str);
    uint32_t seen[12]; int ns = 0;
    for (int i = 0; i < nix; i++) {
        int dup = 0;
        for (int j = 0; j < ns; j++) dup |= seen[j] == ix[i];
        if (dup) continue;
        seen[ns++] = ix[i];
        uint64_t a = va + (uint64_t)ix[i] * str;
        fprintf(passes_f, "    geom: vertex %u at +%llu%s", ix[i], (unsigned long long)((uint64_t)ix[i] * str),
                vsz && (uint64_t)ix[i] * str + 12 > vsz ? " (past the bound size)" : "");
        if (va && vk_mem_resolve(a, 16, &ms)) {
            const uint32_t *u = (const uint32_t *)(uintptr_t)a; const float *f = (const float *)(uintptr_t)a;
            fprintf(passes_f, " words %08x %08x %08x %08x = %g %g %g\n", u[0], u[1], u[2], u[3], f[0], f[1], f[2]);
        } else fprintf(passes_f, " not readable\n");
    }
}

static void vsb_draw(VkExec *ex, uint64_t vs);
void vk_dump_draw(VkExec *ex, const VkPipeState *s, VkPipeline p, const VkPipeInfo *info) {
    plat_mutex_lock(&rec_lock);
    if (!active()) goto out;
    pass_draws++;
    vsb_draw(ex, shader_hash(s->vs));
    if (passes_f && pass_draws <= 6)   // (orbitals) the first draws of each pass, with their arguments
        fprintf(passes_f, "  draw %d: count %lld instances %lld base vertex %lld base instance %lld vs %016llx\n", pass_draws,
                (long long)vk_dump_drawargs[0], (long long)vk_dump_drawargs[1], (long long)vk_dump_drawargs[2],
                (long long)vk_dump_drawargs[3], (unsigned long long)shader_hash(s->vs));
    if (shader_hash(s->vs) == 0x0c0640a47cb8383cull) {   // backdrop quad: note its GPU inputs (qs_flush at pass end)
        VkMemSpan m3;
        qs_w = ex->vbuf[1].address, qs_frame = frame_now();
        if (ex->ubo[0][0].addr && vk_mem_resolve(ex->ubo[0][0].addr, 0x30, &m3)) {
            qs_list = *(const uint64_t *)(uintptr_t)ex->ubo[0][0].addr & ~15ull;
            qs_b1 = *(const uint64_t *)(uintptr_t)(ex->ubo[0][0].addr + 0x10) & ~15ull;
            qs_b2 = *(const uint64_t *)(uintptr_t)(ex->ubo[0][0].addr + 0x20) & ~15ull;
            if (bd_auto && !bd_addr) bd_addr = qs_list;
        }
        qs_c5 = ex->ubo[0][2].addr;
    }
    if (qd_armed(shader_hash(s->vs)) && !qd_c3) {   // NX2NAVI_QD_VS (default the build-up quads): note GPU inputs (qd_flush)
        VkMemSpan m3;
        qs_frame = frame_now();
        snprintf(qd_pfx, sizeof qd_pfx, "d%d_", ++qd_draw_no);
        qd_geom(ex);
        qd_v = ex->vbuf[0].address, qd_c3 = ex->ubo[0][0].addr;
        qd_c5 = ex->ubo[0][2].addr, qd_c5n = ex->ubo[0][2].size ? ex->ubo[0][2].size : 0x17a0;
        qd_c6 = ex->ubo[0][3].addr, qd_c7 = ex->ubo[0][4].addr;
        {   // the driver bank this draw gets (CPU-built, uploaded as is)
            extern void vk_exec_c0_bank(const VkExec *ex, int st, uint8_t *out);
            uint8_t bank[0x800];
            vk_exec_c0_bank(ex, 0, bank);
            char path[600];
            snprintf(path, sizeof path, "%s/qsnap_%llu_%sqc0.bin", dir, (unsigned long long)qs_frame, qd_pfx);
            FILE *cf = fopen(path, "wb");
            if (cf) fwrite(bank, 1, sizeof bank, cf), fclose(cf);
        }
        if (qd_c3 && vk_mem_resolve(qd_c3, 0x70, &m3))
            for (int i = 0; i < 4; i++) {   // c3+0x30.. hold {address, size} pairs
                qd_s[i] = *(const uint64_t *)(uintptr_t)(qd_c3 + 0x30 + 0x10 * i) & ~15ull;
                qd_sn[i] = *(const uint64_t *)(uintptr_t)(qd_c3 + 0x38 + 0x10 * i);
                if (i == 1) qd_list_last = qd_s[1];
                if (qd_sn[i] > 4u << 20) qd_sn[i] = 4u << 20;
            }
        for (int i = 0; qd_c3 && vk_mem_resolve(qd_c3, 16, &m3) && i < 2; i++) {   // the buffer textures behind the handles in c3 words 0 and 2
            uint32_t h = *(const uint32_t *)(uintptr_t)(qd_c3 + 8 * i);
            uint64_t ta = 0, ts = 0;
            int tf = 0;
            uint32_t tb = 0;
            if (vk_texture_buffer_locked((int)(h & 0xfffff), &ta, &ts, &tf, &tb)) {
                qd_t[i] = ta, qd_tn[i] = ts > (4u << 20) ? 4u << 20 : ts;
                if (passes_f) fprintf(passes_f, "    qd: tex handle c3 word %d = %#x -> buffer %#llx size %#llx fmt %d bpe %u\n", 2 * i, h,
                                      (unsigned long long)ta, (unsigned long long)ts, tf, tb);
            } else if (passes_f) fprintf(passes_f, "    qd: tex handle c3 word %d = %#x is not a buffer texture\n", 2 * i, h);
        }
        if (passes_f)   // (orbitals) the vertex stage's uniform blocks c3..c10 as bound: address, size, first 8 words
            for (int u = 0; u < 8; u++) {
                const p_bind_buf *ub = &ex->ubo[0][u];
                VkMemSpan um;
                fprintf(passes_f, "    qd: vs c%d (ubo %d) addr %#llx size %#llx", u + 3, u, (unsigned long long)ub->addr,
                        (unsigned long long)ub->size);
                if (ub->addr && vk_mem_resolve(ub->addr, 32, &um)) {
                    const int32_t *w = (const int32_t *)(uintptr_t)ub->addr;
                    const float *fl = (const float *)(uintptr_t)ub->addr;
                    fprintf(passes_f, " ints %d %d %d %d %d %d %d %d floats %g %g %g %g", w[0], w[1], w[2], w[3], w[4], w[5],
                            w[6], w[7], fl[0], fl[1], fl[2], fl[3]);
                }
                fprintf(passes_f, "%c", 10);
            }
        if (passes_f) fprintf(passes_f, "    qd: v %#llx c3 %#llx c5 %#llx+%#llx\n", (unsigned long long)qd_v,
                              (unsigned long long)qd_c3, (unsigned long long)qd_c5, (unsigned long long)qd_c5n);
    }
    if (passes_f && pass_draws <= 6) {   // (orbitals) the game's stage binding: last bind mask, stages holding a shader
        fprintf(passes_f, "    nvn stages: bind mask %#x, bound:", (unsigned)ex->prog_stages);
        for (int st = 0; st < 8; st++) if (ex->stage_sd[st].data) fprintf(passes_f, " %d", st);
        fprintf(passes_f, "; vulkan fs %s\n", s->fs ? "present" : "none");
    }
    if (passes_f && pass_draws <= 6)   // (orbitals) per-draw constants of stride-0 streams (UE's primitive ID)
        for (int k = 0; k < 16; k++)
            if (ex->vbuf[k].address && !s->vstream[k].stride) {
                VkMemSpan ms;
                if (vk_mem_resolve(ex->vbuf[k].address, 4, &ms))
                    fprintf(passes_f, "    stream %d word %#x\n", k, *(const uint32_t *)(uintptr_t)ex->vbuf[k].address);
                // the primitive's GPU Scene entry: c3+0x50 holds the buffer's address, 704 bytes a primitive (as last
                // frame's GPU left it, with NX2NAVI_MEMORY=import)
                const p_bind_buf *c3 = &ex->ubo[0][0];
                VkMemSpan m3, me;
                if (k == 1 && c3->addr && vk_mem_resolve(c3->addr, 0x58, &m3) && vk_mem_resolve(ex->vbuf[k].address, 4, &ms)) {
                    uint64_t gs = *(const uint64_t *)(uintptr_t)(c3->addr + 0x50) & ~15ull;
                    uint32_t id = *(const uint32_t *)(uintptr_t)ex->vbuf[k].address & 0xfffff;
                    // arrays of 256 float4s: row k of entry id at (k * 256 + id) * 16 (the scatter uploads' indices)
                    static uint64_t mapped_gs;   // once per scene buffer: instance id -> primitive index (row 0 low 20 bits)
                    if (gs && gs != mapped_gs && vk_mem_resolve(gs, 256 * 16, &me)) {
                        mapped_gs = gs;
                        fprintf(passes_f, "    instance->primitive map of %#llx:", (unsigned long long)gs);
                        for (int q = 0; q < 64; q++) fprintf(passes_f, " %d:%u", q, *(const uint32_t *)(uintptr_t)(gs + q * 16) & 0xfffff);
                        fprintf(passes_f, "\n");
                    }
                    if (gs && id < 256 && vk_mem_resolve(gs, 16 * 256 * 16, &me)) {   // (each read checked: F9 crash)
                        fprintf(passes_f, "    gpuscene %#llx id %u:", (unsigned long long)gs, id);
                        for (int q = 0; q < 16; q++) {
                            const float *f = (const float *)(uintptr_t)(gs + ((uint64_t)q * 256 + id) * 16);
                            fprintf(passes_f, " | %g %g %g %g", f[0], f[1], f[2], f[3]);
                        }
                        fprintf(passes_f, "\n");
                    }
                    // the primitive record (UE SceneData.ush: PrimitiveId * 704 bytes, item 1 PositionHigh, items 2-4
                    // LocalToRelativeWorld) in the buffer whose address c3+0 holds; the primitive index is the low
                    // 20 bits of the instance's row 0 word
                    VkMemSpan mp;
                    // (the shaders take the base from c3 byte 0x20 or 0x28: the 0x32000-byte buffer bound there)
                    uint64_t pb = 0;
                    for (int co = 0x20; co <= 0x28 && !pb; co += 8)
                        for (int r = 0; r < 16 && !pb; r++) {
                            uint64_t a = *(const uint64_t *)(uintptr_t)(c3->addr + co) & ~15ull;
                            if (a && ex->ssbo[0][r].addr == a) pb = a;
                        }
                    if (!pb) pb = *(const uint64_t *)(uintptr_t)(c3->addr + 0x20) & ~15ull;
                    if (gs && pb && vk_mem_resolve(gs + (uint64_t)id * 16, 4, &mp)) {
                        uint32_t prim = *(const uint32_t *)(uintptr_t)(gs + (uint64_t)id * 16) & 0xfffff;
                        if (vk_mem_resolve(pb + (uint64_t)prim * 704, 704, &mp)) {
                            fprintf(passes_f, "    primitive %u at %#llx:", prim, (unsigned long long)(pb + (uint64_t)prim * 704));
                            for (int q = 0; q < 8; q++) {
                                const float *f = (const float *)(uintptr_t)(pb + (uint64_t)prim * 704 + q * 16);
                                fprintf(passes_f, " | %g %g %g %g", f[0], f[1], f[2], f[3]);
                            }
                            fprintf(passes_f, "\n");
                        }
                    }
                }
            }
    uint64_t th = bindings_hash(ex, info);
    for (int i = 0; i < nents; i++)
        if (ents[i].p == p && ents[i].th == th) {
            ents[i].draws++;
            goto out;
        }
    if (nents == MAX_ENTS) {
        ents_over++;
        goto out;
    }
    Str b = {0};
    sp(&b, "vs %016llx fs %016llx prog %llu prim %d\n", (unsigned long long)shader_hash(s->vs),
       (unsigned long long)shader_hash(s->fs), (unsigned long long)(ex->prog ? ex->prog->serial : 0), s->prim);
    {   // (orbitals) the state that can keep a draw's colour off its targets
        const unsigned char *cm = (const unsigned char *)&s->chmask, *co = (const unsigned char *)&s->color;
        sp(&b, "    state: depth test %d write %d func %d, stencil test %d func %d/%d ops %d,%d,%d, raster discard %d, a2c %d, chmask",
           (int)s->ds.dt, (int)s->ds.dw, (int)s->ds.df, (int)s->ds.st, (int)s->ds.sf_f, (int)s->ds.sf_b, (int)s->ds.fail_f,
           (int)s->ds.zfail_f, (int)s->ds.zpass_f, s->raster_discard, s->ms.a2c);
        for (size_t k = 0; k < sizeof s->chmask; k++) sp(&b, " %02x", cm[k]);
        sp(&b, ", stencil ref %u mask %x write %x", ex->st_ref[0], ex->st_vmask[0], ex->st_wmask[0]);
        {
            VkRect2D sc = vk_exec_scissor(ex);
            sp(&b, ", vp %.0f,%.0f %.0fx%.0f z %.3f..%.3f, scissor %d,%d %ux%u", ex->vp[0][0], ex->vp[0][1], ex->vp[0][2],
               ex->vp[0][3], ex->vp[0][4], ex->vp[0][5], sc.offset.x, sc.offset.y, sc.extent.width, sc.extent.height);
        }
        sp(&b, ", color");
        for (size_t k = 0; k < sizeof s->color && k < 16; k++) sp(&b, " %02x", co[k]);
        {   // the Vulkan write mask and NVN blend factors actually used for target 0 (vk_pipeline.c target_wmask)
            extern uint32_t vk_pipeline_target_wmask(const VkPipeState *s, int i);
            const blend_st *bl = &s->blend[0];
            sp(&b, ", t0 wmask %x blend %d src %d dst %d eq %d / a src %d dst %d eq %d", vk_pipeline_target_wmask(s, 0),
               (int)(s->color.blend_en & 1), (int)bl->src_rgb, (int)bl->dst_rgb, (int)bl->eq_rgb, (int)bl->src_a,
               (int)bl->dst_a, (int)bl->eq_a);
            sp(&b, ", cull %d front_cw %d poly %d", (int)s->poly.cull, (int)s->poly.front, (int)s->poly.mode);
        }
        sp(&b, "\n    vin:");
        for (int k = 0; k < 16; k++) {
            const va_st *va = (const va_st *)&s->vattr[k];
            if (!s->vattr[k]) continue;
            sp(&b, " a%d fmt %u off %u s%u", k, va->fmt, va->offset, va->stream);
        }
        sp(&b, " |");
        for (int k = 0; k < 16; k++)
            if (ex->vbuf[k].address)
                sp(&b, " s%d %#llx+%llu stride %u div %u", k, (unsigned long long)ex->vbuf[k].address,
                   (unsigned long long)ex->vbuf[k].size, s->vstream[k].stride, s->vstream[k].divisor);
        sp(&b, " ||");
        for (size_t k = 0; k < sizeof s->color && k < 16; k++) sp(&b, " %02x", co[k]);
        sp(&b, "\n");
    }
    describe_bindings(&b, ex, info);
    ents[nents++] = (Ent){p, th, 1, b.s};
out:
    plat_mutex_unlock(&rec_lock);
}

static uint32_t roi_w, roi_h;   // set for one target_line call: copy only that rectangle (at roi_x, roi_y)
static void target_line(VkExec *ex, const VkExecTarget *t, const char *tag, int n) {
    char fn[16], name[160];
    if (!t->image) {
        fprintf(passes_f, "  %s: nvn %#x target %d %ux%u: not renderable, left out\n", tag, t->tex->d.format,
                t->tex->d.target, t->tex->d.width, t->tex->d.height);
        return;
    }
    snprintf(name, sizeof name, "pass_%04d_%s_%ux%u_%d_%s", n, tag, t->width, t->height, (int)t->format,
             fmt_name(t->format, fn));
    // a block view: the plane's blocks, read as the view format (one texel per block)
    uint32_t tw = t->tex->d.width >> t->level ? t->tex->d.width >> t->level : 1;
    uint32_t th = t->tex->d.height >> t->level ? t->tex->d.height >> t->level : 1;
    (void)n;
    const char *why = pass_skip ? pass_skip : t->samples > 1 ? "multisampled" :
                      t->blocks ? copy_region(ex, t->image, t->format, t->aspect, t->level, t->is3d ? 0 : t->base_layer,
                                              t->is3d ? t->base_layer : 0, (VkExtent3D){tw, th, 1}, t->width,
                                              t->height, 1, name) :
                      t->is3d ? "3D slice (the volume is dumped as lut<id> where it is sampled)" :
                      roi_w ? copy_region(ex, t->image, t->format, t->aspect, t->level, t->base_layer, 0,
                                          (VkExtent3D){roi_w, roi_h, 1}, roi_w, roi_h, 1, name) :
                      copy_out(ex, t->image, t->format, t->aspect, t->level, t->base_layer, t->width, t->height, 1, name);
    roi_x = roi_y = 0;
    fprintf(passes_f, "  %s: nvn %#x %s(%d) %ux%u level %u layers %u+%u samples %u addr %#llx vkimage %p -> %s.png%s%s\n",
            tag, t->nvn_format, fmt_name(t->format, fn), (int)t->format, t->width, t->height, t->level, t->base_layer,
            t->layers, t->samples, (unsigned long long)(uintptr_t)t->tex->mem, (void *)t->image, name, why ? " skipped: " : "",
            why ? why : "");
}

static uint64_t targets_sig(const VkExec *ex) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        uint64_t v[3] = {(uint64_t)(uintptr_t)t->image, t->level, t->base_layer};
        for (int k = 0; k < 3; k++) h = (h ^ v[k]) * 1099511628211ull;
    }
    return h;
}

// Sets pass_skip for the pass or dispatch being logged and counts it when it copies.
static void pass_decide(int same_targets) {
    if (frame_now() != dump_frame) copied = 0;
    pass_skip = copied >= MAX_PASSES ? "pass cap" : same_targets ? "same targets as the previous pass" : NULL;
    if (!pass_skip) copied++;
}

// (orbitals) NX2NAVI_DUMP_VSBUF=<vs hash hex>: for each draw with that vertex shader in the dumped frame, every vertex
// stream and buffer texture it binds is saved twice: vsb_f<frame>_p<pass>_d<draw>_<s|b><n>_<addr>_cpu.bin (guest bytes at
// record) and ..._gpu.bin (the GPU buffer copied after the pass ends). At most 4 MiB each.
static struct { uint64_t addr, size; int binding; } vsb_tex[16];
static int vsb_ntex;
void vk_dump_vsb_texel(uint64_t addr, uint64_t size, int binding) {   // under rec_lock (vk_dump_texel)
    if (vsb_ntex < 16) vsb_tex[vsb_ntex].addr = addr, vsb_tex[vsb_ntex].size = size, vsb_tex[vsb_ntex].binding = binding, vsb_ntex++;
}
typedef struct { VkBuffer buf; VmaAllocation alloc; void *map; uint64_t n; char path[600]; } VsbJob;
static struct { VkBuffer buf; VkDeviceSize off, n; char path[600]; } vsb_pend[64];
static int vsb_n;
// (orbitals) one copy per (address, length) per dumped frame: persistent buffers (GPU scene, shared vertex pools) are bound
// by hundreds of draws. passes.txt / texels.txt give each draw's addresses; the file name carries address and length.
static struct { uint64_t addr, n; } vsb_seen[8192];
static uint64_t vsb_seen_frame;
static int vsb_fresh_now;   // set while saving a NX2NAVI_DUMP_VSBUF_FRESH draw: its buffers are saved again at that draw
static int vsb_dup(uint64_t addr, uint64_t n) {
    if (vsb_fresh_now) return 0;
    if (vsb_seen_frame != dump_frame) memset(vsb_seen, 0, sizeof vsb_seen), vsb_seen_frame = dump_frame;
    uint32_t i = (uint32_t)(((addr ^ n * 0x9E3779B97F4A7C15ull) * 0xC2B2AE3D27D4EB4Full) >> 51);
    for (int k = 0; k < 64; k++, i = (i + 1) & 8191) {
        if (vsb_seen[i].addr == addr && vsb_seen[i].n == n) return 1;
        if (!vsb_seen[i].addr) { vsb_seen[i].addr = addr, vsb_seen[i].n = n; return 0; }
    }
    return 0;
}
static void vsb_save_cap(uint64_t addr, uint64_t size, char kind, int idx, int draw, uint64_t cap, int snap) {
    VkMemSpan ms;
    if (!addr || vsb_n >= 64 || !vk_mem_resolve(addr, 1, &ms)) return;
    uint64_t n = size && size < ms.avail ? size : ms.avail;
    if (n > cap) n = cap;
    if (!n || vsb_dup(addr, n)) return;
    {   // NX2NAVI_DUMP_VSBUF_MB (default 3072): bytes saved per dumped frame, both views counted; the rest is skipped
        static uint64_t used, budget, at;
        if (!budget) { const char *e = cfg_get("NX2NAVI_DUMP_VSBUF_MB"); budget = (e && *e ? strtoull(e, NULL, 10) : 3072) << 20; }
        if (at != dump_frame) used = 0, at = dump_frame;
        if (used + 2 * n > budget) {
            static uint64_t warned;
            if (warned != dump_frame) warned = dump_frame, rt_log("vk_dump: DUMP_VSBUF byte budget reached; later buffers not saved");
            return;
        }
        used += 2 * n;
    }
    char base[560];
    snprintf(base, sizeof base, "%s/vsb_f%llu_p%04d_d%d_%c%d_%llx_%llx", dir, (unsigned long long)frame_now(), seq, draw, kind, idx,
             (unsigned long long)addr, (unsigned long long)n);
    char p[600];
    snprintf(p, sizeof p, "%s_cpu.bin", base);
    FILE *f = fopen(p, "wb");
    if (f) fwrite((const void *)(uintptr_t)addr, 1, n, f), fclose(f);
    VkBuffer gb = ms.buffer; VkDeviceSize go = ms.offset;
    VkDescriptorBufferInfo cb;   // a uniform range the draw reads from the submit-time copy (vk_usnap.c): that copy is the GPU's view
    if (snap && vk_usnap_find(addr, n, 1, &cb)) gb = cb.buffer, go = cb.offset;
    if (!gb) return;
    vsb_pend[vsb_n].buf = gb, vsb_pend[vsb_n].off = go, vsb_pend[vsb_n].n = n;
    snprintf(vsb_pend[vsb_n].path, sizeof vsb_pend[vsb_n].path, "%s_gpu%s.bin", base, gb != ms.buffer ? "snap" : "");
    vsb_n++;
}
static void vsb_save(uint64_t addr, uint64_t size, char kind, int idx, int draw) { vsb_save_cap(addr, size, kind, idx, draw, 4u << 20, 0); }
// the storage buffers a wanted draw's vertex stage reads (GPU scene instance/primitive data): noted by the pipeline
// description (describe_bindings), saved with the draw
static struct { uint64_t addr; int idx; } vsb_ssbo[16];
static int vsb_nssbo;
static void vsb_note_ssbo(uint64_t addr, int idx) {
    for (int i = 0; i < vsb_nssbo; i++) if (vsb_ssbo[i].addr == addr) return;   // kept across draws: the description may follow the draw
    if (addr && vsb_nssbo < 16) vsb_ssbo[vsb_nssbo].addr = addr, vsb_ssbo[vsb_nssbo].idx = idx, vsb_nssbo++;
}
static struct { uint64_t addr; int idx; } vsb_cs_ssbo[16];
static int vsb_ncs_ssbo;
static void vsb_note_cs_ssbo(uint64_t addr, int idx) {
    for (int i = 0; i < vsb_ncs_ssbo; i++) if (vsb_cs_ssbo[i].addr == addr) return;
    if (addr && vsb_ncs_ssbo < 16) vsb_cs_ssbo[vsb_ncs_ssbo].addr = addr, vsb_cs_ssbo[vsb_ncs_ssbo].idx = idx, vsb_ncs_ssbo++;
}
static void vsb_flush(VkExec *ex);
// (orbitals) NX2NAVI_DUMP_VSBUF=all also saves every compute dispatch's inputs and outputs (Niagara GPU simulations
// read and write their particles through buffer textures and images): its buffer textures/images (b), uniform blocks
// (u) and storage buffers (g), as vsb_f<frame>_p<pass>_d0_..., guest bytes at record (_cpu) and the GPU buffer
// copied right after the dispatch (_gpu). Called after vkCmdDispatch, with seq already past this dispatch (n).
static void vsb_dispatch(VkExec *ex, int n) {   // under rec_lock, from vk_dump_dispatch
    static int on = -1;
    if (on < 0) { const char *e = cfg_get("NX2NAVI_DUMP_VSBUF"); on = e && !strcmp(e, "all"); }
    if (on) {
        int keep = seq;
        seq = n;
        for (int i = 0; i < vsb_ntex; i++) vsb_save_cap(vsb_tex[i].addr, vsb_tex[i].size, 'b', vsb_tex[i].binding, 0, 16u << 20, 0);
        for (int k = 0; k < 18; k++) vsb_save_cap(ex->ubo[5][k].addr, ex->ubo[5][k].size, 'u', k + 3, 0, 64u << 10, 1);
        for (int i = 0; i < vsb_ncs_ssbo; i++) vsb_save_cap(vsb_cs_ssbo[i].addr, 0, 'g', vsb_cs_ssbo[i].idx, 0, 16u << 20, 0);
        seq = keep;
        vsb_flush(ex);
    }
    vsb_ntex = 0, vsb_ncs_ssbo = 0;
}
static void vsb_draw(VkExec *ex, uint64_t vs) {   // under rec_lock, from vk_dump_draw
    static int on = -1, nwant, all;   // NX2NAVI_DUMP_VSBUF takes up to 8 comma-separated hashes, or "all"
    static uint64_t want[8];
    if (on < 0) {
        const char *e = cfg_get("NX2NAVI_DUMP_VSBUF");
        if (e && !strcmp(e, "all")) all = 1, e = NULL;
        for (char *q = (char *)e; q && *q && nwant < 8; q += *q == ',') {
            uint64_t h = strtoull(q, &q, 16);
            if (h) want[nwant++] = h;
            else break;
        }
        on = nwant > 0 || all;
    }
    // NX2NAVI_DUMP_VSBUF_FRESH=<vs hashes, up to 8>: those draws save their buffers at the draw itself, even when the
    // same address was already saved earlier in the frame (GPU-written particle data changes between passes)
    static int nfresh = -1;
    static uint64_t fresh[8];
    if (nfresh < 0) {
        nfresh = 0;
        const char *e = cfg_get("NX2NAVI_DUMP_VSBUF_FRESH");
        for (char *q = (char *)e; q && *q && nfresh < 8; q += *q == ',') {
            uint64_t h = strtoull(q, &q, 16);
            if (h) fresh[nfresh++] = h;
            else break;
        }
        if (nfresh) on = 1;
    }
    int hit = all;
    for (int i = 0; i < nwant; i++) hit |= vs == want[i];
    for (int i = 0; i < nfresh; i++) if (vs == fresh[i]) hit = 1, vsb_fresh_now = 1;
    if (on && hit) {
        // Ribbon vertex IDs encode segment/tessellation/shape fields. Replaying
        // 0..indexCount-1 invents vertices: retain the actual index stream along
        // with fresh draw-time particle data, including GPU-generated indices.
        if (vsb_fresh_now && (vk_dump_geom[0] == 2 || vk_dump_geom[0] == 3)) {
            uint64_t ib = (uint64_t)vk_dump_geom[3];
            if (passes_f)
                fprintf(passes_f, "    index snapshot: type %lld addr %#llx; GPU bytes in vsb i<type> file\n",
                        (long long)vk_dump_geom[2], (unsigned long long)ib);
            vsb_save_cap(ib, 0, 'i', (int)vk_dump_geom[2], pass_draws, 4u << 20, 0);
        }
        for (int k = 0; k < 16; k++) vsb_save(ex->vbuf[k].address, ex->vbuf[k].size, 's', k, pass_draws);
        for (int i = 0; i < vsb_ntex; i++) vsb_save(vsb_tex[i].addr, vsb_tex[i].size, 'b', vsb_tex[i].binding, pass_draws);
        for (int k = 0; k < 18; k++)   // the vertex stage's uniform blocks, exact bytes (c3 and up)
            vsb_save_cap(ex->ubo[0][k].addr, ex->ubo[0][k].size, 'u', k + 3, pass_draws, 64u << 10, 1);
        for (int i = 0; i < vsb_nssbo; i++) vsb_save_cap(vsb_ssbo[i].addr, 0, 'g', vsb_ssbo[i].idx, pass_draws, 16u << 20, 0);
    }
    vsb_fresh_now = 0;
    vsb_ntex = 0;
}
static void vsb_done(void *arg) {
    VsbJob *j = arg;
    VmaAllocator vma = vk_ctx()->vma;
    vmaInvalidateAllocation(vma, j->alloc, 0, VK_WHOLE_SIZE);
    FILE *f = fopen(j->path, "wb");
    if (f) fwrite(j->map, 1, j->n, f), fclose(f);
    vmaDestroyBuffer(vma, j->buf, j->alloc);
    free(j);
}
static void vsb_flush(VkExec *ex) {
    for (int i = 0; i < vsb_n; i++) {
        VsbJob *j = calloc(1, sizeof *j);
        if (!j) break;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = vsb_pend[i].n, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info) != VK_SUCCESS) { free(j); break; }
        j->map = info.pMappedData, j->n = vsb_pend[i].n;
        snprintf(j->path, sizeof j->path, "%s", vsb_pend[i].path);
        VkMemoryBarrier2 pre = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
        VkMemoryBarrier2 post = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                 .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 // ALL_COMMANDS too: later work (the next dispatch) may overwrite the buffer just read
                                 .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &pre};
        vkCmdPipelineBarrier2(ex->cb, &di);
        VkBufferCopy r = {vsb_pend[i].off, 0, vsb_pend[i].n};
        vkCmdCopyBuffer(ex->cb, vsb_pend[i].buf, j->buf, 1, &r);
        di.pMemoryBarriers = &post;
        vkCmdPipelineBarrier2(ex->cb, &di);
        ex->recorded += 3;
        vk_completion_add(ex->fence, vsb_done, j);
    }
    vsb_n = 0;
}

// Indirect draw arguments of the dumped frame: the CPU view goes to passes.txt at the draw (before its "draw" line),
// the GPU view is copied after the pass ends and logged as "vk_dump: indirect ... (gpu)" when the copy completes.
typedef struct { VkBuffer buf; VmaAllocation alloc; uint32_t *map; uint64_t gaddr, frame; int pass, draw; } IndJob;
static struct { VkBuffer buf; VkDeviceSize off; uint64_t gaddr; int draw; } ind_pend[64];
static int ind_n;
void vk_dump_ind_note(VkExec *ex, uint64_t gaddr, VkBuffer buf, VkDeviceSize off) {
    (void)ex;
    plat_mutex_lock(&rec_lock);
    if (active()) {
        const uint32_t *w = (const uint32_t *)(uintptr_t)gaddr;
        fprintf(passes_f, "  indirect args at %#llx (cpu view at record): count %u instances %u first %u vofs %d binst %u\n",
                (unsigned long long)gaddr, w[0], w[1], w[2], (int32_t)w[3], w[4]);
        if (ind_n < 64) ind_pend[ind_n].buf = buf, ind_pend[ind_n].off = off, ind_pend[ind_n].gaddr = gaddr, ind_pend[ind_n].draw = pass_draws, ind_n++;
    }
    plat_mutex_unlock(&rec_lock);
}
static void ind_done(void *arg) {
    IndJob *j = arg;
    VmaAllocator vma = vk_ctx()->vma;
    vmaInvalidateAllocation(vma, j->alloc, 0, VK_WHOLE_SIZE);
    rt_log("vk_dump: indirect frame %llu pass %04d draw %d args %#llx (gpu): count %u instances %u first %u vofs %d binst %u",
           (unsigned long long)j->frame, j->pass, j->draw, (unsigned long long)j->gaddr, j->map[0], j->map[1], j->map[2],
           (int32_t)j->map[3], j->map[4]);
    vmaDestroyBuffer(vma, j->buf, j->alloc);
    free(j);
}
static void ind_flush(VkExec *ex, int pass) {
    for (int i = 0; i < ind_n; i++) {
        IndJob *j = calloc(1, sizeof *j);
        if (!j) break;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 20, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info) != VK_SUCCESS) { free(j); break; }
        j->map = info.pMappedData, j->gaddr = ind_pend[i].gaddr, j->frame = frame_now(), j->pass = pass, j->draw = ind_pend[i].draw;
        VkMemoryBarrier2 pre = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
        VkMemoryBarrier2 post = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                 .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 // ALL_COMMANDS too: later work (the next dispatch) may overwrite the buffer just read
                                 .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &pre};
        vkCmdPipelineBarrier2(ex->cb, &di);
        VkBufferCopy r = {ind_pend[i].off, 0, 20};
        vkCmdCopyBuffer(ex->cb, ind_pend[i].buf, j->buf, 1, &r);
        di.pMemoryBarriers = &post;
        vkCmdPipelineBarrier2(ex->cb, &di);
        ex->recorded += 3;
        vk_completion_add(ex->fence, ind_done, j);
    }
    ind_n = 0;
}

void vk_dump_pass_end(VkExec *ex) {
    plat_mutex_lock(&rec_lock);
    if (dump_after && dump_frame == UINT64_MAX && ex->depth.tex && (uint64_t)(uintptr_t)ex->depth.tex->mem == dump_after) {
        dump_frame = frame_now();   // this pass and the rest of its frame
        rt_log("vk_dump: a pass drew into %#llx in frame %llu: it and the rest of the frame go to %s",
               (unsigned long long)dump_after, (unsigned long long)dump_frame, dir);
    }
    if (!active()) goto out;
    int n = seq++;
    ind_flush(ex, n);
    vsb_flush(ex);
    uint64_t sig = targets_sig(ex);
    static int nodepth = -1;   // NX2NAVI_DUMP_NO_DEPTH=1: no depth images; depth-only passes do not count to the cap
    if (nodepth < 0) nodepth = rt_knob("DUMP_NO_DEPTH", "0") != 0;
    int has_color = 0;
    for (int i = 0; i < 8; i++) has_color |= ex->color[i].tex != NULL;
    if (nodepth && !has_color) pass_skip = "depth only (NX2NAVI_DUMP_NO_DEPTH)";
    else pass_decide(skip_runs && sig == last_targets);
    last_targets = sig;
    fprintf(passes_f, "pass %04d: render pass, draws %d, area %ux%u layers %u, fence %llu, frame %llu\n", n, pass_draws,
            ex->pass_w, ex->pass_h, ex->pass_layers, (unsigned long long)ex->fence, (unsigned long long)frame_now());
    static int thumbs = -1;
    if (thumbs < 0) thumbs = rt_knob("DUMP_SPLIT_THUMB", "0") != 0;
    // split pass: t0 only (SPLIT_ADDR, or any split with SPLIT_THUMB, as small copies)
    int split_t0 = !pass_skip && (cfg_get("NX2NAVI_DUMP_SPLIT_ADDR") || thumbs) && vk_dump_split(ex);
    for (int i = 0; i < 8; i++) {
        if (!ex->color[i].tex) continue;
        char tag[8];
        snprintf(tag, sizeof tag, "t%d", i);
        const char *sv = pass_skip;
        static int split_t = -1;   // NX2NAVI_DUMP_SPLIT_T=<i>: a split pass saves target i (when it has one) instead of t0
        if (split_t < 0) { const char *e = cfg_get("NX2NAVI_DUMP_SPLIT_T"); split_t = e && *e ? atoi(e) & 7 : 0; }
        int keep = ex->color[split_t].tex ? split_t : 0;
        if (split_t0 && i != keep) pass_skip = "split pass, one target only (NX2NAVI_DUMP_SPLIT_ADDR)";
        if (split_t0 && i == keep) {   // NX2NAVI_DUMP_SPLIT_ROI=x,y,w,h: a split pass's t0 as that rectangle only
            const char *r = cfg_get("NX2NAVI_DUMP_SPLIT_ROI");
            int x, y, w, h;
            if (r && sscanf(r, "%d,%d,%d,%d", &x, &y, &w, &h) == 4 && w > 0 && h > 0 &&
                (uint32_t)(x + w) <= ex->color[0].width && (uint32_t)(y + h) <= ex->color[0].height)
                roi_x = x, roi_y = y, roi_w = (uint32_t)w, roi_h = (uint32_t)h;
        }
        thumb_next = split_t0 && thumbs && !roi_w;
        target_line(ex, &ex->color[i], tag, n);
        thumb_next = 0;
        roi_w = roi_h = 0, roi_x = roi_y = 0;
        pass_skip = sv;
    }
    if (ex->depth.tex) {
        const char *sv = pass_skip;
        if (nodepth) pass_skip = "depth (NX2NAVI_DUMP_NO_DEPTH)";
        target_line(ex, &ex->depth, "td", n);
        pass_skip = sv;
    }
    for (int i = 0; i < nents; i++) {
        fprintf(passes_f, "  pipe %d (draws %d): %s", i, ents[i].draws, ents[i].text);
        free(ents[i].text);
    }
    if (ents_over) fprintf(passes_f, "  ... %d more draws with other pipeline/texture sets\n", ents_over);
    if (!pass_skip) flush_luts(ex, n);
    else nluts = 0;
    buf_snap(ex, n);
    qs_flush(ex);
    fflush(passes_f);
    nents = ents_over = pass_draws = 0;
out:
    plat_mutex_unlock(&rec_lock);
}

// (orbitals, diagnostic) the instance-list producer d0b14dbdb85fcb86 in the dumped frame: its c3 block, every
// {address, size} pair c3 names and its bound storage slots, copied in GPU order before and after the dispatch. Before
// the dispatch, 64 bytes at the list address the build-up draw used last frame are filled with 0xA5A5A5A5, so slots
// the producer does not write keep the sentinel.
static int prod_n;
static void prod_capture(VkExec *ex, const char *when) {
    char tag[24];
    const p_bind_buf *c3 = &ex->ubo[5][0];
    VkMemSpan m;
    qs_frame = frame_now();
    if (c3->addr && vk_mem_resolve(c3->addr, 0x100, &m)) {
        snprintf(tag, sizeof tag, "p%d%s_c3", prod_n, when);
        qs_copy(ex, c3->addr, 0x100, tag);
        for (int k = 0; k < 16; k++) {
            uint64_t a = *(const uint64_t *)(uintptr_t)(c3->addr + 16 * k), n = *(const uint64_t *)(uintptr_t)(c3->addr + 16 * k + 8);
            if (a < 0xf00000000ull || a >= 0x1000000000ull || !n || n > (4u << 20)) continue;
            snprintf(tag, sizeof tag, "p%d%s_c3w%d", prod_n, when, k);
            qs_copy(ex, a & ~15ull, n, tag);
            if (passes_f && *when == 'b') fprintf(passes_f, "    prod: c3+%#x -> %#llx size %#llx%s\n", 16 * k, (unsigned long long)a,
                                                 (unsigned long long)n, qd_list_last >= a && qd_list_last < a + n ? " (holds the list)" : "");
        }
    }
    for (int n = 0; c3->addr && n < 8; n += 2) {   // buffer textures/images behind c3 handle words 0, 2, 4, 6
        if (!vk_mem_resolve(c3->addr + 4 * n, 4, &m)) break;
        uint32_t h = *(const uint32_t *)(uintptr_t)(c3->addr + 4 * n);
        uint64_t ta = 0, ts = 0;
        int tf = 0;
        uint32_t tb = 0;
        if (!h || !vk_texture_buffer_locked((int)(h & 0xfffff), &ta, &ts, &tf, &tb)) {
            if (passes_f && *when == 'b') fprintf(passes_f, "    prod: c3 handle word %d = %#x: not a buffer texture\n", n, h);
            continue;
        }
        snprintf(tag, sizeof tag, "p%d%s_h%d", prod_n, when, n);
        qs_copy(ex, ta, ts > (1u << 20) ? 1u << 20 : ts, tag);
        if (passes_f && *when == 'b') fprintf(passes_f, "    prod: c3 handle word %d = %#x -> buffer %#llx size %#llx fmt %d bpe %u\n", n, h,
                                             (unsigned long long)ta, (unsigned long long)ts, tf, tb);
    }
    for (int k = 0; k < 8; k++) {   // every uniform block of the dispatch
        const p_bind_buf *ub = &ex->ubo[5][k];
        if (!ub->addr || !vk_mem_resolve(ub->addr, 0x100, &m)) continue;
        snprintf(tag, sizeof tag, "p%d%s_ub%d", prod_n, when, k);
        qs_copy(ex, ub->addr, 0x100, tag);
    }
    for (int k = 0; k < 16; k++) {
        const p_bind_buf *sb = &ex->ssbo[5][k];
        if (!sb->addr || !sb->size || sb->size > (4u << 20)) continue;
        snprintf(tag, sizeof tag, "p%d%s_ss%d", prod_n, when, k);
        qs_copy(ex, sb->addr, sb->size, tag);
        if (passes_f && *when == 'b') fprintf(passes_f, "    prod: ssbo %d -> %#llx size %#llx\n", k, (unsigned long long)sb->addr,
                                             (unsigned long long)sb->size);
    }
}
// (orbitals, diagnostic) NX2NAVI_PROD_CS=<hash hex>: the compute shader whose dispatches in the dumped frame get the
// before/after capture (prod_capture); default the instance-list producer d0b14dbd (which also gets the list sentinel)
static uint64_t prod_cs(void) {
    static uint64_t h;
    if (!h) {
        const char *e = cfg_get("NX2NAVI_PROD_CS");
        h = e && *e ? strtoull(e, NULL, 16) : 0xd0b14dbdb85fcb86ull;
    }
    return h;
}
void vk_dump_dispatch_pre(VkExec *ex, const GfxShader *cs) {
    plat_mutex_lock(&rec_lock);
    if (active() && frame_now() == dump_frame && shader_hash(cs) == prod_cs()) {
        prod_capture(ex, "b");
        VkMemSpan s;
        if (prod_cs() == 0xd0b14dbdb85fcb86ull && qd_list_last && vk_mem_resolve(qd_list_last, 64, &s) && s.avail >= 64) {
            VK_MEM_PICK(s);
            VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
            VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
            vkCmdPipelineBarrier2(ex->cb, &di);
            vkCmdFillBuffer(ex->cb, s.buffer, s.offset & ~3ull, 64, 0xA5A5A5A5u);
            vkCmdPipelineBarrier2(ex->cb, &di);
            ex->recorded += 3;
            if (passes_f) fprintf(passes_f, "    prod %d: sentinel 0xA5A5A5A5 over 64 bytes at %#llx\n", prod_n, (unsigned long long)qd_list_last);
        }
    }
    plat_mutex_unlock(&rec_lock);
}

void vk_dump_dispatch(VkExec *ex, const GfxShader *cs, VkPipeline p, const VkPipeInfo *info, const p_dispatch *d) {
    if (frame_now() == dump_frame && shader_hash(cs) == prod_cs()) {
        plat_mutex_lock(&rec_lock);
        if (active()) prod_capture(ex, "a"), prod_n++;
        plat_mutex_unlock(&rec_lock);
    }
    (void)p;
    plat_mutex_lock(&rec_lock);
    if (!active()) goto out;
    int n = seq++;
    pass_decide(0);
    last_targets = 0;
    Str b = {0};
    describe_bindings(&b, ex, info);
    fprintf(passes_f, "pass %04d: dispatch %dx%dx%d cs %016llx prog %llu caller %#llx, fence %llu\n%s", n, d->x, d->y,
            d->z, (unsigned long long)shader_hash(cs), (unsigned long long)(ex->prog ? ex->prog->serial : 0),
            (unsigned long long)d->caller, (unsigned long long)ex->fence, b.s ? b.s : "");
    free(b.s);
    vsb_dispatch(ex, n);
    if (shader_hash(cs) == 0x3924ea4d345459d6ull) {   // (orbitals) UE's GPU Scene scatter upload: what it is asked to write
        const p_bind_buf *c3 = &ex->ubo[5][0], *c4 = &ex->ubo[5][1];
        VkMemSpan m;
        if (c4->addr && vk_mem_resolve(c4->addr, 16, &m)) {
            const uint32_t *w = (const uint32_t *)(uintptr_t)c4->addr;
            fprintf(passes_f, "    scatter: per-item %u items %u start %u w %u", w[0], w[1], w[2], w[3]);
            if (c3->addr && vk_mem_resolve(c3->addr, 0x30, &m)) {
                uint64_t dst = *(const uint64_t *)(uintptr_t)c3->addr, idx = *(const uint64_t *)(uintptr_t)(c3->addr + 0x10);
                fprintf(passes_f, " dst %#llx idx %#llx:", (unsigned long long)dst, (unsigned long long)idx);
                if (idx && vk_mem_resolve(idx, 4, &m))
                    for (uint32_t q = 0; q < w[1] && q < 48 && (q + 1) * 4 <= m.avail; q++)
                        fprintf(passes_f, " %u", ((const uint32_t *)(uintptr_t)idx)[q]);
            }
            fprintf(passes_f, "\n");
        }
    }
    if (dump_ubo) {   // NX2NAVI_DUMP_UBO: the NVN2 push constants as 64-bit words
        fprintf(passes_f, "    push:");
        for (int q = 0; q < 0x80; q += 8) {
            uint64_t v;
            memcpy(&v, ex->cs_push + q, 8);
            fprintf(passes_f, " %02x:%llx", q, (unsigned long long)v);
        }
        fprintf(passes_f, "\n");
    }
    for (uint32_t i = 0; i < info->n_bindings && !pass_skip; i++) {   // storage images the dispatch writes
        const VkPipeBinding *bd = &info->bindings[i];
        if (!bd->res || bd->kind != GFX_RES_IMG || !(bd->res->flags & 4) || bd->type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
            continue;
        uint64_t h = handle_of(ex, 5, bd->res->slot, 1);
        view_t v;
        int hv = 0;
        const VkTexImage *t = h ? vk_texture_by_id(NVN_HANDLE_TEX(h), &v, &hv) : NULL;
        if (!t) continue;
        uint32_t lv = hv && v.set_levels ? v.base_level : 0, layer = hv && v.set_layers ? v.min_layer : 0;
        if (lv >= t->levels || layer >= t->layers) continue;
        uint32_t w = t->width >> lv ? t->width >> lv : 1, hh = t->height >> lv ? t->height >> lv : 1;
        uint32_t dd = t->type == VK_IMAGE_TYPE_3D ? (t->depth >> lv ? t->depth >> lv : 1) : 1;
        char fn[16], name[160];
        snprintf(name, sizeof name, "pass_%04d_img%d_%ux%u%s_%d_%s", n, bd->res->slot, w, hh, dd > 1 ? "x3d" : "",
                 (int)t->format, fmt_name(t->format, fn));
        if (t->type == VK_IMAGE_TYPE_3D) {   // a written volume: this copy replaces the sampled-3D one
            for (int k = 0; k < nluts_done; k++)
                if (luts_done[k] == t->image) luts_done[k] = VK_NULL_HANDLE;
        }
        const char *why = t->samples > 1 ? "multisampled" :
                          copy_out(ex, t->image, vk_exec_raw_9e5(t->image, t->format) ? VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 : t->format,
                                   t->aspect, lv, layer, w, hh, dd, name);
        fprintf(passes_f, "  writes img slot %d level %u layer %u depth %u -> %s.png%s%s\n", bd->res->slot, lv, layer,
                dd, name, why ? " skipped: " : "", why ? why : "");
    }
    if (!pass_skip) flush_luts(ex, n);
    else nluts = 0;
    buf_snap(ex, n);
    fflush(passes_f);
out:
    plat_mutex_unlock(&rec_lock);
}

static const char *tex_desc(const tex_t *t, char *buf, size_t n) {
    if (!t) return "null";
    snprintf(buf, n, "addr %#llx nvn-fmt %#x %ux%ux%u", (unsigned long long)(uintptr_t)t->mem, t->d.format, t->d.width,
             t->d.height, t->d.depth);
    return buf;
}

static void region_desc(const NVNcopyRegion *r, char *buf, size_t n) {
    snprintf(buf, n, "(%d,%d,%d %dx%dx%d)", r->xoffset, r->yoffset, r->zoffset, r->width, r->height, r->depth);
}

void vk_dump_copy(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex;
    plat_mutex_lock(&rec_lock);
    if (!active()) goto out;
    char a[96], b[96], ra[64], rb[64];
    switch (op) {
    case OP_COPY_B2B: {
        const p_copy_b2b *p = payload;
        fprintf(passes_f, "copy after pass %d: buffer %#llx -> buffer %#llx, %llu bytes\n", seq - 1,
                (unsigned long long)p->src, (unsigned long long)p->dst, (unsigned long long)p->size);
        break;
    }
    case OP_COPY_B2T: case OP_COPY_T2B: {
        const p_copy_b2t *p = payload;
        region_desc(&p->r, ra, sizeof ra);
        fprintf(passes_f, "copy after pass %d: %s buffer %#llx %s texture %s region %s flags %#x\n", seq - 1,
                op == OP_COPY_B2T ? "buffer->texture" : "texture->buffer", (unsigned long long)p->src,
                op == OP_COPY_B2T ? "->" : "<-", tex_desc(p->tex, a, sizeof a), ra, p->flags);
        break;
    }
    case OP_COPY_T2T: {
        const p_copy_t2t *p = payload;
        region_desc(&p->sr, ra, sizeof ra);
        region_desc(&p->dr, rb, sizeof rb);
        fprintf(passes_f, "copy after pass %d: texture %s %s -> texture %s %s flags %#x\n", seq - 1,
                tex_desc(p->src, a, sizeof a), ra, tex_desc(p->dst, b, sizeof b), rb, p->flags);
        break;
    }
    case OP_CLEAR_TEX: {
        const p_clear_tex *p = payload;
        region_desc(&p->r, ra, sizeof ra);
        fprintf(passes_f, "clear after pass %d: texture %s region %s mask %#x\n", seq - 1, tex_desc(p->tex, a, sizeof a),
                ra, p->mask);
        break;
    }
    default: break;
    }
out:
    plat_mutex_unlock(&rec_lock);
}

static void make_dirs(char *path) { plat_file_mkdir_p(path); }

// NX2NAVI_DUMP_WHITE=1 (diagnosis): no frame is named; from frame 1800 each present looks at the thumbnail of the frame
// three presents back (gfx_present.c vk_thumbs_*), and the first one whose picture left of the overlay and above the
// dialogue box is blank white makes the dump take the next frame (white flashes come in runs of several frames).
int vk_dump_white;   // vk_backend.c asks for a thumbnail of every present while it is set
#define dump_white vk_dump_white

// passes.txt and images.txt in dir (a new folder per NX2NAVI_DUMP_WHITE attempt)
void rhi_label_write(const char *dir);   // rhi_label.c
static void open_files(void) {
    make_dirs(dir);
    char path[600];
    snprintf(path, sizeof path, "%s/passes.txt", dir);
    passes_f = fopen(path, "w");
    snprintf(path, sizeof path, "%s/images.txt", dir);
    images_f = fopen(path, "w");
    if (!passes_f || !images_f) rt_fatal("vk_dump: cannot write to %s", dir);
}

// (orbitals) NX2NAVI_DUMP_UBO / NX2NAVI_DUMP_TEX: what the dumped frame records. Read once, from vk_dump_init when a frame
// is named and from vk_dump_hotkey for F9 / dump.now dumps (vk_dump_init returns before this when no frame is named, so
// the hotkey path used to dump without them).
static void parse_content_knobs(void) {
    static int done;
    if (done) return;
    done = 1;
    const char *du = cfg_get("NX2NAVI_DUMP_UBO");
    dump_ubo = du && *du ? atoi(du) : 0;
    if (dump_ubo == 1) dump_ubo = 16;
    if (dump_ubo > 4096) dump_ubo = 4096;
    if (dump_ubo < 0) dump_ubo = 0;
    const char *wt = cfg_get("NX2NAVI_DUMP_TEX");
    for (char *q = (char *)wt; q && *q && nwant_tex < 16; q += *q == ',') {
        uint32_t *w = want_tex[nwant_tex++];
        w[0] = (uint32_t)strtoul(q, &q, 10);
        w[1] = *q == 'x' ? (uint32_t)strtoul(q + 1, &q, 10) : 0;
    }
}

void vk_dump_init(void) {
    const char *e = cfg_get("NX2NAVI_DUMP");
    dump_white = rt_knob("DUMP_WHITE", "0");
    // NX2NAVI_DUMP_AFTER=<address>: the first render pass whose depth target lives at that address and the rest of its frame
    // (a target drawn once, at an area load)
    const char *da = cfg_get("NX2NAVI_DUMP_AFTER");
    dump_after = da && *da ? strtoull(da, NULL, 0) : 0;
    if (dump_white || dump_after) {
        dump_frame = UINT64_MAX;
        vk_thumbs_enable(16);
    } else if (!e || !*e || !(dump_frame = strtoull(e, NULL, 10))) return;
    const char *lf = cfg_get("NX2NAVI_DUMP_LOG");
    if (lf && *lf) log_from = strtoull(lf, NULL, 10);
    const char *sr = cfg_get("NX2NAVI_DUMP_SKIP_RUNS");
    skip_runs = sr && *sr && *sr != '0';
    parse_content_knobs();
    const char *d = cfg_get("NX2NAVI_DUMP_DIR");
    snprintf(base_dir, sizeof base_dir, "%s", d && *d ? d : "dump");
    snprintf(dir, sizeof dir, "%s", base_dir);
    char path[600];
    if (!dump_white) open_files();
    const char *bf = cfg_get("NX2NAVI_DUMP_BUF");
    if (bf && !strcmp(bf, "quadlist")) {   // (orbitals) the backdrop quad's c3+0 list, address taken at its first draw
        bd_auto = 1, bd_size = 4096;
        snprintf(path, sizeof path, "%s/bufs.txt", dir);
        if (!(bufs_f = fopen(path, "w"))) bd_size = 0;
    } else if (bf && *bf) {
        char *q;
        bd_addr = strtoull(bf, &q, 0);
        if (*q == ':') bd_size = strtoull(q + 1, &q, 0);
        if (*q == ':') bd_off = strtoull(q + 1, &q, 0);
        if (*q == ':') bd_stride = strtoull(q + 1, &q, 0) & ~3ull;
        if (*q == ':') bd_w = (uint32_t)strtoul(q + 1, &q, 0);
        if (*q == ':') bd_h = (uint32_t)strtoul(q + 1, &q, 0);
        if (!bd_w || !bd_h) bd_stride = 0;
        snprintf(path, sizeof path, "%s/bufs.txt", dir);
        if (bd_size && !(bufs_f = fopen(path, "w"))) bd_size = 0;
    }
    vk_dump_armed = 1;
    rt_log("vk_dump: frame %llu goes to %s", (unsigned long long)dump_frame, dir);
}

// The thumbnail of frame f is blank white left of the overlay and above the dialogue box: -1 not available, 0 no, 1 yes.
static int thumb_white(uint64_t f) {
    static uint8_t px[FLIGHT_THUMB_W * FLIGHT_THUMB_H * 4];
    if (!vk_thumbs_read(f, px)) return -1;
    int n = 0, bright = 0;
    for (int y = 0; y < 105; y++)
        for (int x = 0; x < 250; x++, n++) {
            const uint8_t *q = px + 4 * (y * FLIGHT_THUMB_W + x);
            bright += (q[0] * 30 + q[1] * 59 + q[2] * 11) / 100 > 245;
        }
    return bright * 100 > n * 97;
}

static void white_check(uint64_t frame) {
    static int run;   // consecutive white thumbnails: white runs are mostly 1-2 frames, too short for a dump 4 frames on
    if (frame < 1800) return;
    int w = thumb_white(frame - 3);
    if (w < 0) return;
    run = w ? run + 1 : 0;
    if (run >= 3) {   // frames frame-5..frame-3 white: the run is likely still on 4 frames later
        run = 0;
        dump_frame = frame + 1;
        snprintf(dir, sizeof dir, "%s/f%llu", base_dir, (unsigned long long)dump_frame);
        open_files();
        rt_log("vk_dump: frames %llu-%llu were white; frame %llu goes to %s", (unsigned long long)(frame - 5),
               (unsigned long long)(frame - 3), (unsigned long long)dump_frame, dir);
    }
}

// (orbitals) F9: the frame after next goes to dump/f<frame> (gfx_window_sdl.c); any thread.
// NX2NAVI_DUMP_HOT_N=<n> (default 1): one press dumps n frames, each about 3 frames after the last (a dump stalls to
// write its files), so a flickering object is caught both shown and gone; the NX2NAVI_OQ_WIN log covers the same span.
static volatile int hot_pending, hot_left;
// Each dumped frame's presented image goes to <its dir>/final_f<frame>.png (gfx_present.c asks at present time): saved
// whatever the pass-image cap did, so which frames show an object is always known. Last 16 dumps.
static struct { uint64_t f; char d[600]; } finals[16];
static int nfinals;
static void final_note(void) {
    finals[nfinals % 16].f = dump_frame;
    snprintf(finals[nfinals % 16].d, sizeof finals[0].d, "%s", dir);
    nfinals++;
}
int vk_dump_final_path(uint64_t frame, char *out, size_t cap) {
    for (int i = 0; i < 16 && i < nfinals; i++)
        if (finals[i].f == frame && frame) {
            snprintf(out, cap, "%s/final_f%llu.png", finals[i].d, (unsigned long long)frame);
            finals[i].f = 0;
            return 1;
        }
    return 0;
}
void vk_dump_hotkey(void) {
    {
        const char *n = cfg_get("NX2NAVI_DUMP_HOT_N");
        int k = n && *n ? atoi(n) : 1;
        hot_left = (k < 1 ? 1 : k > 16 ? 16 : k) - 1;
        extern void vk_oq_win_set(uint64_t from, long n);
        vk_oq_win_set(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1, 4L * (hot_left + 1) + 4);
    }
    if (!base_dir[0]) {
        const char *d = cfg_get("NX2NAVI_DUMP_DIR");
        snprintf(base_dir, sizeof base_dir, "%s", d && *d ? d : "dump");
    }
    parse_content_knobs();
    hot_pending = 1;
    vk_dump_armed = 1;
}

void vk_dump_present(uint64_t frame) {
    {   // NX2NAVI_DUMP_HIDE_STALL=1: present gaps over 100 ms within 4 frames of a dump count as stall, less one 30 fps frame
        extern volatile uint64_t rt_tick_hidden;
        extern uint64_t rt_ticks_raw(void);
        static int on = -1;
        static uint64_t prev, near_until;
        if (on < 0) { const char *e = cfg_get("NX2NAVI_DUMP_HIDE_STALL"); on = e && *e == '1'; }
        uint64_t now = rt_ticks_raw();
        if (hot_pending || frame == dump_frame) near_until = frame + 4;
        if (on && prev && frame <= near_until && now - prev > RT_SYSTEM_TICK_HZ / 10u) {
            rt_tick_hidden += now - prev - (RT_SYSTEM_TICK_HZ / 30u);
            rt_log("vk_dump: hid %.3f s of stall from the guest clock at frame %llu", (now - prev - (RT_SYSTEM_TICK_HZ / 30u)) / (double)RT_SYSTEM_TICK_HZ,
                   (unsigned long long)frame);
        }
        prev = now;
    }
    if (hot_pending) {
        hot_pending = 0;
        if (passes_f) fclose(passes_f), passes_f = NULL;
        if (images_f) fclose(images_f), images_f = NULL;
        plat_mutex_lock(&rec_lock);   // a second dump in the same session starts from a clean state
        seq = copied = 0, last_targets = 0, pass_skip = NULL, png_bytes = 0;
        nluts = nluts_done = nents = ents_over = pass_draws = 0;
        finished = 0;
        plat_mutex_unlock(&rec_lock);
        dump_frame = frame + 2;
        snprintf(dir, sizeof dir, "%s/f%llu", base_dir, (unsigned long long)dump_frame);
        open_files();
        rt_log("vk_dump: F9: frame %llu goes to %s", (unsigned long long)dump_frame, dir);
        final_note();
    }
    if (dump_white && white_last && frame == white_last + 3) {   // was the dumped frame itself white?
        int w = thumb_white(white_last);
        rt_log("vk_dump: dumped frame %llu was %s", (unsigned long long)white_last, w > 0 ? "white: done" : "not white: trying again");
        if (w > 0 || ++white_tries >= 8) white_caught = 1;
        white_last = 0;
    }
    if (dump_white && !white_caught && !white_last && dump_frame == UINT64_MAX) white_check(frame);
    if (frame != dump_frame) return;
    plat_mutex_lock(&rec_lock);
    finished = 1;
    plat_mutex_unlock(&rec_lock);
    vk_sync_finish();   // the frame's copies are converted and written
    rhi_label_write(dir);   // NX2NAVI_RHI_LABELS: guest address -> engine name, as of the dumped frame
    plat_mutex_lock(&img_lock);
    fclose(images_f);
    images_f = NULL;
    if (bufs_f) fclose(bufs_f), bufs_f = NULL;
    plat_mutex_unlock(&img_lock);
    fclose(passes_f);
    if (texels_f) fclose(texels_f), texels_f = NULL;
    passes_f = NULL;
    rt_log("vk_dump: frame %llu: %d passes and dispatches, about %llu MB of PNGs in %s", (unsigned long long)frame, seq,
           (unsigned long long)(png_bytes >> 20), dir);
    if (hot_left > 0) hot_left--, hot_pending = 1;   // NX2NAVI_DUMP_HOT_N: the next frame of this press
    if (dump_white) {   // ready for another attempt (vk_dump_present checks this one's thumbnail 3 presents on)
        white_last = frame;
        plat_mutex_lock(&rec_lock);
        seq = copied = 0, last_targets = 0, pass_skip = NULL, png_bytes = 0;
        nluts = nluts_done = nents = ents_over = pass_draws = 0;
        dump_frame = UINT64_MAX;
        finished = 0;
        plat_mutex_unlock(&rec_lock);
    }
}
