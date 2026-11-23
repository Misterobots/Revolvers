// Contains code Copyright (c) 2026 MacNativeGames contributors (MIT; see THIRD_PARTY_NOTICES.md).
// The hardware Opus decoder (nn::codec), decoded with libopus on the host. The software keyboard (nn::swkbd) is
// in swkbd_win.c.
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <opus/opus.h>

#include "host.h"
#include "platform/platform.h"

#define CODEC_R_INVALID 0x1fcd1u   // the "invalid argument" result for nn::codec
#define OPUS_WORK_SIZE 0x9000u     // what the game is told to reserve: libopus stereo decoder state (about 26 KiB) plus 4 KiB

// One record per initialised decoder object (the guest's HardwareOpusDecoderType). The registry lock only guards the
// list; a decoder's own lock serialises its libopus state, so different streams decode in parallel. Lock order:
// registry, then decoder (a caller takes the decoder lock before it drops the registry lock, so Finalize cannot free a
// decoder another thread is using).
typedef struct Decoder {
    uint64_t guest;
    OpusDecoder *dec;
    int rate, channels;
    PlatMutex use;
    struct Decoder *next;
} Decoder;
static Decoder *decoders;
static PlatMutex lock = PLAT_MUTEX_INIT;

static int valid_format(int rate, int channels) {
    return (rate == 48000 || rate == 24000 || rate == 16000 || rate == 12000 || rate == 8000) &&
           (channels == 1 || channels == 2);
}

// Returns the decoder with its use lock held.
static Decoder *acquire(uint64_t guest, const char *fn) {
    plat_mutex_lock_shared(&lock);
    Decoder *d = decoders;
    while (d && d->guest != guest) d = d->next;
    if (d) plat_mutex_lock(&d->use);
    plat_mutex_unlock_shared(&lock);
    if (!d) rt_fatal("nn::codec::%s: unknown hardware Opus decoder %#llx", fn, (unsigned long long)guest);
    return d;
}

static void release(Decoder *d) { plat_mutex_unlock(&d->use); }

static OpusDecoder *opus_new(int rate, int channels) {
    int err = 0;
    OpusDecoder *o = opus_decoder_create(rate, channels, &err);
    if (!o || err != OPUS_OK) rt_fatal("nn::codec: opus_decoder_create(%d, %d): %s", rate, channels, opus_strerror(err));
    return o;
}

// size_t GetHardwareOpusDecoderWorkBufferSize(int rate, int channels, int option)
static void ov_work_size(Cpu *c) {
    ret_x(c, valid_format((int)arg_x(c, 0), (int)arg_x(c, 1)) ? OPUS_WORK_SIZE : 0);
}

// Result InitializeHardwareOpusDecoder(HardwareOpusDecoderType *d, int rate, int channels, void *work, size_t size,
// int option). The state lives in libopus on the host; the guest's work buffer is reserved but unused. Initialising
// the same object again without Finalize replaces its decoder.
static void ov_initialize(Cpu *c) {
    uint64_t guest = arg_x(c, 0), size = arg_x(c, 4);
    int rate = (int)arg_x(c, 1), channels = (int)arg_x(c, 2);
    if (!guest || !valid_format(rate, channels) || size < OPUS_WORK_SIZE) { ret_x(c, CODEC_R_INVALID); return; }
    OpusDecoder *fresh = opus_new(rate, channels);
    plat_mutex_lock(&lock);
    Decoder *d = decoders;
    while (d && d->guest != guest) d = d->next;
    if (d) {
        plat_mutex_lock(&d->use);   // waits for a decode in flight
        OpusDecoder *old = d->dec;
        d->dec = fresh;
        d->rate = rate;
        d->channels = channels;
        plat_mutex_unlock(&d->use);
        plat_mutex_unlock(&lock);
        opus_decoder_destroy(old);
    } else {
        d = calloc(1, sizeof *d);
        if (!d) rt_fatal("nn::codec: decoder allocation");
        plat_mutex_init(&d->use);
        d->guest = guest;
        d->dec = fresh;
        d->rate = rate;
        d->channels = channels;
        d->next = decoders;
        decoders = d;
        plat_mutex_unlock(&lock);
    }
    ret_x(c, 0);
}

// void FinalizeHardwareOpusDecoder(HardwareOpusDecoderType *d)
static void ov_finalize(Cpu *c) {
    uint64_t guest = arg_x(c, 0);
    plat_mutex_lock(&lock);
    Decoder **p = &decoders;
    while (*p && (*p)->guest != guest) p = &(*p)->next;
    if (!*p) rt_fatal("nn::codec::FinalizeHardwareOpusDecoder: unknown decoder %#llx", (unsigned long long)guest);
    Decoder *d = *p;
    plat_mutex_lock(&d->use);   // a decode in flight finishes first; the registry lock keeps new ones out
    *p = d->next;
    plat_mutex_unlock(&lock);
    opus_decoder_destroy(d->dec);
    free(d);
}

// int GetHardwareOpusDecoderSampleRate / GetHardwareOpusDecoderChannelCount(const HardwareOpusDecoderType *d)
static void ov_sample_rate(Cpu *c) {
    Decoder *d = acquire(arg_x(c, 0), "GetHardwareOpusDecoderSampleRate");
    int v = d->rate;
    release(d);
    ret_x(c, (uint64_t)v);
}

static void ov_channel_count(Cpu *c) {
    Decoder *d = acquire(arg_x(c, 0), "GetHardwareOpusDecoderChannelCount");
    int v = d->channels;
    release(d);
    ret_x(c, (uint64_t)v);
}

// Result ResetHardwareOpusDecoder(HardwareOpusDecoderType *d)
static void ov_reset(Cpu *c) {
    Decoder *d = acquire(arg_x(c, 0), "ResetHardwareOpusDecoder");
    int ok = opus_decoder_ctl(d->dec, OPUS_RESET_STATE);
    release(d);
    ret_x(c, ok == OPUS_OK ? 0 : CODEC_R_INVALID);
}

static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

// Result DecodeOpusInterleavedWithHardware(HardwareOpusDecoderType *d, size_t *consumed, int *samples, int16_t *out,
// size_t out_size, const void *in, size_t in_size). The input is Nintendo's frame: an 8-byte big-endian header
// {payload size, final range} and the Opus packet. A header range of 0 is not a recorded value (some encoders write 0
// into every packet), so it is not checked. A range mismatch is logged once and the samples are kept.
static void ov_decode(Cpu *c) {
    static int32_t warned;
    uint64_t guest = arg_x(c, 0), out_size = arg_x(c, 4), in_size = arg_x(c, 6);
    size_t *consumed = arg_p(c, 1);
    int *samples = arg_p(c, 2);
    int16_t *out = arg_p(c, 3);
    const uint8_t *in = arg_p(c, 5);
    if (!consumed || !samples || !out || !in || in_size < 8) { ret_x(c, CODEC_R_INVALID); return; }
    uint32_t n = be32(in), range = be32(in + 4);
    if (n > in_size - 8 || n > INT_MAX) { ret_x(c, CODEC_R_INVALID); return; }
    Decoder *d = acquire(guest, "DecodeOpusInterleavedWithHardware");
    size_t capacity = out_size / (sizeof(int16_t) * (size_t)d->channels);
    if (capacity > INT_MAX) { release(d); ret_x(c, CODEC_R_INVALID); return; }
    int got = opus_decode(d->dec, in + 8, (opus_int32)n, out, (int)capacity, 0);
    opus_uint32 final = 0;
    if (got >= 0) opus_decoder_ctl(d->dec, OPUS_GET_FINAL_RANGE(&final));
    release(d);
    if (got < 0) {
        rt_log("nn::codec: opus_decode: %s", opus_strerror(got));
        ret_x(c, CODEC_R_INVALID);
        return;
    }
    if (n && range && final != range && !plat_xchg32(&warned, 1))
        rt_log("nn::codec: Opus final range %#x != stream %#x (logged once)", (unsigned)final, (unsigned)range);
    *consumed = 8 + n;
    *samples = got;
    ret_x(c, 0);
}

HOST_OVERRIDE("_ZN2nn5codec36GetHardwareOpusDecoderWorkBufferSizeEiii", ov_work_size);
HOST_OVERRIDE("_ZN2nn5codec29InitializeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeEiiPvmi", ov_initialize);
HOST_OVERRIDE("_ZN2nn5codec33DecodeOpusInterleavedWithHardwareEPNS0_23HardwareOpusDecoderTypeEPmPiPsmPKvm", ov_decode);

// ---- The Switch 2 queued hardware Opus API. A manager owns a completion event; each decoder takes Opus packets with
// Enqueue, the manager decodes everything pending with Execute and signals its event, and the game collects PCM with
// Dequeue, one block per enqueued packet. Decoding happens in Execute on the calling thread with libopus; the guest
// work buffers are reserved but unused. Signatures follow the SDK's mangled names; argument meanings the names do
// not settle (the manager's int/TimeSpan parameters) are ignored.
#define QOPUS_WORK_SIZE 0x10000u
#define QOPUS_IO_BUFFERS 64        // reported free I/O buffers: requests a manager may have in flight
#define QOPUS_MAX_FRAME 5760       // 120 ms at 48 kHz, the longest Opus packet

typedef struct QBlock { struct QBlock *next; int samples; int16_t pcm[]; } QBlock;
typedef struct QPacket { struct QPacket *next; size_t size; uint8_t data[]; } QPacket;
typedef struct QDecoder {
    uint64_t guest, manager;
    OpusDecoder *dec;
    int rate, channels;
    QPacket *in_head, *in_tail;    // enqueued, waiting for Execute
    QBlock *out_head, *out_tail;   // decoded, waiting for Dequeue
    int waiting, completed;
    struct QDecoder *next;
} QDecoder;
typedef struct QManager { uint64_t guest, event; struct QManager *next; } QManager;
static QDecoder *qdecoders;
static QManager *qmanagers;
static PlatMutex qlock = PLAT_MUTEX_INIT;   // one lock: Execute walks every decoder of a manager

static QDecoder *qfind(uint64_t guest) {
    QDecoder *d = qdecoders;
    while (d && d->guest != guest) d = d->next;
    return d;
}
static QManager *qmanager(uint64_t guest) {
    QManager *m = qmanagers;
    while (m && m->guest != guest) m = m->next;
    return m;
}
static void qdrain(QDecoder *d) {
    while (d->in_head) { QPacket *p = d->in_head; d->in_head = p->next; free(p); }
    while (d->out_head) { QBlock *b = d->out_head; d->out_head = b->next; free(b); }
    d->in_tail = NULL;
    d->out_tail = NULL;
    d->waiting = d->completed = 0;
}

// size_t GetHardwareOpusDecoderManagerWorkBufferSize(int, int, TimeSpan, int, int)
static void ov_qmgr_work_size(Cpu *c) { ret_x(c, QOPUS_WORK_SIZE); }

// Result InitializeHardwareOpusDecoderManager(Manager *m, os::SystemEventType *ev, int, int, TimeSpan, int, int,
// void *work, size_t size): ev becomes the manager's completion event, signalled by each Execute.
static void ov_qmgr_init(Cpu *c) {
    uint64_t guest = arg_x(c, 0), ev = arg_x(c, 1);
    if (!guest) { ret_x(c, CODEC_R_INVALID); return; }
    if (ev) kern_bind_system_event(c, ev, 1);
    plat_mutex_lock(&qlock);
    QManager *m = qmanager(guest);
    if (!m) {
        m = calloc(1, sizeof *m);
        if (!m) rt_fatal("nn::codec: manager allocation");
        m->guest = guest;
        m->next = qmanagers;
        qmanagers = m;
    }
    m->event = ev;
    plat_mutex_unlock(&qlock);
    ret_x(c, 0);
}

// void FinalizeHardwareOpusDecoderManager(Manager *m)
static void ov_qmgr_fini(Cpu *c) {
    uint64_t guest = arg_x(c, 0);
    plat_mutex_lock(&qlock);
    for (QManager **p = &qmanagers; *p; p = &(*p)->next)
        if ((*p)->guest == guest) { QManager *m = *p; *p = m->next; free(m); break; }
    plat_mutex_unlock(&qlock);
}

// int GetHardwareOpusDecoderManagerFreeIoBufferCount(const Manager *m)
static void ov_qmgr_free_io(Cpu *c) {
    uint64_t guest = arg_x(c, 0);
    int used = 0;
    plat_mutex_lock(&qlock);
    for (QDecoder *d = qdecoders; d; d = d->next)
        if (d->manager == guest) used += d->waiting + d->completed;
    plat_mutex_unlock(&qlock);
    ret_x(c, (uint64_t)(used >= QOPUS_IO_BUFFERS ? 0 : QOPUS_IO_BUFFERS - used));
}

// Result InitializeHardwareOpusDecoder(HardwareOpusDecoderType *d, int rate, int channels, TimeSpan, Manager *m)
static void ov_qdec_init(Cpu *c) {
    uint64_t guest = arg_x(c, 0), manager = arg_x(c, 4);
    int rate = (int)arg_x(c, 1), channels = (int)arg_x(c, 2);
    if (!guest || !valid_format(rate, channels)) { ret_x(c, CODEC_R_INVALID); return; }
    OpusDecoder *fresh = opus_new(rate, channels);
    plat_mutex_lock(&qlock);
    QDecoder *d = qfind(guest);
    if (d) {
        opus_decoder_destroy(d->dec);
        qdrain(d);
    } else {
        d = calloc(1, sizeof *d);
        if (!d) rt_fatal("nn::codec: decoder allocation");
        d->guest = guest;
        d->next = qdecoders;
        qdecoders = d;
    }
    d->dec = fresh;
    d->rate = rate;
    d->channels = channels;
    d->manager = manager;
    plat_mutex_unlock(&qlock);
    ret_x(c, 0);
}

// The queued decoder shares Finalize/Reset/SampleRate/ChannelCount with the one-shot decoder (same entry points, same
// HardwareOpusDecoderType). These wrappers try the queued registry first and fall back to the one-shot handlers.
static void ov_any_finalize(Cpu *c) {
    uint64_t guest = arg_x(c, 0);
    plat_mutex_lock(&qlock);
    for (QDecoder **p = &qdecoders; *p; p = &(*p)->next)
        if ((*p)->guest == guest) {
            QDecoder *d = *p;
            *p = d->next;
            plat_mutex_unlock(&qlock);
            opus_decoder_destroy(d->dec);
            qdrain(d);
            free(d);
            return;
        }
    plat_mutex_unlock(&qlock);
    ov_finalize(c);
}
static void ov_any_reset(Cpu *c) {
    plat_mutex_lock(&qlock);
    QDecoder *d = qfind(arg_x(c, 0));
    if (d) {
        int ok = opus_decoder_ctl(d->dec, OPUS_RESET_STATE);
        qdrain(d);
        plat_mutex_unlock(&qlock);
        ret_x(c, ok == OPUS_OK ? 0 : CODEC_R_INVALID);
        return;
    }
    plat_mutex_unlock(&qlock);
    ov_reset(c);
}
static void ov_any_rate(Cpu *c) {
    plat_mutex_lock(&qlock);
    QDecoder *d = qfind(arg_x(c, 0));
    int v = d ? d->rate : -1;
    plat_mutex_unlock(&qlock);
    if (v < 0) ov_sample_rate(c); else ret_x(c, (uint64_t)v);
}
static void ov_any_channels(Cpu *c) {
    plat_mutex_lock(&qlock);
    QDecoder *d = qfind(arg_x(c, 0));
    int v = d ? d->channels : -1;
    plat_mutex_unlock(&qlock);
    if (v < 0) ov_channel_count(c); else ret_x(c, (uint64_t)v);
}

// Result EnqueueOpusDataToHardwareOpusDecoder(HardwareOpusDecoderType *d, size_t *consumed, const void *in, size_t n).
// Nintendo's framed packet (8-byte big-endian {payload size, final range} header) is taken one frame at a time, as
// the one-shot decoder does. Input whose header does not describe a payload that fits is taken as one raw Opus
// packet: the SDK's own framing for this call is not confirmed, so both forms are accepted.
static void ov_qdec_enqueue(Cpu *c) {
    uint64_t guest = arg_x(c, 0), n = arg_x(c, 3);
    size_t *consumed = arg_p(c, 1);
    const uint8_t *in = arg_p(c, 2);
    if (!in || !n) { ret_x(c, CODEC_R_INVALID); return; }
    size_t off = 0, len = n, used = n;
    if (n >= 8) {
        uint32_t payload = be32(in);
        if (payload && payload <= n - 8) { off = 8; len = payload; used = 8 + (size_t)payload; }
    }
    QPacket *p = malloc(sizeof *p + len);
    if (!p) rt_fatal("nn::codec: packet allocation");
    p->next = NULL;
    p->size = len;
    memcpy(p->data, in + off, len);
    plat_mutex_lock(&qlock);
    QDecoder *d = qfind(guest);
    if (!d) {
        plat_mutex_unlock(&qlock);
        free(p);
        rt_fatal("nn::codec::EnqueueOpusDataToHardwareOpusDecoder: unknown decoder %#llx", (unsigned long long)guest);
    }
    if (d->in_tail) d->in_tail->next = p; else d->in_head = p;
    d->in_tail = p;
    d->waiting++;
    plat_mutex_unlock(&qlock);
    if (consumed) *consumed = used;
    ret_x(c, 0);
}

// Result ExecuteHardwareOpusDecoderDecoding(Manager *m): decode every pending packet of the manager's decoders, then
// signal the manager's event. A packet libopus rejects becomes one 20 ms block of silence so the game's request count
// stays in step; the error is logged once.
static void ov_qmgr_execute(Cpu *c) {
    static int32_t warned;
    uint64_t guest = arg_x(c, 0), ev = 0;
    plat_mutex_lock(&qlock);
    QManager *m = qmanager(guest);
    if (m) ev = m->event;
    for (QDecoder *d = qdecoders; d; d = d->next) {
        if (d->manager != guest) continue;
        while (d->in_head) {
            QPacket *p = d->in_head;
            d->in_head = p->next;
            if (!d->in_head) d->in_tail = NULL;
            QBlock *b = malloc(sizeof *b + sizeof(int16_t) * QOPUS_MAX_FRAME * (size_t)d->channels);
            if (!b) rt_fatal("nn::codec: PCM allocation");
            b->next = NULL;
            int got = opus_decode(d->dec, p->data, (opus_int32)p->size, b->pcm, QOPUS_MAX_FRAME, 0);
            if (got < 0) {
                if (!plat_xchg32(&warned, 1)) rt_log("nn::codec: queued opus_decode: %s (logged once)", opus_strerror(got));
                got = d->rate / 50;
                memset(b->pcm, 0, sizeof(int16_t) * (size_t)got * (size_t)d->channels);
            }
            b->samples = got;
            free(p);
            if (d->out_tail) d->out_tail->next = b; else d->out_head = b;
            d->out_tail = b;
            d->waiting--;
            d->completed++;
        }
    }
    plat_mutex_unlock(&qlock);
    if (ev) kern_signal_system_event(ev);
    ret_x(c, 0);
}

// Result DequeuePcmInterleavedDataFromHardwareOpusDecoder(HardwareOpusDecoderType *d, int *samples, int16_t *out,
// size_t out_size): the oldest decoded block. Nothing completed: 0 samples. A block larger than out is cut to fit.
static void ov_qdec_dequeue(Cpu *c) {
    uint64_t guest = arg_x(c, 0), out_size = arg_x(c, 3);
    int *samples = arg_p(c, 1);
    int16_t *out = arg_p(c, 2);
    if (!samples) { ret_x(c, CODEC_R_INVALID); return; }
    plat_mutex_lock(&qlock);
    QDecoder *d = qfind(guest);
    QBlock *b = d ? d->out_head : NULL;
    if (b) {
        d->out_head = b->next;
        if (!d->out_head) d->out_tail = NULL;
        d->completed--;
    }
    int ch = d ? d->channels : 1;
    plat_mutex_unlock(&qlock);
    if (!b) { *samples = 0; ret_x(c, 0); return; }
    size_t cap = out ? out_size / (sizeof(int16_t) * (size_t)ch) : 0;
    int n = (size_t)b->samples < cap ? b->samples : (int)cap;
    if (n > 0) memcpy(out, b->pcm, sizeof(int16_t) * (size_t)n * (size_t)ch);
    *samples = n;
    free(b);
    ret_x(c, 0);
}

static int qcount(uint64_t guest, int completed) {
    plat_mutex_lock(&qlock);
    QDecoder *d = qfind(guest);
    int v = !d ? 0 : completed ? d->completed : d->waiting;
    plat_mutex_unlock(&qlock);
    return v;
}
static void ov_qdec_waiting(Cpu *c) { ret_x(c, (uint64_t)qcount(arg_x(c, 0), 0)); }
static void ov_qdec_executing(Cpu *c) { ret_x(c, 0); }   // Execute decodes synchronously: nothing is ever in flight
static void ov_qdec_completed(Cpu *c) { ret_x(c, (uint64_t)qcount(arg_x(c, 0), 1)); }

HOST_OVERRIDE("_ZN2nn5codec43GetHardwareOpusDecoderManagerWorkBufferSizeEiiNS_8TimeSpanEii", ov_qmgr_work_size);
HOST_OVERRIDE("_ZN2nn5codec36InitializeHardwareOpusDecoderManagerEPNS0_30HardwareOpusDecoderManagerTypeEPNS_2os15SystemEventTypeEiiNS_8TimeSpanEiiPvm",
              ov_qmgr_init);
HOST_OVERRIDE("_ZN2nn5codec34FinalizeHardwareOpusDecoderManagerEPNS0_30HardwareOpusDecoderManagerTypeE", ov_qmgr_fini);
HOST_OVERRIDE("_ZN2nn5codec46GetHardwareOpusDecoderManagerFreeIoBufferCountEPKNS0_30HardwareOpusDecoderManagerTypeE",
              ov_qmgr_free_io);
HOST_OVERRIDE("_ZN2nn5codec34ExecuteHardwareOpusDecoderDecodingEPNS0_30HardwareOpusDecoderManagerTypeE", ov_qmgr_execute);
HOST_OVERRIDE("_ZN2nn5codec29InitializeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeEiiNS_8TimeSpanEPNS0_30HardwareOpusDecoderManagerTypeE",
              ov_qdec_init);
HOST_OVERRIDE("_ZN2nn5codec36EnqueueOpusDataToHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeEPmPKvm", ov_qdec_enqueue);
HOST_OVERRIDE("_ZN2nn5codec48DequeuePcmInterleavedDataFromHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeEPiPsm",
              ov_qdec_dequeue);
HOST_OVERRIDE("_ZN2nn5codec47GetHardwareOpusDecoderWaitingDecodeRequestCountEPKNS0_23HardwareOpusDecoderTypeE",
              ov_qdec_waiting);
HOST_OVERRIDE("_ZN2nn5codec49GetHardwareOpusDecoderExecutingDecodeRequestCountEPKNS0_23HardwareOpusDecoderTypeE",
              ov_qdec_executing);
HOST_OVERRIDE("_ZN2nn5codec49GetHardwareOpusDecoderCompletedDecodeRequestCountEPKNS0_23HardwareOpusDecoderTypeE",
              ov_qdec_completed);

HOST_OVERRIDE("_ZN2nn5codec27FinalizeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE", ov_any_finalize);
HOST_OVERRIDE("_ZN2nn5codec32GetHardwareOpusDecoderSampleRateEPKNS0_23HardwareOpusDecoderTypeE", ov_any_rate);
HOST_OVERRIDE("_ZN2nn5codec34GetHardwareOpusDecoderChannelCountEPKNS0_23HardwareOpusDecoderTypeE", ov_any_channels);
HOST_OVERRIDE("_ZN2nn5codec24ResetHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE", ov_any_reset);
