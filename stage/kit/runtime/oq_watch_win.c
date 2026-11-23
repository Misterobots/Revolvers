// (orbitals, diagnostic) NX2NAVI_OQ_READS=1: who reads the game's occlusion (samples-passed) report records, and what
// they hold at that moment. Each record the replay reports into (vk_exec_xfer.c oq_report) gets a generation per
// report; its 4 KiB page is guarded (PAGE_GUARD), so the first access by any thread raises an exception here. For the
// first read of each generation by a thread other than the port's own completion write, one "oqw read" line gives the
// generation, whether the port had already written that generation's GPU result, the 16 record bytes the reader
// sees (count, timestamp), the host instruction (exe-relative: nm orbitals.exe names the lifted f_<guest address>)
// and the guest thread's x30. "oqw result" lines give each generation's completed count. The guard is re-armed after a
// single step. Bounded: 64 record addresses, 4000 lines.
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "host.h"

#define OQW_N 64
typedef struct { uint64_t addr, gen, seq, frame, count; int done, reads, writes; } OqwRec;
static OqwRec R[OQW_N];
static volatile LONG nrec;
static uint64_t pages[OQW_N];
static DWORD page_prot[OQW_N];
static volatile LONG npages;
static volatile LONG lines;
static SRWLOCK lk = SRWLOCK_INIT;
static _Thread_local uint64_t tl_rearm;
static _Thread_local int tl_port;
static uint64_t exe_base;

static int oqw_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("OQ_READS", "0") != 0;
    return on;
}

static int page_index(uint64_t pg) {
    for (LONG i = 0; i < npages; i++)
        if (pages[i] == pg) return (int)i;
    return -1;
}

static void log_line(const char *s) {
    if (InterlockedIncrement(&lines) <= 4000) rt_log("%s", s);
}

static LONG CALLBACK oqw_veh(EXCEPTION_POINTERS *ep) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code == STATUS_GUARD_PAGE_VIOLATION) {
        uint64_t a = ep->ExceptionRecord->ExceptionInformation[1], pg = a & ~4095ull;
        int wr = (int)ep->ExceptionRecord->ExceptionInformation[0];
        int pi = page_index(pg);
        if (pi < 0) return EXCEPTION_CONTINUE_SEARCH;   // another guard page (thread stacks)
        if (!tl_port) {
            AcquireSRWLockExclusive(&lk);
            for (LONG i = 0; i < nrec; i++) {
                OqwRec *r = &R[i];
                if (a < r->addr || a >= r->addr + 16) continue;
                if (!r->gen) break;
                if (wr ? r->writes++ == 0 : r->reads++ == 0) {
                    uint64_t v[2];
                    memcpy(v, (const void *)(uintptr_t)r->addr, 16);   // the guard is off until re-armed
                    Cpu *c = kern_cpu();
                    char b[400];
                    snprintf(b, sizeof b, "oqw %s: record %#llx gen %llu (reported frame %llu) by host +%#llx guest x30 %#llx tid %lu: "
                             "port result %s, record holds count %llu ts %#llx",
                             wr ? "write" : "read", (unsigned long long)r->addr, (unsigned long long)r->gen,
                             (unsigned long long)r->frame,
                             (unsigned long long)((uint64_t)ep->ContextRecord->Rip - exe_base),
                             c ? (unsigned long long)c->x[30] : 0ull, (unsigned long)GetCurrentThreadId(),
                             r->done ? "already written" : "NOT yet written", (unsigned long long)v[0], (unsigned long long)v[1]);
                    ReleaseSRWLockExclusive(&lk);
                    log_line(b);
                    goto rearm;
                }
                break;
            }
            ReleaseSRWLockExclusive(&lk);
        }
    rearm:
        tl_rearm = pg | 1;
        ep->ContextRecord->EFlags |= 0x100;   // single step, then re-arm the guard
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (code == EXCEPTION_SINGLE_STEP && tl_rearm) {
        uint64_t pg = tl_rearm & ~1ull;
        int pi = page_index(pg);
        DWORD o;
        if (pi >= 0) VirtualProtect((void *)(uintptr_t)pg, 4096, page_prot[pi] | PAGE_GUARD, &o);
        tl_rearm = 0;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

// The replay reported samples passed into addr (report sequence seq) in frame `frame`.
void oqw_report(uint64_t addr, uint64_t seq, uint64_t frame) {
    if (!oqw_on() || !addr) return;
    static int init;
    if (!init) {
        init = 1;
        exe_base = (uint64_t)(uintptr_t)GetModuleHandleW(NULL);
        AddVectoredExceptionHandler(1, oqw_veh);
        rt_log("oqw: watching occlusion report reads (exe base %#llx)", (unsigned long long)exe_base);
    }
    AcquireSRWLockExclusive(&lk);
    OqwRec *r = NULL;
    for (LONG i = 0; i < nrec; i++)
        if (R[i].addr == addr) { r = &R[i]; break; }
    if (!r && nrec < OQW_N) r = &R[nrec], r->addr = addr, nrec++;
    if (r) {
        r->gen++, r->seq = seq, r->frame = frame, r->done = 0, r->reads = 0, r->writes = 0, r->count = 0;
        uint64_t pg = addr & ~4095ull;
        if (page_index(pg) < 0 && npages < OQW_N) {
            MEMORY_BASIC_INFORMATION mb;
            if (VirtualQuery((void *)(uintptr_t)pg, &mb, sizeof mb) && mb.State == MEM_COMMIT) {
                DWORD prot = mb.Protect & ~(DWORD)(PAGE_GUARD | PAGE_NOCACHE | PAGE_WRITECOMBINE), o;
                pages[npages] = pg, page_prot[npages] = prot;
                MemoryBarrier();
                npages++;
                VirtualProtect((void *)(uintptr_t)pg, 4096, prot | PAGE_GUARD, &o);
            }
        }
    }
    ReleaseSRWLockExclusive(&lk);
}

// Around the port's own write of a report record (completion thread): not a game read.
void oqw_port_write(int begin) { tl_port = begin; }

// The completed count of report seq to addr is in memory now.
void oqw_result(uint64_t addr, uint64_t seq, uint64_t count) {
    if (!oqw_on() || !addr) return;
    AcquireSRWLockExclusive(&lk);
    for (LONG i = 0; i < nrec; i++) {
        OqwRec *r = &R[i];
        if (r->addr != addr || r->seq != seq) continue;
        r->done = 1, r->count = count;
        char b[200];
        snprintf(b, sizeof b, "oqw result: record %#llx gen %llu count %llu (%s)", (unsigned long long)addr,
                 (unsigned long long)r->gen, (unsigned long long)count, r->reads ? "already read before this" : "not read yet");
        ReleaseSRWLockExclusive(&lk);
        log_line(b);
        return;
    }
    ReleaseSRWLockExclusive(&lk);
}
