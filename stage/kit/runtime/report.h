// F8 bug reports (report.c): a zip a player can attach to a GitHub issue. The flight recorder (flight.c) keeps the
// last frames as thumbnails; F8 adds a screenshot, the log (user folder paths scrubbed) and system info, packs them into
// <exe dir>/reports/orbitals-report-<time>.zip and writes a shortcut that opens a pre-filled GitHub issue.
#pragma once
#include <stddef.h>
#include <stdint.h>

int report_enabled(void);                          // NX2NAVI_REPORT (default 1)
int report_begin(uint64_t frame);                  // F8 (render thread): 0 when a report is still being saved
void report_frames_dir(char *out, size_t cap);     // where flight.c writes the report's thumbnails
void report_finish(const char *frames_dir, uint64_t frame);   // flight.c's dump thread, once the frames are written

// gfx_present.c (present thread): 1 when the next presented image should be captured; report_shot_take gets it as
// tightly packed RGBA and owns the buffer from then on.
int report_shot_wanted(void);
void report_shot_take(uint8_t *rgba, int w, int h);

// The menu overlay: a one-line status ("Saving bug report...", "Bug report saved: ...") while it is fresh; 0 otherwise.
int report_notice(char *out, size_t cap);

// flight.c helpers report.c uses
int flight_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h);
size_t flight_deflate_raw(const uint8_t *src, size_t n, uint8_t *out);   // out: n * 9 / 8 + 64 bytes
uint32_t flight_crc32(const uint8_t *p, size_t n);
