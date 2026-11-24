"""Find the game's OodleLZ_Decompress in an extracted main (a flat NSO from make_game.py's exefs step).

Same rule as nxnative's hle_oodle.c: the one function whose code takes the address (adrp + add) of the
NUL-terminated string "OodleLZ_Decompress", with its start from the module's .eh_frame_hdr. The string belongs to
the Oodle library itself, so the rule does not depend on the engine.

Usage: python oodle_find.py <exefs/main>      prints the offset in main, e.g. 0x4f77e10 (exit 1: not found)
"""
import struct
import sys

import numpy as np

NEEDLE = b'OodleLZ_Decompress'


def image(raw):
    """main as loaded: each segment at its memory offset. Returns (image, text_end, ro_off, data_off)."""
    segs = [struct.unpack_from('<III', raw, 0x10 + i * 0x10) for i in range(3)]   # file_off, mem_off, size
    end = max(m + s for _, m, s in segs)
    img = bytearray(end)
    for f, m, s in segs:
        img[m:m + s] = raw[f:f + s]
    return bytes(img), segs[0][1] + segs[0][2], segs[1][1], segs[2][1]


def eh_read(b, o, enc, ehs):
    fmt, app = enc & 0xF, enc & 0x70
    if fmt == 0x3:
        v = struct.unpack_from('<I', b, o)[0]; n = 4
    elif fmt == 0xB:
        v = struct.unpack_from('<i', b, o)[0]; n = 4
    elif fmt == 0x4:
        v = struct.unpack_from('<Q', b, o)[0]; n = 8
    else:
        raise ValueError(f'.eh_frame_hdr encoding {enc:#x} not handled')
    if app == 0x10:
        v += o
    elif app == 0x30:
        v += ehs
    return v & 0xFFFFFFFFFFFFFFFF, o + n


def func_starts(b):
    m0 = struct.unpack_from('<I', b, 4)[0]
    if b[m0:m0 + 4] != b'MOD0':
        raise ValueError('main has no MOD0')
    ehs = m0 + struct.unpack_from('<i', b, m0 + 16)[0]
    penc, cenc, tenc = b[ehs + 1], b[ehs + 2], b[ehs + 3]
    o = ehs + 4
    _, o = eh_read(b, o, penc, ehs)
    n, o = eh_read(b, o, cenc, ehs)
    starts = []
    for _ in range(n):
        s, o = eh_read(b, o, tenc, ehs)
        _, o = eh_read(b, o, tenc, ehs)
        starts.append(s)
    return np.array(sorted(starts), dtype=np.uint64)


def find(path):
    b, text_end, ro, data = image(open(path, 'rb').read())
    strs = set()
    p = ro
    while True:
        p = b.find(NEEDLE, p, data)
        if p < 0:
            break
        if b[p - 1] == 0 and b[p + len(NEEDLE)] == 0:
            strs.add(p)
        p += len(NEEDLE)
    if not strs:
        return None
    w = np.frombuffer(b[:text_end & ~3], dtype='<u4')
    adrp = np.nonzero((w & 0x9F000000) == 0x90000000)[0]
    starts = func_starts(b)
    fns = set()
    for i in adrp:
        x = int(w[i]); pc = i * 4
        imm = ((x >> 29) & 3) | (((x >> 5) & 0x7FFFF) << 2)
        if imm & (1 << 20):
            imm -= 1 << 21
        page = (pc & ~0xFFF) + (imm << 12)
        if not any(page <= s < page + 0x1000 for s in strs):
            continue
        for k in range(1, 8):
            if i + k >= len(w):
                break
            y = int(w[i + k])
            if (y & 0xFF800000) != 0x91000000 or ((y >> 5) & 31) != (x & 31):
                continue
            if page + ((y >> 10) & 0xFFF) in strs:
                j = np.searchsorted(starts, pc + 4 * k, side='right') - 1
                fns.add(int(starts[j]))
            break
    if len(fns) > 1:
        raise ValueError(f'{len(fns)} functions reference "OodleLZ_Decompress" (expected exactly 1): '
                         + ', '.join(hex(f) for f in sorted(fns)))
    return fns.pop() if fns else None


if __name__ == '__main__':
    off = find(sys.argv[1])
    if off is None:
        print('main has no "OodleLZ_Decompress" reference', file=sys.stderr)
        sys.exit(1)
    print(hex(off))
