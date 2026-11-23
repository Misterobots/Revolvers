# Offline check of one dumped texture (vk_dump with NX2NAVI_DUMP_TEX): guest storage vs the GPU image, per mip level.
#   python compare_tex.py <dump frame dir> <texture slot id>
# Inputs (written by the dump): stor_tex<id>_*_layout.txt / _storage.bin / _lv<n>_linear.bin (vk_texture_dump_storage)
# and pass_*_tex<id>_*_blocks.bin, pass_*_tex<id>_*_lv<n>_blocks.bin (raw blocks of the image, each mip).
# Steps per level:
#   1. de-tile the raw storage independently (Tegra block-linear: 64-byte x 8-row GOBs, 2^tileH GOBs per block) and
#      compare with the port's linear blocks (nvn_deswizzle): checks upload content and level placement;
#   2. decode the storage blocks (ASTC or BC, texture2ddecoder) and the image's BC3 blocks to RGBA8 and compare per
#      channel, alpha included: mean/max abs error and the share of texels off by more than 16;
#   3. for a mismatch, also try the neighbouring levels and channel permutations, to tell misplacement or channel
#      interpretation from ordinary compression loss.
import glob, os, re, sys
import numpy as np
import texture2ddecoder as td

d, tid = sys.argv[1], sys.argv[2]
lay = glob.glob(os.path.join(d, f'stor_tex{tid}_*_layout.txt'))
if not lay: sys.exit(f'no storage capture for tex {tid} in {d}')
pre = lay[0][:-len('_layout.txt')]
L = open(lay[0]).read()
print(L)
g = lambda k: re.search(k, L)
fmt = int(g(r'nvn_fmt (0x[0-9a-f]+|\d+)').group(1), 0)
W, H = map(int, g(r' (\d+)x(\d+)x\d+ levels').groups())
bw, bh, bpb = map(int, g(r'block (\d+)x(\d+) bpb (\d+)').groups())
astc = 0x79 <= fmt <= 0x94
levels = {int(m.group(1)): dict(rb=int(m.group(2)), rows=int(m.group(3)), base=int(m.group(4)), th=int(m.group(5)))
          for m in re.finditer(r'level (\d+): row_bytes (\d+) rows (\d+) slices \d+ base (\d+) tileH (\d+)', L)}
storage = open(pre + '_storage.bin', 'rb').read()

def detile(buf, base, rb, rows, th):   # independent block-linear -> linear (block rows of rb bytes)
    bhg = 1 << th
    gpr = (rb + 63) // 64
    out = bytearray(rb * rows)
    for y in range(rows):
        for x in range(rb):
            a = (y // (8 * bhg)) * gpr * bhg * 512 + (x // 64) * bhg * 512 + ((y % (8 * bhg)) // 8) * 512 \
                + ((x % 64) // 32) * 256 + ((y % 8) // 2) * 64 + ((x % 32) // 16) * 32 + (y % 2) * 16 + (x % 16)
            if base + a < len(buf): out[y * rb + x] = buf[base + a]
    return bytes(out)

def decode(blocks, w, h, astc_src):
    if astc_src: raw = td.decode_astc(blocks, w, h, bw, bh)
    else: raw = td.decode_bc3(blocks, w, h)
    return np.frombuffer(raw, np.uint8).reshape(h, w, 4)[..., [2, 1, 0, 3]].astype(int)   # BGRA -> RGBA

def img_blocks(lv):
    pat = f'pass_*_tex{tid}_*_blocks.bin' if lv == 0 else f'pass_*_tex{tid}_*_lv{lv}_blocks.bin'
    fs = [f for f in glob.glob(os.path.join(d, pat)) if (lv != 0 or '_lv' not in os.path.basename(f))]
    return open(fs[0], 'rb').read() if fs else None

def stats(a, b):
    e = np.abs(a - b)
    return ' '.join(f'{c}:mean {e[..., i].mean():5.1f} max {e[..., i].max():3d} >16 {100 * (e[..., i] > 16).mean():5.1f}%'
                    for i, c in enumerate('RGBA'))

src = {}
for lv, gm in sorted(levels.items()):
    w, h = max(W >> lv, 1), max(H >> lv, 1)
    mine = detile(storage, gm['base'], gm['rb'], gm['rows'], gm['th'])
    lp = f'{pre}_lv{lv}_linear.bin'
    port = open(lp, 'rb').read() if os.path.exists(lp) else None
    same = port is not None and port[:len(mine)] == mine
    print(f'level {lv} {w}x{h}: independent de-tile vs port linear blocks: {"identical" if same else "DIFFERENT" if port else "no port copy"}')
    try: src[lv] = decode(mine, w, h, astc)
    except Exception as ex: print('  storage decode failed:', ex); continue
    a = src[lv]
    print(f'  storage decoded: alpha min {a[..., 3].min()} max {a[..., 3].max()} mean {a[..., 3].mean():.1f}')
for lv in sorted(src):
    w, h = max(W >> lv, 1), max(H >> lv, 1)
    ib = img_blocks(lv)
    if ib is None: print(f'level {lv}: no image blocks dumped'); continue
    im = decode(ib, w, h, False)
    print(f'level {lv} image(BC3) vs storage: {stats(im, src[lv])}')
    print(f'  image alpha min {im[..., 3].min()} max {im[..., 3].max()} mean {im[..., 3].mean():.1f}')
    for o in (lv - 1, lv + 1):   # misplacement: the image level against a neighbouring storage level, resized
        if o in src:
            s = src[o]
            r = np.array([[s[min(int(y * s.shape[0] / h), s.shape[0] - 1), min(int(x * s.shape[1] / w), s.shape[1] - 1)]
                           for x in range(w)] for y in range(h)])
            print(f'  vs storage level {o} (resampled): {stats(im, r)}')
    for perm in ([2, 1, 0, 3], [3, 0, 1, 2], [0, 1, 2, 0], [3, 3, 3, 3]):   # channel interpretation
        print(f'  image vs storage channels {perm}: mean abs {np.abs(im - src[lv][..., perm]).mean():.1f}')
