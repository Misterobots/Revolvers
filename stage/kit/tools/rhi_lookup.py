# Engine names for the addresses of one vk_dump frame (NX2NAVI_RHI_LABELS builds).
#   python rhi_lookup.py <dump frame dir> [addr ...]
# Reads <dir>/rhi_labels.txt (guest address -> name, sorted). With addresses: names each. Without: names every vsb_*.bin
# file (address in its name) and sums up the 0x... addresses in passes.txt by name.
# A hit is exact, or inside the nearest entry below when that entry's size is known. Without a size (v1 files, textures)
# the nearest entry below within 1 MiB is given, marked '~' (a guess). v2 files also say when the memory held another
# label within the 3 frames before the dump ('reused'), in which case the name is uncertain.
import bisect, glob, os, re, sys

d = sys.argv[1]
ents, dump_frame = [], 0
for l in open(os.path.join(d, 'rhi_labels.txt')):
    m = re.match(r'# dump frame (\d+)', l)
    if m: dump_frame = int(m.group(1))
    if l.startswith('#') or not l.strip(): continue
    cur, _, prev = l.rstrip('\n').partition(' | prev ')
    t = cur.split(None, 5)
    if len(t) >= 6 and t[4].isdigit():   # v2: addr size kind how frame name
        a, sz, kind, how, fr, name = t
        fr = int(fr)
    else:                                # v1: addr size kind how name
        a, sz, kind, how, name = cur.split(None, 4)
        fr = 0
    pv = None
    if prev:
        p = prev.split(None, 4)
        if len(p) == 5 and p[3].isdigit(): pv = (p[1], int(p[3]), p[4])   # kind, frame, name
    ents.append((int(a, 16), int(sz, 16), kind, how, fr, name, pv))
keys = [e[0] for e in ents]


def look(a):
    i = bisect.bisect_right(keys, a) - 1
    if i < 0: return None
    b, sz, kind, how, fr, name, pv = ents[i]
    if a == b: where = ''
    elif sz and a < b + sz: where = f' +{a - b:#x}'
    elif not sz and a - b < 1 << 20: where = f' +{a - b:#x} ~'
    else: return None
    s = f'{kind} {name}{where} ({how})'
    if pv and dump_frame and pv[1] >= dump_frame - 3: s += f' [reused: was {pv[0]} {pv[2]} in frame {pv[1]}]'
    return s


if len(sys.argv) > 2:
    for s in sys.argv[2:]: print(s, look(int(s, 16)) or '-')
    sys.exit()
print(f'{len(ents)} labelled addresses, dump frame {dump_frame or "?"}')
for f in sorted(glob.glob(os.path.join(d, 'vsb_*.bin'))):
    m = re.search(r'_([0-9a-f]{9,})_([0-9a-f]+)_(cpu|gpu)', os.path.basename(f))
    if m and m.group(3) == 'cpu': print(os.path.basename(f), '->', look(int(m.group(1), 16)) or '-')
p = os.path.join(d, 'passes.txt')
if os.path.exists(p):
    seen = {}
    for a in re.findall(r'0x([0-9a-f]{9,12})\b', open(p, errors='replace').read()):
        a = int(a, 16)
        if a not in seen: seen[a] = look(a)
    hit = {a: v for a, v in seen.items() if v}
    sure = {a: v for a, v in hit.items() if '~' not in v and 'reused' not in v}
    print(f'passes.txt: {len(hit)} of {len(seen)} distinct addresses named, {len(sure)} without a guess or reuse')
    by = {}
    for a, v in sure.items(): by.setdefault(v.split(' +')[0].split(' (')[0], []).append(a)
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])): print(f'  {len(v):5d}  {k}')
