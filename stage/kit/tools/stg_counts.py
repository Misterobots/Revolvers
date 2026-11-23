# Niagara GPU instance-count readbacks from logs/rhi_staging.log (NX2NAVI_RHI_STAGING builds).
#   python stg_counts.py <rhi_staging.log> <first frame> <last frame> [slot ...]
# Prints, for each count readback (the small, variable-size ones) in the frame range, the value of every slot, or of
# the slots given. Without slots, prints only slots that are non-zero somewhere in the range AND read 0 or 1 in some
# readback (the ones that could make a ribbon skip).
import re, sys
log, f0, f1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
want = [int(s) for s in sys.argv[4:]]
rows = []
for l in open(log, errors='replace'):
    m = re.match(r'f(\d+) s32 .* size (\d+) -> \w+:(.*)', l)
    if not m: continue
    fr, size = int(m.group(1)), int(m.group(2))
    if not (f0 <= fr <= f1) or size > 4096: continue
    rows.append((fr, size, [int(x) for x in m.group(3).split()]))
if not rows: sys.exit('no count readbacks in that range')
n = max(len(r[2]) for r in rows)
slots = want or [i for i in range(n) if any(i < len(r[2]) and r[2][i] > 1 for r in rows)
                 and any(i < len(r[2]) and r[2][i] < 2 for r in rows)]
print('frame size  ' + ' '.join(f'{i:>5}' for i in slots))
for fr, size, v in rows:
    print(f'{fr:5d} {size:4d}  ' + ' '.join(f'{v[i] if i < len(v) else "-":>5}' for i in slots))
