"""Extract files matching a pattern from Orbitals' Switch2.pak (inside romfs.bin), Oodle via the user's oo2core DLL."""
import ctypes, json, os, re, struct, sys

ROM = r'D:\Games\Orbitals-Deluxe-Build\romfs\romfs.bin'
IDX = r'D:\Games\Orbitals-Deluxe-Build\romfs\index.json'
OUT = sys.argv[1]
PAT = re.compile(sys.argv[2], re.I)

pak = [f for f in json.load(open(IDX))['files'] if f['path'].endswith('Orbitals-Switch2.pak')][0]
base, size = pak['offset'], pak['size']
rom = open(ROM, 'rb')
def rd(off, n):
    rom.seek(base + off); return rom.read(n)

tail = rd(size - 300, 300)
m = tail.find(b'\xe1\x12\x6f\x5a')
version, idx_off, idx_size = struct.unpack_from('<IQQ', tail, m + 4)
methods = [tail[m + 44 + 1 + i * 32: m + 44 + 1 + (i + 1) * 32].split(b'\0')[0].decode() for i in range(5)]
if not methods[0]:
    methods = [tail[m + 44 + i * 32: m + 44 + (i + 1) * 32].split(b'\0')[0].decode() for i in range(5)]
print('pak version', version, 'methods', methods)

class R:
    def __init__(s, b): s.b, s.p = b, 0
    def u32(s): v = struct.unpack_from('<I', s.b, s.p)[0]; s.p += 4; return v
    def i32(s): v = struct.unpack_from('<i', s.b, s.p)[0]; s.p += 4; return v
    def u64(s): v = struct.unpack_from('<Q', s.b, s.p)[0]; s.p += 8; return v
    def raw(s, n): v = s.b[s.p:s.p + n]; s.p += n; return v
    def fstr(s):
        n = s.i32()
        if n == 0: return ''
        if n < 0: v = s.raw(-n * 2).decode('utf-16-le'); return v.rstrip('\0')
        return s.raw(n).decode('latin-1').rstrip('\0')

ix = R(rd(idx_off, idx_size))
mount = ix.fstr(); n = ix.i32(); ix.u64()
if ix.u32(): ix.u64(); ix.u64(); ix.raw(20)
has_fdi = ix.u32()
fdi_off, fdi_size = ix.u64(), ix.u64(); ix.raw(20)
enc_size = ix.u32(); enc = ix.raw(enc_size)
print('mount', mount, 'entries', n)

d = R(rd(fdi_off, fdi_size))
files = []
for _ in range(d.u32()):
    dname = d.fstr()
    for _ in range(d.u32()):
        fname = d.fstr(); eoff = d.i32()
        files.append((mount + dname + fname, eoff))
print('files', len(files))

oo = ctypes.WinDLL(r'D:\Games\oo2core_9_win64.dll')
oo.OodleLZ_Decompress.restype = ctypes.c_int64
def oodle(src, out_len):
    dst = ctypes.create_string_buffer(out_len)
    r = oo.OodleLZ_Decompress(src, ctypes.c_int64(len(src)), dst, ctypes.c_int64(out_len), 1, 0, 0, None, 0, None, None, None, 0, 3)
    if r != out_len: raise RuntimeError('oodle %d != %d' % (r, out_len))
    return dst.raw

def extract(eoff):
    bits = struct.unpack_from('<I', enc, eoff)[0]; p = eoff + 4
    if bits & (1 << 31): off = struct.unpack_from('<I', enc, p)[0]; p += 4
    else: off = struct.unpack_from('<Q', enc, p)[0]; p += 8
    # the full FPakEntry is serialized in front of the data
    h = R(rd(off, 4096))
    h.u64(); csize = h.u64(); usize = h.u64(); meth = h.u32(); h.raw(20)
    blocks = []
    if meth:
        for _ in range(h.u32()): blocks.append((h.u64(), h.u64()))
    h.raw(1); h.u32()
    if not meth: return rd(off + h.p, usize)
    if 'odle' not in methods[meth - 1].lower(): raise RuntimeError('method ' + methods[meth - 1])
    out = b''; left = usize
    bs = struct.unpack_from('<I', rd(off + 8 + 8 + 8 + 4 + 20 + 4 + len(blocks) * 16 + 1, 4))[0]
    for s, e in blocks:
        n = min(bs, left)
        out += oodle(rd(off + s, e - s), n); left -= n
    return out

os.makedirs(OUT, exist_ok=True)
for path, eoff in files:
    if PAT.search(path) and eoff >= 0:
        try:
            data = extract(eoff)
        except Exception as ex:
            print('FAIL', path, ex); continue
        dst = os.path.join(OUT, path.replace('../', '').lstrip('/').replace('/', os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, 'wb').write(data)
        print(len(data), path)

