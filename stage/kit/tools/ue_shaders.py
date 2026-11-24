"""The shaders of an Unreal Engine 5 game's IoStore shader libraries, for the shader preparation (prepare_shaders.py).

Unreal Engine 5 for the Switch 2 (Orbitals: Orbitals/Content/Paks/Orbitals-Switch2.utoc / .ucas, TOC version 8, not
encrypted, Oodle in 64 KB blocks) keeps its shaders in the IoStore container: one ShaderCodeLibrary chunk per library
(chunk type 8: the tables) and one ShaderCode chunk per group of shaders (type 9), each group Oodle-compressed as a whole.
Both layers need the game's own Oodle decoder, which runs in the build's executable (--oodle-serve, runtime/oodle_serve.c;
tools/oodle_find.py finds it in main). The container and library layouts follow nxnative's dm/gfx/tools/uepak.py (Toc,
io_archive), which reads the same format on the Mac.

Each decoded shader is an NVN2 record (SP_GLSL_SAGE-like): a header with the shader's name, u32 code length, the code,
u32 control length, the control section (magic 0x98761234, version 0xc9) and a short tail. It is keyed as the runtime
keys an NVN2 stage (shader_census.nvn2_stages): the control section's first 0xe0 bytes in a zeroed 0x880 one, and the
code up to the end of its c1 constants.

Game-derived output stays in the build directory; nothing here is shipped.

    python tools/ue_shaders.py <build dir> [--romfs DIR] [--exefs DIR] [--exe orbitals.exe]   # census: counts only
"""
import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oodle_find  # noqa: E402
from shader_census import NVN2_CONTROL, NVN2_STAGES, digest  # noqa: E402

TOC_MAGIC, IO_SHADER_LIBRARY = b'-==--==--==--==-', 8
CTL_NVN2 = b'\x34\x12\x76\x98\xc9\0\0\0'


class RomFS:
    """Files of the build's RomFS image (romfs.bin + index.json, make_game.py's romfs step)."""
    def __init__(self, romfs):
        self.dir = Path(romfs)
        self.entries = {r['path']: r for r in json.loads((self.dir / 'index.json').read_text(encoding='utf-8'))['files']}
        self.f = open(self.dir / 'romfs.bin', 'rb')

    def read(self, path, off=0, n=None):
        e = self.entries[path]
        n = e['size'] - off if n is None else n
        if off < 0 or off + n > e['size']:
            raise ValueError(f'{path}: read past the end')
        self.f.seek(e['offset'] + off)
        return self.f.read(n)


class Oodle:
    """The game's OodleLZ_Decompress through the build's executable: decode(compressed, raw length) -> bytes."""
    def __init__(self, build, exe, exefs, romfs):
        at = oodle_find.find(str(Path(exefs) / 'main'))
        if at is None:
            raise ValueError('main has no "OodleLZ_Decompress" reference')
        env = dict(os.environ, NX2NAVI_OODLE_AT=hex(at))
        self.p = subprocess.Popen([str(Path(build) / exe), '--exefs', str(exefs), '--romfs', str(romfs), '--oodle-serve'],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  env=env, cwd=str(build))

    def __call__(self, comp, n):
        self.p.stdin.write(struct.pack('<QQ', len(comp), n) + comp)
        self.p.stdin.flush()
        hdr = self.p.stdout.read(8)
        if len(hdr) != 8:
            raise ValueError(f'the Oodle server stopped (exit {self.p.poll()})')
        got = struct.unpack('<Q', hdr)[0]
        if got != n:
            raise ValueError(f'the game\'s Oodle failed on a {len(comp)} -> {n} byte stream')
        return self.p.stdout.read(got)

    def close(self):
        if self.p.stdin:
            self.p.stdin.close()
        self.p.wait()


class R:
    def __init__(self, b, what):
        self.b, self.o, self.what = b, 0, what

    def take(self, fmt):
        n = struct.calcsize(fmt)
        if self.o + n > len(self.b):
            raise ValueError(f'{self.what}: ends early')
        v = struct.unpack_from(fmt, self.b, self.o)
        self.o += n
        return v if len(v) > 1 else v[0]

    def raw(self, n):
        if self.o + n > len(self.b):
            raise ValueError(f'{self.what}: ends early')
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def array(self, c, n):
        return struct.unpack(f'<{n}{c}', self.raw(struct.calcsize(c) * n))


class Toc:
    """An IoStore container inside the RomFS image (uepak.py Toc): chunk IDs, ranges, compression blocks."""
    def __init__(self, rom, path):
        b = rom.read(path)
        if b[:16] != TOC_MAGIC:
            raise ValueError(f'{path}: not an IoStore TOC')
        self.rom, self.path, self.version = rom, path, b[16]
        hsz, n, nblk, bsz, nmeth, mlen, self.block_size, _, nparts = struct.unpack_from('<9I', b, 20)
        if bsz != 12:
            raise ValueError(f'{path}: compression block entries of {bsz} bytes')
        if b[80] >> 1 & 1:
            raise ValueError(f'{path}: the container is encrypted')
        if nparts > 1:
            raise ValueError(f'{path}: {nparts} partitions are not read')
        nseeds = struct.unpack_from('<I', b, 84)[0] if self.version >= 4 else 0
        nover = struct.unpack_from('<I', b, 96)[0] if self.version >= 5 else 0
        o = hsz
        self.ids = [b[o + 12 * i:o + 12 * i + 12] for i in range(n)]
        o += 12 * n
        self.ranges = [(int.from_bytes(b[o + 10 * i:o + 10 * i + 5], 'big'),
                        int.from_bytes(b[o + 10 * i + 5:o + 10 * i + 10], 'big')) for i in range(n)]
        o += 10 * n + 4 * nseeds + 4 * nover
        self.blocks = []
        for i in range(nblk):
            e = b[o + 12 * i:o + 12 * i + 12]
            self.blocks.append((int.from_bytes(e[:5], 'little'), int.from_bytes(e[5:8], 'little'),
                                int.from_bytes(e[8:11], 'little'), e[11]))
        o += 12 * nblk
        self.methods = [b[o + mlen * i:o + mlen * (i + 1)].rstrip(b'\0').decode('latin1') for i in range(nmeth)]
        self.ucas = path[:-5] + '.ucas'
        self.index = {c: i for i, c in enumerate(self.ids)}

    def of_type(self, t):
        return [i for i, c in enumerate(self.ids) if c[11] == t] if self.version >= 4 else []

    def read(self, i, decode):
        off, ln = self.ranges[i]
        first, out = off // self.block_size, []
        for k in range(first, (off + ln - 1) // self.block_size + 1 if ln else first):
            at, size, raw, m = self.blocks[k]
            d = self.rom.read(self.ucas, at, size)
            meth = self.methods[m - 1] if m else ''
            if not meth:
                out.append(d[:raw])
            elif meth == 'Oodle':
                out.append(decode(d, raw))
            elif meth == 'Zlib':
                import zlib
                out.append(zlib.decompress(d))
            elif meth == 'LZ4':
                import lz4.block
                out.append(lz4.block.decompress(d, uncompressed_size=raw))
            else:
                raise ValueError(f'{self.path}: compression method {meth} is not read')
        s = off - first * self.block_size
        return b''.join(out)[s:s + ln]


def library_shaders(toc, i, decode, chunk):
    """Every shader of one IoStore shader library (FIoStoreShaderCodeArchive, uepak.py io_archive), decoded, in entry order."""
    what = f'{toc.path}: shader library chunk {i}'
    r = R(toc.read(i, decode), what)
    if r.take('<I') != 1:
        raise ValueError(f'{what} is not version 1')
    r.raw(20 * r.take('<I'))
    nsh = r.take('<I')
    r.raw(20 * nsh)
    gids = [r.raw(12) for _ in range(r.take('<I'))]
    [r.take('<II') for _ in range(r.take('<I'))]
    n = r.take('<I')
    ents = r.array('Q', n)
    groups = [r.take('<IIII') for _ in range(r.take('<I'))]
    idx = r.array('I', r.take('<I'))
    if r.o != len(r.b) or n != nsh or len(gids) != len(groups):
        raise ValueError(f'{what} does not parse')
    code = [None] * n
    for g, (gid, (first, cnt, raw, stored)) in enumerate(zip(gids, groups)):
        owner = chunk(gid)
        if not owner:
            raise ValueError(f'{what}: shader group {g} ({gid.hex()}) is in no container')
        d = owner[0].read(owner[1], decode)
        if len(d) != stored:
            raise ValueError(f'{what}: shader group {g} is {len(d)} bytes, its entry says {stored}')
        if stored < raw:
            d = decode(d, raw)
        mine = sorted((ents[s] >> 34, s) for s in idx[first:first + cnt])
        for (at, s), (nxt, _) in zip(mine, mine[1:] + [(raw, None)]):
            code[s] = d[at:nxt]
    for c in code:
        if c is not None:
            yield c


def nvn2_record(s):
    """(name, control, code) of a decoded NVN2 shader record (uepak.py sage_record), or None if the layout differs."""
    k = s.find(CTL_NVN2)
    if k < 12 or k + 0x48 > len(s):
        return None
    nctl, ncode = struct.unpack_from('<I', s, k - 4)[0], struct.unpack_from('<I', s, k + 0x44)[0]
    c = k - 4 - ncode
    if c < 4 or struct.unpack_from('<I', s, c - 4)[0] != ncode or k + nctl > len(s):
        return None
    name = bytes(s[:c - 4]).rstrip(b'\0').rsplit(b'\0', 1)[-1].decode('latin1', 'replace')
    return name, s[k:k + nctl], s[c:c + ncode]


def keyed(control, code):
    """(key, stage, control, code) as shader_census.nvn2_stages keys an NVN2 stage, or None."""
    stage = NVN2_STAGES.get(struct.unpack_from('<I', control, 0x54)[0] >> 16)
    if not stage:
        return None
    control = control[:NVN2_CONTROL] + bytes(0x880 - NVN2_CONTROL)
    size, c1_start, c1_end = struct.unpack_from('<I', control, 0x38)[0], *struct.unpack_from('<II', control, 0x40)
    length = c1_end if c1_start >= size and c1_end >= c1_start and c1_end <= len(code) else size
    if not 0 < length <= len(code):
        return None
    return digest(control + code[:length]), stage, control, code[:length]


def io_tocs(rom):
    return [Toc(rom, p) for p in sorted(rom.entries) if p.endswith('.utoc')]


def has_shader_libraries(romfs):
    try:
        rom = RomFS(romfs)
        return any(t.of_type(IO_SHADER_LIBRARY) for t in io_tocs(rom))
    except (OSError, ValueError, KeyError, struct.error):
        return False


def inventory(build, romfs, exefs, exe, progress=None):
    """key -> dict(stage, control, code, source, origin, startup), the form shader_census.inventory returns."""
    rom = RomFS(romfs)
    tocs = io_tocs(rom)
    chunk = {}
    for t in reversed(tocs):
        for k, c in enumerate(t.ids):
            chunk.setdefault(c, (t, k))
    libs = [(t, i) for t in tocs for i in t.of_type(IO_SHADER_LIBRARY)]
    shaders, odd = {}, 0
    dec = Oodle(build, exe, exefs, romfs)
    try:
        for n, (t, i) in enumerate(libs):
            for s in library_shaders(t, i, dec, chunk.get):
                rec = nvn2_record(s)
                k = rec and keyed(rec[1], rec[2])
                if not k:
                    odd += 1
                    continue
                key, stage, control, code = k
                shaders.setdefault(key, dict(stage=stage, control=control, code=code, source=rec[0],
                                             origin='iostore', startup=True))
            if progress:
                progress(n + 1, len(libs))
    finally:
        dec.close()
    return shaders, odd


STAGES = {'vs': 0, 'ps': 4, 'cs': 5, 'ts': 6, 'ms': 7}   # runtime/gfx.h GFX_STAGE_*
ANS_HEAD, PRG_HEAD = b'NX2NSANS', b'NX2NSPRG'


def code_hash(stage, control, code, c1):
    """gfx_shader.c build_code_hash (prepare_shaders.code_hash)."""
    import hashlib
    return hashlib.sha256(struct.pack('<I', stage) + control + struct.pack('<Q', len(code)) + code +
                          struct.pack('<I', len(c1)) + c1).digest()


def programs(s):
    """The program records an NVN2 shader can be recorded as, {code hash: record}: the runtime keeps most stages
    without their c1 constants (code up to the control section's code size, c1 empty) and some with them (the whole
    code through the c1 end, and the c1 part again as c1), so both are made and the list picks the one play recorded.
    Checked against a recorded shader-programs.bin: 1,518 of 1,519 programs byte-identical (the other is a stage the
    runtime does not prepare)."""
    stage, control, code = STAGES[s['stage']], s['control'][:0x880], s['code']
    size, c1_start = struct.unpack_from('<I', control, 0x38)[0], struct.unpack_from('<I', control, 0x40)[0]
    out = {}
    splits = [(code[:size], b'')]
    if size <= c1_start < len(code):   # keyed() cut the code at the c1 end
        splits.append((code, code[c1_start:]))
    for body, c1 in splits:
        if not body:
            continue
        h = code_hash(stage, control, body, c1)
        out[h] = (struct.pack('<III', stage, len(body), len(c1)) + h + control + body + c1, (control, body, c1))
    return out


IN_MAGIC, IN_HEAD = b'TKSI', struct.Struct('<6I')   # gfx_shader.c InHeader: magic, version, stage, length, c1 len, answers size
LIST_MAGIC, LIST_REC = b'OBSI', struct.Struct('<32sIIII32s')   # input id, stage, length, c1 len, answers size, code hash


def strip_inputs(in_dir, out):
    """The shader cache's stored inputs (<cache>/in/<input id>.in: stage, the full answers, control, code, c1) without
    their control, code and c1: input id, stage, code length, c1 length, the answers and the code hash. Run on a
    build that has been played; the result is the shipped list (no game code). The answers keep what
    shader-answers.bin cannot hold (the retype answers in each input id), so every shader a pipeline key names
    comes back. Returns the record count."""
    recs = []
    for p in sorted(Path(in_dir).glob('*.in')):
        b = p.read_bytes()
        magic, ver, stage, length, c1n, asz = IN_HEAD.unpack_from(b)
        if magic != struct.unpack('<I', IN_MAGIC)[0] or len(b) != IN_HEAD.size + asz + 0x880 + length + c1n:
            continue
        o = IN_HEAD.size
        answers = b[o:o + asz]; o += asz
        control = b[o:o + 0x880]; o += 0x880
        code, c1 = b[o:o + length], b[o + length:o + length + c1n]
        recs.append(LIST_REC.pack(bytes.fromhex(p.stem), stage, length, c1n, asz,
                                  code_hash(stage, control, code, c1)) + answers)
    Path(out).write_bytes(LIST_MAGIC + struct.pack('<II', ver if recs else 1, len(recs)) + b''.join(recs))
    return len(recs)


def read_inputs(path):
    """[(input id, stage, length, c1 len, answers bytes, code hash)] of a strip_inputs list."""
    b = Path(path).read_bytes()
    if b[:4] != LIST_MAGIC:
        raise ValueError(f'{path}: not a stored-input list')
    ver, n = struct.unpack_from('<II', b, 4)
    o, out = 12, []
    for _ in range(n):
        iid, stage, length, c1n, asz, h = LIST_REC.unpack_from(b, o)
        o += LIST_REC.size
        out.append((iid, ver, stage, length, c1n, b[o:o + asz], h))
        o += asz
    return out


def restore_inputs(build, inputs, parts, say=print):
    """<build>/shader-cache/in/<input id>.in for every listed input whose program is in the game files (parts: code
    hash -> (control, code, c1)). The boot prewarm translates them (gfx_shader_prewarm) and builds the recorded
    pipelines from them. Returns (written, missing)."""
    d = Path(build) / 'shader-cache' / 'in'
    d.mkdir(parents=True, exist_ok=True)
    written = missing = 0
    for iid, ver, stage, length, c1n, answers, h in inputs:
        p = parts.get(h)
        if not p or len(p[1]) != length or len(p[2]) != c1n:
            missing += 1
            continue
        control, code, c1 = p
        (d / (iid.hex() + '.in')).write_bytes(IN_HEAD.pack(struct.unpack('<I', IN_MAGIC)[0], ver, stage, length, c1n,
                                                           len(answers)) + answers + control + code + c1)
        written += 1
    return written, missing


def read_answers(path):
    """The records of a shader-answers.bin ("NX2NSANS", version, record size): [(code hash, record)]."""
    b = Path(path).read_bytes()
    if b[:8] != ANS_HEAD:
        raise ValueError(f'{path}: not a shader answers file')
    size = struct.unpack_from('<I', b, 12)[0]
    return b[:16], [(b[o + 4:o + 36], b[o:o + size]) for o in range(16, len(b) - size + 1, size)]


def prepare(build, romfs, exefs, exe, answers, keys=None, inputs=None, run=True, say=print):
    """The shader list (a shader-answers.bin recorded in play: code hashes and bind states, no code) joined with the
    shaders of the player's own IoStore: shader-programs.bin and shader-answers.bin as play records them, the same as
    seed-programs.bin / seed-answers.bin for `<exe> --prepare-shaders`, and pipeline-keys.bin (pipeline states with the
    shaders named by input id, no code) for the boot prewarm. Then runs the preparation. Returns its exit code."""
    import shutil
    import time
    build = Path(build)
    t0 = time.monotonic()
    head, recs = read_answers(answers)
    need = {h for h, _ in recs}
    shaders, odd = inventory(build, romfs, exefs, exe)
    progs, parts = {}, {}
    for s in shaders.values():
        if s['stage'] not in STAGES:
            continue
        for h, (rec, part) in programs(s).items():
            parts.setdefault(h, part)
            if h in need:
                progs.setdefault(h, rec)
    kept = [r for h, r in recs if h in progs]
    say(f'shaders: {len(shaders)} in the game files; {len(progs)} of the list\'s {len(need)} programs found, '
        f'{len(kept)} of {len(recs)} bind states kept ({time.monotonic() - t0:.1f} s)')
    prg = PRG_HEAD + struct.pack('<II', 1, 0) + b''.join(progs.values())
    ans = head + b''.join(kept)
    for name, data in (('shader-programs.bin', prg), ('shader-answers.bin', ans),
                       ('seed-programs.bin', prg), ('seed-answers.bin', ans)):
        (build / name).write_bytes(data)
    if keys and Path(keys).is_file():
        shutil.copyfile(keys, build / 'pipeline-keys.bin')
    if inputs and Path(inputs).is_file():
        w, m = restore_inputs(build, read_inputs(inputs), parts)
        say(f'shaders: {w} stored inputs restored for the recorded pipelines, {m} not in the game files')
    if not run:
        return 0
    r = subprocess.run([str(build / exe), '--prepare-shaders'], cwd=str(build))
    for _ in range(3):   # prepare_shaders.py RETRIES: a driver that crashes on a variant crashes on it every run
        if not r.returncode:
            break
        say(f'shaders: the preparation stopped (rc={r.returncode}); building again without the variants it was on')
        r = subprocess.run([str(build / exe), '--warm-shaders'], cwd=str(build))
    for seed in ('seed-programs.bin', 'seed-answers.bin'):
        (build / seed).unlink(missing_ok=True)
    if not r.returncode and inputs:   # the stored inputs and recorded pipelines too, not on the first boot screen
        p = subprocess.run([str(build / exe), '--prewarm'], cwd=str(build))
        say(f'shaders: {exe} --prewarm rc={p.returncode} (recorded pipelines built ahead of the first start)')
    say(f'shaders: {exe} --prepare-shaders rc={r.returncode}; {time.monotonic() - t0:.1f} s')
    return r.returncode


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('build', type=Path)
    ap.add_argument('--romfs', type=Path)
    ap.add_argument('--exefs', type=Path)
    ap.add_argument('--exe', default='orbitals.exe')
    ap.add_argument('--answers', type=Path, help='the shader list: a shader-answers.bin recorded in play')
    ap.add_argument('--keys', type=Path, help='a pipeline-keys.bin recorded in play')
    ap.add_argument('--inputs', type=Path, help='a stored-input list (--strip)')
    ap.add_argument('--no-run', action='store_true', help='write the files only')
    ap.add_argument('--strip', type=Path, metavar='OUT',
                    help='developer: write the stored-input list of <build>/shader-cache/in to OUT and stop')
    a = ap.parse_args()
    if a.strip:
        print(f'{strip_inputs(a.build / "shader-cache" / "in", a.strip)} stored inputs -> {a.strip}')
        sys.exit(0)
    romfs, exefs = a.romfs or a.build / 'romfs', a.exefs or a.build / 'exefs'
    if a.answers:
        sys.exit(prepare(a.build, romfs, exefs, a.exe, a.answers, a.keys, a.inputs, run=not a.no_run))
    shaders, odd = inventory(a.build, romfs, exefs, a.exe)
    by = {}
    for s in shaders.values():
        by[s['stage']] = by.get(s['stage'], 0) + 1
    print(f'{len(shaders)} unique shaders {by}; {odd} records not read as NVN2')
