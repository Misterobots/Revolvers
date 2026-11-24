"""RecompLauncher prepare step: builds Orbitals from the player's .nup into the install folder.

Runs make_game.py --events and turns its "@@" JSON lines into RecompLauncher's
"progress <0..1> <label>" lines, then prepares the shaders (ue_shaders.py) from the shipped shader list
(data/orbitals: code hashes, bind states and pipeline states recorded in play, no game code) and the player's own
game files, so the first play does not stop to translate and build them. Exits 0 on success.

The shaders are prepared with the translator settings the game starts with (recomplauncher.json "start.env"), which
the launcher passes here through "prepare.env": the shader cache is keyed by them.

Usage: python recomplauncher_prepare.py --nup <file> --out <install> [--msys <ucrt64>]
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

# Share of the whole build each step takes (roughly by time).
# Measured on the Deluxe .nup: exefs 0 s, romfs 14 s, lift 182 s, compile 466 s, shaders 16 s (ue_shaders.py).
STEPS = [('exefs', 0.01), ('romfs', 0.03), ('lift', 0.27), ('compile', 0.66), ('shaders', 0.03)]
DATA = Path(__file__).with_name('data') / 'orbitals'
LABELS = {'exefs': 'Reading game code', 'romfs': 'Unpacking game files', 'lift': 'Translating code',
          'compile': 'Compiling', 'shaders': 'Preparing shaders'}


def find_msys():
    for c in (r'C:\msys64', r'D:\msys64', os.path.expandvars(r'%LOCALAPPDATA%\Programs\msys64'),
              os.path.expanduser(r'~\msys64')):
        if Path(c, 'ucrt64', 'bin', 'clang.exe').exists():
            return str(Path(c, 'ucrt64'))
    return None


def overall(step, frac):
    base = 0.0
    for name, w in STEPS:
        if name == step:
            return min(base + w * frac, 0.99)
        base += w
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nup', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--msys')
    a = ap.parse_args()

    msys = a.msys or find_msys()
    if not msys:
        print('error: MSYS2 UCRT64 with clang was not found (run setup-msys2.ps1)', file=sys.stderr, flush=True)
        return 2

    print('progress 0 Starting build', flush=True)
    cmd = [sys.executable, str(Path(__file__).with_name('make_game.py')),
           '--nup', a.nup, '--out', a.out, '--msys', msys, '--events']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    last = -1.0
    for line in p.stdout:
        if not line.startswith('@@'):
            print(line.rstrip(), flush=True)   # keep the log readable
            continue
        try:
            ev = json.loads(line[2:])
        except ValueError:
            continue
        if ev.get('ev') == 'progress' and ev.get('total'):
            step = ev.get('step')
            v = overall(step, ev['done'] / ev['total'])
            if v is not None and v - last >= 0.005:
                last = v
                print(f'progress {v:.3f} {LABELS.get(step, step)}', flush=True)
        elif ev.get('ev') == 'error':
            print(f"error: {ev.get('message', '')}", file=sys.stderr, flush=True)
    code = p.wait()
    if code:
        return code

    # shaders: the build's make_game.py step skips UE5 games; the IoStore libraries are read here
    if (DATA / 'shader-answers.bin').is_file():
        print(f"progress {overall('shaders', 0):.3f} {LABELS['shaders']}", flush=True)
        sys.path.insert(0, str(Path(__file__).parent))
        import ue_shaders
        out = Path(a.out)
        try:
            rc = ue_shaders.prepare(out, out / 'romfs', out / 'exefs', 'orbitals.exe', DATA / 'shader-answers.bin',
                                    DATA / 'pipeline-keys.bin', DATA / 'shader-inputs.bin',
                                    say=lambda m: print(m, flush=True))
        except (OSError, ValueError) as e:
            rc = 1
            print(f'shaders: {e}', flush=True)
        if rc:   # the game still runs, translating shaders as it meets them
            print('shaders: not prepared; the game will prepare them as it meets them', flush=True)
    print('progress 1 Build finished', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
