"""Run a build probe with a wall-clock limit and preserve its console and runtime log."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--timeout', type=int, default=60)
    ap.add_argument('--trace', default='')
    ap.add_argument('args', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if Path(a.name).name != a.name or a.name in ('.', '..'):
        ap.error('--name must be a filename component')
    out = a.build / 'diagnostics'
    out.mkdir(exist_ok=True)
    args = a.args[1:] if a.args[:1] == ['--'] else a.args
    exe = a.build / 'orbitals.exe'
    profile = out / 'probe-profile'
    profile.mkdir(exist_ok=True)
    command = [str(exe), '--exefs', str(a.build / 'exefs'), '--romfs', str(a.build / 'romfs'),
               '--profile', str(profile), *args]
    env = dict(os.environ)
    env['NX2NAVI_TRACE'] = a.trace
    started = time.time()
    timed_out = False
    with (out / (a.name + '.txt')).open('w') as log:
        process = subprocess.Popen(command, cwd=a.build, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            code = process.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            code = process.wait()
    runtime_log = a.build / 'logs' / 'nx2navi.log'
    if runtime_log.is_file():
        shutil.copy2(runtime_log, out / (a.name + '.runtime.log'))
    with exe.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    result = {'command': command, 'exe_sha256': digest, 'exit': code, 'timeout': timed_out,
              'elapsed_seconds': time.time() - started}
    (out / (a.name + '.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    raise SystemExit(124 if timed_out else code)


if __name__ == '__main__':
    main()
