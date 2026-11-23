"""Compare the extracted updated dump with a built ExeFS and packed RomFS."""
import argparse
import hashlib
import json
from pathlib import Path
from make_game import EXEFS_FILES, MODULES, flat_nso


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--update', type=Path, required=True)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    a = ap.parse_args()
    report = {'update': str(a.update.resolve()), 'build': str(a.build.resolve()), 'exefs': [], 'romfs': []}
    for name in EXEFS_FILES:
        raw = (a.update / 'exefs' / name).read_bytes()
        flat = flat_nso(raw, name) if name in MODULES else raw
        built = (a.build / 'exefs' / name).read_bytes()
        report['exefs'].append({'name': name, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                               'flat_sha256': hashlib.sha256(flat).hexdigest(), 'match': flat == built})
    index = json.loads((a.build / 'romfs' / 'index.json').read_text())
    entries = {e['path']: e for e in index['files']}
    files = {p.relative_to(a.update / 'romfs').as_posix(): p
             for p in (a.update / 'romfs').rglob('*') if p.is_file() and p.name != '.listing'}
    report['missing_from_build'] = sorted(files.keys() - entries.keys())
    report['missing_from_dump'] = sorted(entries.keys() - files.keys())
    with (a.build / 'romfs' / 'romfs.bin').open('rb') as image:
        for name in sorted(files.keys() & entries.keys()):
            e, path = entries[name], files[name]
            image.seek(e['offset'])
            h, match, size = hashlib.sha256(), True, 0
            with path.open('rb') as src:
                while chunk := src.read(4 << 20):
                    h.update(chunk)
                    size += len(chunk)
                    match &= chunk == image.read(len(chunk))
            report['romfs'].append({'path': name, 'sha256': h.hexdigest(), 'size': size,
                                   'match': match and size == e['size']})
    report['pass'] = (not report['missing_from_build'] and not report['missing_from_dump']
                      and all(e['match'] for e in report['exefs'] + report['romfs']))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))
    print(f"input audit: {'PASS' if report['pass'] else 'FAIL'}, {len(report['exefs'])} ExeFS, {len(report['romfs'])} RomFS files")
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
