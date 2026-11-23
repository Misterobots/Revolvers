"""Independently classify SIMD words with Capstone and report half FP usage."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import capstone
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'recomp'))
from nso import Nso
from simd_c import SimdC
from lifter import Unsupported


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exefs', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    a = ap.parse_args()
    decoder = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    emitter = SimdC()
    report = {'capstone': capstone.__version__, 'modules': {}}
    # Independent known words: S ADD, S SUB, D ADD; an H ADD must never emit integer/S/D math.
    for word, expected in [(0x4e22d420, 'SV_FOP_S(FOP_ADD'), (0x4ea2d420, 'SV_FOP_S(FOP_SUB'),
                           (0x4e62d420, 'SV_FOP_D(FOP_ADD')]:
        assert emitter.dp(word, 0).startswith(expected)
    for name in ('main', 'subsdk0', 'sdk'):
        nso = Nso(a.exefs / name)
        start, size = nso.text
        words = np.frombuffer(nso.image, dtype='<u4', count=size // 4, offset=start)
        group = (words >> 25) & 15
        candidates = words[(group == 7) | (group == 15)]
        unique, counts = np.unique(candidates, return_counts=True)
        half, unsupported, examples = Counter(), Counter(), []
        for word, count in zip(unique, counts):
            word, count = int(word), int(count)
            decoded = next(decoder.disasm_lite(word.to_bytes(4, 'little'), 0), None)
            if not decoded:
                continue
            _, _, mnemonic, operands = decoded
            ishalf = (mnemonic.startswith('f') and re.search(r'\bh\d+\b|\.\d*h\b', operands))
            if ishalf:
                half[mnemonic] += count
                try:
                    emission = emitter.dp(word, 0)
                except Unsupported as e:
                    emission = 'UNSUPPORTED: ' + str(e)
                    unsupported[mnemonic] += count
                examples.append({'word': f'{word:08x}', 'asm': mnemonic + ' ' + operands,
                                 'count': count, 'emission': emission})
        report['modules'][name] = {'text_words': len(words), 'half_fp_counts': dict(half),
                                    'unsupported_half_counts': dict(unsupported), 'examples': examples}
        print(name, dict(half), 'unsupported', dict(unsupported), flush=True)
    a.report.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
