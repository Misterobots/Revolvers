"""Check independent opcode fixtures and execute emitted FNEG/UDOT C."""
import argparse
from pathlib import Path
import subprocess
import sys
import capstone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'recomp'))
from simd_c import SimdC


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clang', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = SimdC()
    decoder = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    for word, expected in [(0x4e22d420, 'SV_FOP_S(FOP_ADD'), (0x4ea2d420, 'SV_FOP_S(FOP_SUB'),
                           (0x4e62d420, 'SV_FOP_D(FOP_ADD')]:
        assert s.dp(word, 0).startswith(expected)
    code = '#include "simd.h"\n#include <stdio.h>\n'
    for q in (0, 1):
        for n in range(32):
            for rd in range(32):
                word = 0x2ef8f800 | (q << 30) | (n << 5) | rd
                _, _, mnemonic, operands = next(decoder.disasm_lite(word.to_bytes(4, 'little'), 0))
                assert mnemonic == 'fneg' and ('.8h' if q else '.4h') in operands
        for m in range(32):
            word = 0x2e809400 | (q << 30) | (m << 16) | 0x21
            assert next(decoder.disasm_lite(word.to_bytes(4, 'little'), 0))[2] == 'udot'
        code += f'void neg{q}(Cpu *c){{' + s.dp(0x2ef8f800 | (q << 30), 0) + '}\n'
        code += f'void dot{q}(Cpu *c){{' + s.dp(0x2e809400 | (q << 30), 0) + '}\n'
    code += '''int main(void){Cpu state={0},*c=&state;
for(unsigned x=0;x<65536;x++){
 for(int i=0;i<8;i++) VREG(vu16,0)[i]=x;
 neg1(c); for(int i=0;i<8;i++) if(VREG(vu16,0)[i]!=(x^0x8000)) return 1;
 for(int i=0;i<8;i++) VREG(vu16,0)[i]=x;
 neg0(c); for(int i=0;i<8;i++) if(VREG(vu16,0)[i]!=(i<4?(x^0x8000):0)) return 2;
}
for(unsigned x=0;x<256;x++){
 for(int i=0;i<16;i++) VREG(vu8,0)[i]=x;
 uint32_t acc=VREG(vu32,0)[0];
 dot1(c); for(int i=0;i<4;i++) if(VREG(vu32,0)[i]!=acc+4*x*x) return 3;
 for(int i=0;i<16;i++) VREG(vu8,0)[i]=x;
 dot0(c); for(int i=0;i<4;i++) if(VREG(vu32,0)[i]!=(i<2?acc+4*x*x:0)) return 4;
}
puts("PASS: half bit patterns, Q clearing, UDOT overlap and wraparound");return 0;
}
'''
    source = a.out / 'simd-regression.c'
    source.write_text(code)
    exe = source.with_suffix('.exe')
    runtime = Path(__file__).resolve().parents[1] / 'runtime'
    subprocess.run([a.clang, '-O2', '-march=x86-64-v3', '-I', str(runtime), str(source), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()
