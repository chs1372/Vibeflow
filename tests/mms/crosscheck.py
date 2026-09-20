#!/usr/bin/env python3
"""Cross-check gate: the C++ implementation must reproduce the Python reference.

Runs both MMS gates and compares the L2 error of every (case, mesh, resolution)
combination. The two implementations share no code, so agreement to solver
tolerance is real evidence -- a bug would have to be made twice, identically.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "prototype"
BIN = ROOT / "build" / "tests" / "mms_diffusion"
FIX = ROOT / "tests" / "fixtures"

REL_TOL = 1e-11
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)")
HDR = re.compile(r"^([AB]: .+?) / (\w+) mesh")


def parse(text):
    out, key = {}, None
    for line in text.splitlines():
        h = HDR.match(line)
        if h:
            key = (h.group(1), h.group(2))
            continue
        m = ROW.match(line)
        if m and key:
            out[(key[0], key[1], int(m.group(1)))] = float(m.group(3))
    return out


def main():
    py = subprocess.run([sys.executable, str(PROTO / "mms_diffusion.py"), "8", "16", "32"],
                        cwd=PROTO, capture_output=True, text=True)
    if py.returncode != 0:
        print(py.stdout); print("python gate FAILED"); return 1

    if not BIN.exists():
        print(f"C++ gate binary not built ({BIN}) -- skipping cross-check")
        print("build it with: cmake -S . -B build && cmake --build build -j")
        return 0

    cpp = subprocess.run([str(BIN), str(FIX)], cwd=ROOT, capture_output=True, text=True,
                         env={"OMP_PROC_BIND": "false", "PATH": "/usr/bin:/bin"})
    if cpp.returncode != 0:
        print(cpp.stdout); print("C++ gate FAILED"); return 1

    a, b = parse(py.stdout), parse(cpp.stdout)
    shared = sorted(set(a) & set(b))
    if not shared:
        print("cross-check FAILED: no comparable rows parsed"); return 1

    print(f"{'case':<24}{'mesh':<12}{'N':>4}{'python L2':>15}{'c++ L2':>15}{'rel diff':>12}")
    worst, bad = 0.0, 0
    for k in shared:
        rel = abs(a[k] - b[k]) / max(abs(a[k]), 1e-300)
        worst = max(worst, rel)
        if rel > REL_TOL:
            bad += 1
        print(f"{k[0]:<24}{k[1]:<12}{k[2]:>4}{a[k]:>15.8e}{b[k]:>15.8e}{rel:>12.2e}")

    print(f"\n{len(shared)} rows compared, worst relative difference {worst:.2e} "
          f"(tolerance {REL_TOL:.0e})")
    print("cross-check gate: " + ("PASS" if bad == 0 else f"FAIL ({bad} rows over tolerance)"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
