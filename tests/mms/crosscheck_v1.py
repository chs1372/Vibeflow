#!/usr/bin/env python3
"""Cross-check gate for v1: the C++ operators must reproduce the Python reference.

The two implementations share no code. Agreement on every case is evidence a
single implementation cannot produce on its own -- a bug would have to be made
twice, identically, in two languages.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "prototype"
BUILD = ROOT / "build" / "tests"
REL_TOL = 1e-8
ENV = {"OMP_PROC_BIND": "false", "PATH": "/usr/bin:/bin"}

ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.eE+-]+)")
HDR = re.compile(r"^(convection-diffusion[^(]*?)\s*$")


def parse(text):
    out, key = {}, None
    for line in text.splitlines():
        h = HDR.match(line.split("(")[0].rstrip()) if "convection-diffusion" in line else None
        if h:
            key = "orthogonal" if "/ orthogonal" in line else "distorted"
            continue
        m = ROW.match(line)
        if m and key:
            out[(key, int(m.group(1)))] = float(m.group(3))
    return out


def main():
    binary = BUILD / "mms_convection"
    if not binary.exists():
        print(f"C++ gate not built ({binary}) -- skipping"); return 0

    py = subprocess.run([sys.executable, str(PROTO / "mms_convection.py"), "8", "16", "32"],
                        cwd=PROTO, capture_output=True, text=True)
    cpp = subprocess.run([str(binary), "8", "16", "32"], cwd=ROOT,
                         capture_output=True, text=True, env=ENV)
    if py.returncode or cpp.returncode:
        print(py.stdout, cpp.stdout); print("a gate FAILED"); return 1

    a, b = parse(py.stdout), parse(cpp.stdout)
    shared = sorted(set(a) & set(b))
    if not shared:
        print("cross-check FAILED: no comparable rows"); return 1

    print(f"{'case':<14}{'N':>4}{'python L2':>22}{'c++ L2':>22}{'rel diff':>12}")
    worst, bad = 0.0, 0
    for k in shared:
        rel = abs(a[k] - b[k]) / max(abs(a[k]), 1e-300)
        worst = max(worst, rel)
        bad += rel > REL_TOL
        print(f"{k[0]:<14}{k[1]:>4}{a[k]:>22.12e}{b[k]:>22.12e}{rel:>12.2e}")
    print(f"\n{len(shared)} rows compared, worst relative difference {worst:.2e} "
          f"(tolerance {REL_TOL:.0e})")
    print("v1 cross-check gate: " + ("PASS" if not bad else f"FAIL ({bad} rows over)"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
