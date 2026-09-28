#!/usr/bin/env python3
"""The zero-pressure-gradient flat plate against NASA TMR's SST results
(ADR-042, gate 4).

Runs build/tests/flat_plate on TMR's grids -- 69x49, 137x97, 273x193 and
545x385 (ADR-042 decided to run it; VIBEFLOW_FP_FINEST=273x193 stops a grid
short) -- in SST-1994, the variant TMR ran, then judges on the three finest,
69x49 reported:

  4a  Cf at x = 0.97008 monotone over the three finest grids run, observed
      order in [0.8, 3.0], Richardson extrapolation within 1% of 0.0026964,
      the mean of TMR's extrapolations (CFL3D 0.00269681, FUN3D 0.00269607).
  4b  u+ against y+ at x = 0.97008 on the finest grid run within 2% of TMR's
      CFL3D profile (545x385), 1 <= y+ <= 500.
  4c  the log law u+ = ln(y+)/0.41 + 5.0 within 3%, 60 <= y+ <= 250.
  4d  the peak of nu_t/nu at x = 0.97008 within 2% of 221.7.
Gate 1's check on these grids is here too: d = y above the plate and
sqrt(x^2 + y^2) ahead of it, to 1e-13.
Every run must reach flat_plate's steady state. SST-2003 is run on 273x193
and reported, not gated.

TMR's files are in cases/flatplate/tmr (flatplate_u+y+_sstv.dat,
cf_convergence_sstv.dat), as downloaded from
https://tmbwg.github.io/turbmodels/flatplate_sst.html.

Run:  python3 flat_plate_gate.py [out dir]      (default build/flatplate)
      VIBEFLOW_FP_REUSE=1 judges the runs already in the out dir.
"""
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases" / "flatplate"
BIN = ROOT / "build" / "tests" / "flat_plate"
NU = 2e-7
CF_REF = 0.0026964
NUT_REF = 221.7
# Step size: 0.01 on every grid, the step the first 35x25 and 69x49 runs
# took. The steady state does not depend on it (ADR-037).
DT = {"69x49": 0.01, "137x97": 0.01, "273x193": 0.01, "545x385": 0.01}
FINAL = re.compile(r"^FINAL Cf (\S+) CD (\S+) tau (\S+) nutPeak (\S+) steps (\d+) steady (\w+)")
WALLD = re.compile(r"wall distance: max\|d - d_exact\| cells (\S+), boundary faces (\S+)")


def zones(path):
    out, cur = {}, None
    for line in open(path):
        s = line.strip()
        if not s or s.lower().startswith("variables"):
            continue
        if s.lower().startswith("zone"):
            cur = s
            out[cur] = []
            continue
        out[cur].append([float(v) for v in s.split()])
    return {k: np.array(v) for k, v in out.items()}


def wall_distance(level, out_dir):
    """Gate 1 on a flat-plate grid: the largest |d - d_exact|, from the run's
    log or, for a log written before flat_plate checked it, a run of the
    check alone."""
    log = out_dir / f"{level}_1994.log"
    text = log.read_text() if log.exists() else ""
    m = WALLD.search(text)
    if m is None:
        r = subprocess.run([str(BIN), str(CASE / f"{level}.hex"), str(out_dir / f"{level}_wd")],
                           cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, "OMP_PROC_BIND": "false",
                                "VIBEFLOW_FP_WALLDIST_ONLY": "1"})
        m = WALLD.search(r.stdout)
    return None if m is None else max(float(m.group(1)), float(m.group(2)))


def run(level, variant, out_dir, reuse):
    prefix = out_dir / f"{level}_{variant}"
    log = Path(str(prefix) + ".log")
    if not (reuse and log.exists()):
        mesh = CASE / f"{level}.hex"
        if not mesh.exists():
            subprocess.run([sys.executable, str(CASE / "make_mesh.py"), level], check=True)
        with open(log, "w") as fh:
            subprocess.run([str(BIN), str(mesh), str(prefix), variant, str(DT[level]), "40000"],
                           cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                           env={**os.environ, "OMP_PROC_BIND": "false"})
    m = None
    for line in open(log):
        m = FINAL.match(line) or m
    if m is None:
        return None
    cf, cd, tau, peak = (float(m.group(i)) for i in range(1, 5))
    prof = np.loadtxt(str(prefix) + ".profile")
    return dict(cf=cf, cd=cd, tau=tau, peak=peak, steps=int(m.group(5)),
                steady=m.group(6) == "yes", prof=prof)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "flatplate"
    out_dir.mkdir(parents=True, exist_ok=True)
    reuse = os.environ.get("VIBEFLOW_FP_REUSE") == "1"
    run_levels = ["69x49", "137x97", "273x193", "545x385"]
    if os.environ.get("VIBEFLOW_FP_FINEST") == "273x193":
        run_levels = run_levels[:-1]
    levels = run_levels[-3:]                  # judged; the rest reported
    ok = True
    res = {}
    print("Flat plate, SST-1994, against TMR's SST-V (CFL3D, FUN3D)")
    for lv in run_levels:
        r = run(lv, "1994", out_dir, reuse)
        tag = "" if lv in levels else "  (reported)"
        if r is None:
            print(f"  {lv}: no result{tag}")
            ok &= lv not in levels
            continue
        res[lv] = r
        print(f"  {lv:<8} Cf(0.97008) {r['cf']:.8f}  CD {r['cd']:.8f}  nu_t/nu peak "
              f"{r['peak']:.3f}  steps {r['steps']}  steady {'yes' if r['steady'] else 'NO'}{tag}",
              flush=True)
        if lv in levels:
            ok &= r["steady"]
    if any(lv not in res for lv in levels):
        print("\nv2b flat-plate GATE (C++): FAIL")
        return 1
    worst = [wall_distance(lv, out_dir) for lv in run_levels]
    g1 = all(w is not None and w <= 1e-13 for w in worst)
    print("  -> 1 (flat-plate grids): d = y above the plate, sqrt(x^2 + y^2) ahead of it; "
          "largest |d - d_exact| " + ", ".join("-" if w is None else f"{w:.1e}" for w in worst)
          + f" (within 1e-13): {'PASS' if g1 else 'FAIL'}")
    ok &= g1
    c1, c2, c3 = (res[lv]["cf"] for lv in levels)            # coarse to fine
    mono = (c2 - c1) * (c3 - c2) > 0
    p = math.log(abs(c2 - c1) / abs(c3 - c2)) / math.log(2.0) if mono else float("nan")
    ext = c3 + (c3 - c2) / (2.0 ** p - 1.0) if mono else float("nan")
    a = mono and 0.8 <= p <= 3.0 and abs(ext - CF_REF) / CF_REF <= 0.01
    print(f"  -> 4a: {'monotone' if mono else 'NOT MONOTONE'}, observed order {p:.3f} "
          f"(in [0.8, 3.0]), Richardson Cf {ext:.8f}, {100 * (ext - CF_REF) / CF_REF:+.3f}% "
          f"from {CF_REF} (within 1%): {'PASS' if a else 'FAIL'}")
    ok &= a

    fine = res[levels[-1]]
    ut = math.sqrt(fine["tau"])
    y, u, nut = fine["prof"][:, 0], fine["prof"][:, 1], fine["prof"][:, 2]
    yp, up = y * ut / NU, u / ut
    tmr = zones(CASE / "tmr" / "flatplate_u+y+_sstv.dat")
    ref = next(v for k, v in tmr.items() if "0.97008" in k)
    ly, lu = ref[:, 0], ref[:, 1]
    sel = (10 ** ly >= 1.0) & (10 ** ly <= 500.0)
    ours = np.interp(ly[sel], np.log10(yp), up)
    dev = np.abs(ours - lu[sel]) / lu[sel]
    b = dev.max() <= 0.02
    print(f"  -> 4b: u+ against TMR's CFL3D profile, 1 <= y+ <= 500 ({sel.sum()} points): "
          f"largest departure {100 * dev.max():.2f}% at y+ = {10 ** ly[sel][dev.argmax()]:.1f} "
          f"(within 2%): {'PASS' if b else 'FAIL'}")
    ok &= b
    ll = (yp >= 60.0) & (yp <= 250.0)
    law = np.log(yp[ll]) / 0.41 + 5.0
    devl = np.abs(up[ll] - law) / law
    c = ll.any() and devl.max() <= 0.03
    print(f"  -> 4c: log law, 60 <= y+ <= 250 ({ll.sum()} cells): largest departure "
          f"{100 * devl.max():.2f}% (within 3%): {'PASS' if c else 'FAIL'}")
    ok &= c
    d = abs(fine["peak"] - NUT_REF) / NUT_REF <= 0.02
    print(f"  -> 4d: nu_t/nu peak {fine['peak']:.3f}, {100 * (fine['peak'] - NUT_REF) / NUT_REF:+.2f}% "
          f"from {NUT_REF} (within 2%): {'PASS' if d else 'FAIL'}")
    ok &= d

    # SST-2003 on 273x193, reported: a grid short of the finest, for the cost
    # (ADR-042).
    r3 = run("273x193", "2003", out_dir, reuse)
    if r3:
        r1 = res.get("273x193")
        rel = f", {100 * (r3['cf'] - r1['cf']) / r1['cf']:+.3f}% from SST-1994" if r1 else ""
        print(f"  reported: SST-2003 on 273x193: Cf {r3['cf']:.8f}{rel}  CD {r3['cd']:.8f}  "
              f"nu_t/nu peak {r3['peak']:.3f}  steady {'yes' if r3['steady'] else 'no'}")
    print("\nv2b flat-plate GATE (C++): " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
