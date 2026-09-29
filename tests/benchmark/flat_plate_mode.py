#!/usr/bin/env python3
"""ADR-043, step 1: the odd-even mode's growth per step, measured in C++.

Each run restarts flat_plate from a grid's steady state (VIBEFLOW_FP_INIT)
at a fixed step (VIBEFLOW_FP_RAMP=1) with n PISO correctors and one outer
iteration, and prints every step's odd-even change (VIBEFLOW_FP_MODE):

    A(phi) = max over interior cells |d(i-1) - 2 d(i) + d(i+1)| / 4,

d the change of phi over the step, i the column. A run ends after 200 steps,
or when A(p) passes 1e-2. The growth factor per step, lambda, is the
geometric mean of A(p)'s ratio over the run's last 20 steps; for a run that
reached the bound before its 23rd step, over its steps 3 on, or its last
ratio if it ended by step 4 (revised after the first runs, ADR-043).

Run:  flat_plate_mode.py <build dir> <out dir> <run name> [<run name> ...]
      flat_plate_mode.py --list      the runs ADR-043 names
      flat_plate_mode.py --table <out dir>   lambda from the logs there
The steady states are read from <out dir>/<grid>_c4.state, which the
"steady:<grid>" runs write (four correctors, ADR-042's ramp and criteria;
545x385's and 273x193's may be copied in from their own marches).
"""

import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MESH = ROOT / "cases" / "flatplate"
STEPS, WINDOW, STOP = 200, 20, 1e-2

ABLATIONS = {                       # ADR-043: at n = 2, dt = 1e-2 on 545x385
    "notranspose": {"VIBEFLOW_FP_NO_TRANSPOSE": "1"},
    "bdf1": {"VIBEFLOW_FP_BDF1": "1"},
    "v1": {"VIBEFLOW_OLDFLUX": "v1"},
    "momtol": {"VIBEFLOW_MOMENTUM_TOL": "1e-15"},
}


def runs():
    r = {}
    for dt in ("1e-3", "3e-3", "1e-2"):
        for n in (2, 3):
            r[f"545x385:n{n}:{dt}"] = ("545x385", n, dt, {})
        for g in ("273x193", "137x97"):
            for n in (1, 2):
                r[f"{g}:n{n}:{dt}"] = (g, n, dt, {})
    r["545x385:n4:1e-2"] = ("545x385", 4, "1e-2", {})
    for name, env in ABLATIONS.items():
        r[f"545x385:n2:1e-2:{name}"] = ("545x385", 2, "1e-2", env)
    for g in ("545x193", "273x385"):
        r[f"{g}:n2:1e-2"] = (g, 2, "1e-2", {})
    # Test (b) of a cause (ADR-043's rules): D_f without the vertical
    # diffusion, at n = 2 and 1e-2 on 545x385.
    r["545x385:n2:1e-2:rcaxis"] = ("545x385", 2, "1e-2", {"VIBEFLOW_RC_AXIS_OFF": "1"})
    return r


def steady(build, out, grid, init=None):
    """March a grid to its steady state with four correctors (the default),
    saving it as <out>/<grid>_c4.state."""
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="false", VIBEFLOW_FP_SAVE="1")
    if init:
        env["VIBEFLOW_FP_INIT"] = str(init)
    prefix = Path(out) / f"{grid}_c4"
    with open(f"{prefix}.log", "w") as log:
        subprocess.run([str(Path(build) / "tests" / "flat_plate"), str(MESH / f"{grid}.hex"),
                        str(prefix), "1994", "0.01", "40000"], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=False)


def run(build, out, name):
    grid, n, dt, extra = runs()[name]
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="false",
               VIBEFLOW_FP_INIT=str(Path(out) / f"{grid}_c4.state"), VIBEFLOW_FP_RAMP="1",
               VIBEFLOW_FP_MODE="1", VIBEFLOW_FP_MODE_STOP=str(STOP), VIBEFLOW_CORRECTORS=str(n))
    env.update(extra)
    tag = name.replace(":", "_")
    with open(Path(out) / f"mode_{tag}.log", "w") as log:
        subprocess.run([str(Path(build) / "tests" / "flat_plate"), str(MESH / f"{grid}.hex"),
                        str(Path(out) / f"mode_{tag}"), "1994", dt, str(STEPS)], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=False)


MODE = re.compile(r"^\s+mode\s+(\d+)\s+A\(p\) (\S+) at \((\S+), (\S+)\)\s+A\(u\) (\S+) at "
                  r"\((\S+), (\S+)\)\s+A\(v\) (\S+)")


def growth(log):
    steps, ap, where = [], [], []
    for line in open(log):
        m = MODE.match(line)
        if m:
            steps.append(int(m.group(1)))
            ap.append(float(m.group(2)))
            where.append((float(m.group(3)), float(m.group(4))))
    if not ap or not all(math.isfinite(a) and a > 0 for a in ap):
        return None
    if len(ap) > WINDOW + 2:
        lam = (ap[-1] / ap[-1 - WINDOW]) ** (1.0 / WINDOW)
    elif len(ap) >= 5:
        # Revised after the first runs (ADR-043): a run that reached the bound
        # before its 23rd step, from its steps 3 on; the first two carry the
        # restart's own disturbance.
        lam = (ap[-1] / ap[2]) ** (1.0 / (len(ap) - 3))
    elif len(ap) >= 2:
        lam = ap[-1] / ap[-2]                 # ended by step 4: the last ratio
    else:
        return None
    return {"steps": steps[-1], "lambda": lam, "A_p": ap[-1], "at": where[-1],
            "stopped": ap[-1] > STOP}


def growth_phase(log, width=10, short=3, floor=1e-10):
    """ADR-043's third measure, stated after the first two and judged on runs
    made after it: the steepest sustained growth of A(p), the largest
    (A(k+w)/A(k))^(1/w) over k >= 3 with A(k) above the solvers' noise
    (floor), w = 10 steps; a run that ends before step 13, w = 3; one that
    ends before step 6, its largest single-step ratio after step 2."""
    ap = []
    for line in open(log):
        m = MODE.match(line)
        if m:
            ap.append(float(m.group(2)))
    ap = [a if math.isfinite(a) else float("inf") for a in ap]
    w = width if len(ap) >= 3 + width else short
    best = None
    for k in range(2, len(ap) - w):                 # index k is step k + 1
        if ap[k] >= floor:
            g = (ap[k + w] / ap[k]) ** (1.0 / w)
            best = g if best is None else max(best, g)
    if best is None:
        for k in range(2, len(ap)):
            if ap[k - 1] > 0:
                g = ap[k] / ap[k - 1]
                best = g if best is None else max(best, g)
    return best


def table(out):
    print(f"{'run':34s} {'steps':>5s} {'lambda':>9s} {'A(p) last':>10s}  where")
    for name in runs():
        log = Path(out) / f"mode_{name.replace(':', '_')}.log"
        if not log.exists():
            continue
        g = growth(log)
        if g is None:
            print(f"{name:34s}  (no growth factor: too few steps or not finite)")
            continue
        gp = growth_phase(log)
        print(f"{name:34s} {g['steps']:5d} {g['lambda']:9.4f} {g['A_p']:10.2e}  "
              f"x = {g['at'][0]:.4f}, y = {g['at'][1]:.2e}"
              f"{'  (stopped at the bound)' if g['stopped'] else ''}"
              f"   steepest growth {gp if gp is None else round(gp, 4)}")


def main(argv):
    if len(argv) >= 2 and argv[1] == "--list":
        for k in runs():
            print(k)
        return 0
    if len(argv) >= 3 and argv[1] == "--table":
        table(argv[2])
        return 0
    if len(argv) < 4:
        print(__doc__)
        return 2
    build, out = argv[1], argv[2]
    Path(out).mkdir(parents=True, exist_ok=True)
    for name in argv[3:]:
        if name.startswith("steady:"):
            parts = name.split(":")
            steady(build, out, parts[1], parts[2] if len(parts) > 2 else None)
        else:
            run(build, out, name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
