#!/usr/bin/env python3
"""ADR-045: the backward-facing step against TMR's SST results from CFL3D.

Builds the meshes (gate 1); marches level 2 first order from the uniform
stream (L2_up, only a start: ADR-045's fourth revision), then in linear
upwind twice -- first with the inlet at U, then from that state with the
inlet speed that gives CFL3D's centre velocity at x = -4 -- and level 1 from
level 2's state; then judges level 1 against CFL3D's results on the same
grid (TMR's level 1):

  2. at x = -4: Cf within 2% of CFL3D's, u within 0.01 of its profile at
     each of its points in 1 <= y <= 2.5;
  3. the reattachment point within 2% of CFL3D's (6.54);
  4. Cp within 0.015 of CFL3D's on the bottom wall, -4 <= x <= 30;
  5. u within 0.03 of CFL3D's at x = 1, 4, 6, 10, at its points in 0 <= y <= 3;
  6. every march steady or quasi-steady.

Levels 3 and 4 are not run (ADR-045's revision: their outlet columns
diverge). Each march's ramp follows its Courant number (the third
revision), and a march settles at the dt it can take, never past its
level's. Reported: level 2 beside level 1, the corner bubble, the
experiment (backstep_report.py).

Run:  backstep_gate.py [build dir, default build] [out dir, default
      build/backstep]      VIBEFLOW_BS_REUSE=1 reuses the logs already there;
      VIBEFLOW_BS_UP_INIT=<state> starts L2_up from a state instead of the
      uniform stream (the fourth revision's continued exploration run).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases" / "backstep"
TMR = CASE / "tmr"
UC_CFL3D = 0.998          # CFL3D's u at (x, y) = (-4, 5)
XR_CFL3D = None           # from its Cf file, below
DT = {2: 0.16, 1: 0.08}


def run(build, out, name, level, u_in, init=None, extra=None):
    log = Path(out) / f"{name}.log"
    if os.environ.get("VIBEFLOW_BS_REUSE") and log.exists() and "FINAL" in log.read_text():
        return log
    # Threads: VIBEFLOW_BS_THREADS, or a file `threads` in the output
    # directory read as each march starts -- the cost only.
    threads = os.environ.get("VIBEFLOW_BS_THREADS", "1")
    tf = Path(out) / "threads"
    if tf.exists():
        threads = tf.read_text().strip() or threads
    env = dict(os.environ, OMP_NUM_THREADS=threads, OMP_PROC_BIND="false", VIBEFLOW_BS_SAVE="1")
    print(f"  {name}: {threads} thread(s)", flush=True)
    if init:
        env["VIBEFLOW_BS_INIT"] = str(init)
    env.update(extra or {})
    with open(log, "w") as fh:
        subprocess.run([str(Path(build) / "tests" / "backstep"), str(CASE / f"L{level}.hex"),
                        str(Path(out) / name), str(DT[level]), "5000", f"{u_in:.12g}"],
                       env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    return log


def parse_final(line):
    """backstep's FINAL line: name-value pairs, the bubble's two values after
    its name."""
    t, d, i = line.split()[1:], {}, 0
    while i < len(t):
        if t[i] == "bubble":
            d["bubble"] = (float(t[i + 1]), float(t[i + 2])); i += 3
        else:
            d[t[i]] = t[i + 1]; i += 2
    return d


def final(log):
    for line in open(log):
        if line.startswith("FINAL"):
            d = parse_final(line)
            return {"xr": float(d["xr"]), "cf4": float(d["cf4"]), "uc4": float(d["uc4"]),
                    "settled": d["settled"], "steps": int(d["steps"]), "full": int(d["full"]),
                    "dt": float(d.get("dt", "nan")), "seconds": float(d["seconds"]),
                    "bubble": d["bubble"]}
    return None


def cfl3d_zones(path):
    zones, cur = {}, None
    for line in open(path):
        if line.upper().startswith("ZONE"):
            cur = line.split('"')[1]
            zones[cur] = []
        elif cur and line.strip() and not line.startswith(("#", "VAR", "var")):
            zones[cur].append([float(v) for v in line.split()])
    return {k: np.array(v) for k, v in zones.items()}


def two_col(path):
    rows = [ln.split() for ln in open(path) if ln.strip() and not ln.lstrip().startswith(("#", "v", "V"))]
    return np.array(rows, dtype=float)


def floor_values(xq, upstream, x, v):
    """Ours at the points xq on one floor -- upstream of the step (y = 1,
    x < 0) or behind it (y = 0, x > 0) -- never across the step: linear
    between faces and, towards the corner past the floor's last face, linear
    from its two nearest faces."""
    m = x < 0.0 if upstream else x > 0.0
    xs, vs = x[m], v[m]
    xq = np.asarray(xq, dtype=float)
    out = np.interp(xq, xs, vs)
    lo, hi = xq < xs[0], xq > xs[-1]
    out[lo] = vs[0] + (xq[lo] - xs[0]) * (vs[1] - vs[0]) / (xs[1] - xs[0])
    out[hi] = vs[-1] + (xq[hi] - xs[-1]) * (vs[-1] - vs[-2]) / (xs[-1] - xs[-2])
    return out


def split_floors(ref):
    """CFL3D's wall points in the file's order along the wall: the step's
    corner x = 0 appears twice, first as the upstream floor's end, then as
    the lower floor's start."""
    zero = np.flatnonzero(ref[:, 0] == 0.0)
    assert len(zero) == 2, "expected the step corner twice"
    return ref[:zero[0] + 1], ref[zero[1]:]


def reattachment(x, cf):
    xr = None
    for i in range(1, len(x)):
        if x[i - 1] >= 0.0 and cf[i - 1] < 0.0 <= cf[i]:
            xr = x[i - 1] - cf[i - 1] * (x[i] - x[i - 1]) / (cf[i] - cf[i - 1])
    return xr


def main(argv):
    build = argv[1] if len(argv) > 1 else str(ROOT / "build")
    out = argv[2] if len(argv) > 2 else str(ROOT / "build" / "backstep")
    Path(out).mkdir(parents=True, exist_ok=True)
    ok = True

    print("1. the mesh")
    r = subprocess.run([sys.executable, str(CASE / "make_mesh.py"), "2", "1"], capture_output=True, text=True)
    print(r.stdout.rstrip())
    ok &= r.returncode == 0

    print("\nmarches")
    # The start: first order from the uniform stream (fourth revision).
    up = final(run(build, out, "L2_up", 2, 1.0, os.environ.get("VIBEFLOW_BS_UP_INIT"),
                   {"VIBEFLOW_CONVECTION": "upwind"}))
    if up is None:
        print("  level 2, first order: no FINAL line"); return 1
    print(f"  level 2, first order (the start): {up['steps']} steps, {up['full']} at dt {up['dt']:g}, "
          f"{up['settled']}, {up['seconds']:.0f} s")
    cal = final(run(build, out, "L2_cal", 2, 1.0, Path(out) / "L2_up.state"))
    if cal is None:
        print("  level 2, U_in = 1: no FINAL line"); return 1
    u_in = UC_CFL3D / cal["uc4"]
    print(f"  level 2, U_in = 1: u_c(-4) {cal['uc4']:.6f} -> U_in = {u_in:.6f}  "
          f"({cal['steps']} steps, {cal['full']} at dt {cal['dt']:g}, {cal['settled']}, "
          f"{cal['seconds']:.0f} s)")
    if not np.isfinite(u_in):
        print("  level 2, U_in = 1: no velocity to set U_in from"); return 1
    marches = {}
    marches[2] = final(run(build, out, "L2", 2, u_in, Path(out) / "L2_cal.state"))
    marches[1] = final(run(build, out, "L1", 1, u_in, Path(out) / "L2.state"))
    for lv in (2, 1):
        m = marches[lv]
        if m is None:
            print(f"  level {lv}: no FINAL line"); return 1
        print(f"  level {lv}: x_r {m['xr']:.4f}  Cf(-4) {m['cf4']:.6e}  u_c(-4) {m['uc4']:.6f}  "
              f"bubble {m['bubble'][0]:.3f}-{m['bubble'][1]:.3f}  {m['steps']} steps, {m['full']} at "
              f"dt {m['dt']:g}, {m['settled']}, {m['seconds']:.0f} s")

    cf_ref = two_col(TMR / "backstep_cfl3d_cf_sst.dat")
    xr_ref = reattachment(cf_ref[:, 0], cf_ref[:, 1])
    cp_ref = two_col(TMR / "backstep_cfl3d_cp_sst.dat")
    vel = cfl3d_zones(TMR / "backstep_cfl3d_vel_sst.dat")
    cf1 = two_col(Path(out) / "L1.cf")
    cp1 = two_col(Path(out) / "L1.cp")
    prof = two_col(Path(out) / "L1.prof")

    print("\n2. the upstream layer at x = -4 (level 1)")
    cf4_ref = np.interp(-4.0, cf_ref[:, 0], cf_ref[:, 1])
    cf4 = marches[1]["cf4"]
    g2a = abs(cf4 - cf4_ref) <= 0.02 * abs(cf4_ref)
    ref = vel["x/H=-4"]
    ours = prof[np.abs(prof[:, 0] + 4.0) < 1e-9]
    m = (ref[:, 1] >= 1.0) & (ref[:, 1] <= 2.5)
    du = np.abs(np.interp(ref[m, 1], ours[:, 1], ours[:, 2]) - ref[m, 2])
    g2b = du.max() <= 0.01
    print(f"  Cf {cf4:.6e} vs CFL3D {cf4_ref:.6e} ({100 * (cf4 / cf4_ref - 1):+.2f}%, within 2%): "
          f"{'PASS' if g2a else 'FAIL'}")
    print(f"  u: largest difference {du.max():.4f} at y = {ref[m, 1][du.argmax()]:.4f} over "
          f"{m.sum()} points (within 0.01): {'PASS' if g2b else 'FAIL'}")
    ok &= g2a and g2b

    print("\n3. reattachment (level 1)")
    xr = marches[1]["xr"]
    g3 = abs(xr - xr_ref) <= 0.02 * xr_ref
    print(f"  x_r {xr:.4f} vs CFL3D {xr_ref:.4f} ({100 * (xr / xr_ref - 1):+.2f}%, within 2%): "
          f"{'PASS' if g3 else 'FAIL'}")
    print(f"  reported: level 2 {marches[2]['xr']:.4f}; the experiment 6.26 +- 0.10")
    ok &= g3

    print("\n4. the pressure recovery (level 1)")
    # Each floor on its own: CFL3D's file has the corner x = 0 on both.
    xs, dcp = [], []
    for upstream, part in zip((True, False), split_floors(cp_ref)):
        part = part[(part[:, 0] >= -4.0) & (part[:, 0] <= 30.0)]
        xs.append(part[:, 0])
        dcp.append(np.abs(floor_values(part[:, 0], upstream, cp1[:, 0], cp1[:, 1]) - part[:, 1]))
    xs, dcp = np.concatenate(xs), np.concatenate(dcp)
    g4 = dcp.max() <= 0.015
    print(f"  largest |Cp - CFL3D| {dcp.max():.4f} at x = {xs[dcp.argmax()]:.3f} over "
          f"{len(dcp)} points (within 0.015): {'PASS' if g4 else 'FAIL'}")
    ok &= g4

    print("\n5. the profiles (level 1)")
    g5 = True
    for xs in (1.0, 4.0, 6.0, 10.0):
        ref = vel[f"x/H={xs:g}"]
        ours = prof[np.abs(prof[:, 0] - xs) < 1e-9]
        m = (ref[:, 1] >= 0.0) & (ref[:, 1] <= 3.0)
        du = np.abs(np.interp(ref[m, 1], ours[:, 1], ours[:, 2]) - ref[m, 2])
        g = du.max() <= 0.03
        g5 &= g
        print(f"  x = {xs:g}: largest difference {du.max():.4f} at y = {ref[m, 1][du.argmax()]:.4f} "
              f"over {m.sum()} points (within 0.03): {'PASS' if g else 'FAIL'}")
    ok &= g5

    print("\n6. settled")
    g6 = all(mm["settled"] in ("steady", "quasi") for mm in (cal, marches[2], marches[1]))
    print(f"  level 2 (U_in = 1) {cal['settled']}, level 2 {marches[2]['settled']}, "
          f"level 1 {marches[1]['settled']}: {'PASS' if g6 else 'FAIL'}")
    ok &= g6

    print(f"\nv2b backward-facing step GATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
