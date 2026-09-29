#!/usr/bin/env python3
"""ADR-045's reported, not gated, quantities for the backward-facing step.

Reads the marches backstep_gate.py left in its output directory (L2_cal, L2,
L1: their logs, .cf, .cp and .prof) and prints, beside CFL3D's SST results
and Driver & Seegmiller's experiment as TMR publishes them:

  - U_in, the core velocity at (-4, 5), the steps and the cost of each march;
  - the reattachment point on levels 2 and 1, CFL3D's and the experiment's;
  - the corner bubble under the step (the first two sign changes of Cf
    behind it, the harness's definition, applied to CFL3D's Cf as well);
  - Cf and Cp on the bottom wall at the experiment's stations;
  - u at the experiment's points at x = -4, 1, 4, 6 and 10.

Run:  backstep_report.py [out dir, default build/backstep]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backstep_gate import floor_values, parse_final  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TMR = ROOT / "cases" / "backstep" / "tmr"
XR_EXP, XR_EXP_ERR = 6.26, 0.10
STATIONS = (-4.0, 1.0, 4.0, 6.0, 10.0)


def rows(path):
    out = []
    for ln in open(path):
        s = ln.strip()
        if not s or s.startswith("#") or s[0].isalpha() or s.startswith('"'):
            continue
        out.append([float(v) for v in s.replace(",", " ").split()])
    return np.array(out)


def zones(path):
    """Tecplot-style zones: {title: rows}."""
    zs, cur = {}, None
    for ln in open(path):
        s = ln.strip()
        if s.lower().startswith("zone"):
            cur = s.split('"')[1]
            zs[cur] = []
        elif cur and s and not s.startswith("#") and not s[0].isalpha():
            zs[cur].append([float(v) for v in s.split()])
    return {k: np.array(v) for k, v in zs.items()}


def separation(x, cf):
    """backstep.cpp's: x_r the last change from negative to non-negative
    behind the step, the bubble the first two sign changes of three or more."""
    zs, xr = [], None
    for i in range(1, len(x)):
        if x[i - 1] < 0.0:
            continue
        if (cf[i - 1] < 0.0) != (cf[i] < 0.0):
            zs.append(x[i - 1] - cf[i - 1] * (x[i] - x[i - 1]) / (cf[i] - cf[i - 1]))
        if cf[i - 1] < 0.0 <= cf[i]:
            xr = x[i - 1] - cf[i - 1] * (x[i] - x[i - 1]) / (cf[i] - cf[i - 1])
    bubble = (zs[0], zs[1]) if len(zs) >= 3 else None
    return xr, bubble


def march(out, name):
    log = Path(out) / f"{name}.log"
    if not log.exists():
        return None
    m = {"name": name}
    for ln in open(log):
        if ln.startswith("backward-facing step"):
            m["u_in"] = float(ln.split("U_in")[1])
        if ln.startswith("FINAL"):
            d = parse_final(ln)
            m.update(xr=float(d["xr"]), uc4=float(d["uc4"]), steps=int(d["steps"]),
                     full=int(d["full"]), settled=d["settled"], seconds=float(d["seconds"]))
    return m if "xr" in m else None


def main(argv):
    out = Path(argv[1]) if len(argv) > 1 else ROOT / "build" / "backstep"
    cf_ref = rows(TMR / "backstep_cfl3d_cf_sst.dat")
    cp_ref = rows(TMR / "backstep_cfl3d_cp_sst.dat")
    vel_ref = zones(TMR / "backstep_cfl3d_vel_sst.dat")
    cf_exp = rows(TMR / "cf.exp.dat")
    cp_exp = zones(TMR / "cp.expnew.dat")["bottom wall"]
    prof_exp = zones(TMR / "profiles.exp.dat")
    xr_ref, bub_ref = separation(cf_ref[:, 0], cf_ref[:, 1])

    print("the marches (U_in, the core velocity at (-4, 5), steps, cost)")
    ms = [march(out, n) for n in ("L2_cal", "L2", "L1")]
    for m in ms:
        if m:
            print(f"  {m['name']:7s} U_in {m['u_in']:.6f}  u_c(-4) {m['uc4']:.6f}  {m['steps']} steps "
                  f"({m['full']} at the full dt), {m['settled']}, {m['seconds']:.0f} s")
    print("  CFL3D's u_c(-4): 0.998")

    levels = [(lv, f"L{lv}") for lv in (2, 1) if (out / f"L{lv}.cf").exists()]
    data = {}
    for lv, name in levels:
        cf = rows(out / f"{name}.cf")
        data[lv] = {"cf": cf, "cp": rows(out / f"{name}.cp"), "prof": rows(out / f"{name}.prof"),
                    "sep": separation(cf[:, 0], cf[:, 1])}

    print("\nreattachment and the corner bubble")
    for lv in data:
        xr, b = data[lv]["sep"]
        bs = f"{b[0]:.4f}-{b[1]:.4f}" if b else "none"
        print(f"  level {lv}: x_r {xr:.4f} ({100 * (xr / xr_ref - 1):+.2f}% of CFL3D, "
              f"{xr - XR_EXP:+.3f} from the experiment); bubble {bs}")
    print(f"  CFL3D: x_r {xr_ref:.4f}; bubble {bub_ref[0]:.4f}-{bub_ref[1]:.4f}")
    print(f"  experiment: x_r {XR_EXP} +- {XR_EXP_ERR}")
    if 1 in data and 2 in data:
        print(f"  level 2 - level 1: {data[2]['sep'][0] - data[1]['sep'][0]:+.4f}")

    print("\nCf on the bottom wall at the experiment's stations (x, exp +- err, CFL3D, levels)")
    hdr = "".join(f"  level {lv:<3d}" for lv in data)
    print(f"  {'x':>7s}  {'exp':>9s} {'err':>8s}  {'CFL3D':>9s}{hdr}")
    inside = {lv: 0 for lv in data}
    for x, c, e in cf_exp:
        ref = np.interp(x, cf_ref[:, 0], cf_ref[:, 1])
        line = f"  {x:7.3f}  {c:9.2e} {e:8.2e}  {ref:9.2e}"
        for lv in data:
            v = floor_values([x], x < 0.0, data[lv]["cf"][:, 0], data[lv]["cf"][:, 1])[0]
            inside[lv] += abs(v - c) <= e
            line += f"  {v:9.2e}"
        print(line)
    for lv in data:
        print(f"  level {lv}: {inside[lv]} of {len(cf_exp)} stations within the experiment's error")

    print("\nCp on the bottom wall against the experiment (shifted to 0 near x = 40)")
    for lv in data:
        cp = data[lv]["cp"]
        xe = cp_exp[:, 0]
        d = np.where(xe < 0.0, floor_values(xe, True, cp[:, 0], cp[:, 1]),
                     floor_values(xe, False, cp[:, 0], cp[:, 1])) - cp_exp[:, 1]
        dr = np.interp(cp_exp[:, 0], cp_ref[:, 0], cp_ref[:, 1]) - cp_exp[:, 1]
        i = np.argmax(np.abs(d))
        print(f"  level {lv}: largest |Cp - exp| {abs(d[i]):.4f} at x = {cp_exp[i, 0]:.2f}, "
              f"rms {np.sqrt(np.mean(d ** 2)):.4f} over {len(d)} stations "
              f"(CFL3D: largest {np.abs(dr).max():.4f}, rms {np.sqrt(np.mean(dr ** 2)):.4f})")

    print("\nu against the experiment (the experiment's points; CFL3D beside)")
    for xs in STATIONS:
        ez = next(v for k, v in prof_exp.items() if k.endswith(f"x/H={xs:g}"))
        ye, ue = ez[:, 1], ez[:, 2]
        rz = vel_ref[f"x/H={xs:g}"]
        dr = np.interp(ye, rz[:, 1], rz[:, 2]) - ue
        line = f"  x = {xs:5g}: {len(ye)} points, CFL3D largest {np.abs(dr).max():.3f} rms {np.sqrt(np.mean(dr ** 2)):.3f}"
        for lv in data:
            p = data[lv]["prof"]
            q = p[np.abs(p[:, 0] - xs) < 1e-9]
            d = np.interp(ye, q[:, 1], q[:, 2]) - ue
            line += f";  level {lv} largest {np.abs(d).max():.3f} rms {np.sqrt(np.mean(d ** 2)):.3f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
