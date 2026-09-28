"""
v2b gates, Python side: the wall distance and the k-omega SST model equations
(ADR-042, gates 1 and 2).

Written before the code they judge; run against a solver without the model
they fail with the reason.

  1. Wall distance. The three box families, walls at y = 0 and y = 1: their
     boundaries stay planar, so the exact distance is min(y, 1 - y), at every
     cell centre, to 1e-13.

  2. Manufactured solutions (sst_ms.py). The unit cube with a wall at y = 0,
     exact Dirichlet values of every field on the other faces too, both model
     variants:
       2a. frozen velocity: the exact velocity and a discretely
           divergence-free face flux; k and omega solved. MS-A (blending on,
           limiter off) and MS-B (limiter on).
       2b. coupled: u, p, k and omega solved together. MS-A.
     Orders of k and omega (and u) in [1.85, 2.15] on the orthogonal meshes,
     [1.6, 2.3] and approaching 2 on the smooth distortion, on 6/12/24 cells.
     Each run marched until the largest change per step of every field,
     relative to its largest value, is below 1e-12; no cell may be bounded.
     The properties each solution needs are checked on the exact fields at
     every mesh's cell centres first.

Run:  python3 sst_gates.py [gates]    gates: any of 1, a, b (default all)
"""

import sys
import time

import numpy as np

import ethier_steinman as ES
import sst_ms as MS
from flux import face_average, face_flux_from_potential, integrate_face_flux
from mesh import HexMesh

VARIANTS = ("2003", "1994")
GRIDS = (6, 12, 24)
FAMILIES = ((0.0, 1.85, 2.15, False, "orthogonal"),
            (0.25, 1.6, 2.3, True, "smooth distortion"))
TOL = 1e-12


# ------------------------------------------------------------------ helpers
def l2(err, vol):
    err = err if err.ndim == 2 else err[:, None]
    return np.sqrt((np.einsum("ij,ij->i", err, err) * vol).sum() / vol.sum())


def scalar_face_average(m, fn):
    return face_average(m, lambda q: fn(q)[:, None], ncomp=1)[:, 0]


def orders(errs, hs):
    return [np.log(errs[i - 1] / errs[i]) / np.log(hs[i - 1] / hs[i])
            for i in range(1, len(errs))]


def verdict(label, orders_, lo, hi, approaching=False):
    """As the v2a gates: the last order in [lo, hi]; with approaching, each
    order no further from 2 than the one before it, within 0.02."""
    last = orders_[-1]
    ok = lo <= last <= hi
    msg = f"  -> {label}: orders {', '.join(f'{o:.3f}' for o in orders_)}; last in [{lo}, {hi}]"
    if approaching and len(orders_) > 1:
        r = all(abs(orders_[i] - 2.0) <= abs(orders_[i - 1] - 2.0) + 0.02
                for i in range(1, len(orders_)))
        ok &= r
        msg += f", {'approaching 2' if r else 'MOVING AWAY FROM 2'}"
    print(msg + (": PASS" if ok else ": FAIL"), flush=True)
    return ok


def rel_change(new, old):
    return np.abs(new - old).max() / max(np.abs(new).max(), 1e-300)


# ------------------------------------------------------------ 1. wall distance
def gate_wall_distance():
    print("\n1. wall distance: walls at y = 0 and y = 1, d = min(y, 1 - y)")
    from walldist import wall_distance
    ok = True
    for skew, mode, tag in ((0.0, "smooth", "orthogonal"), (0.25, "smooth", "smooth distortion"),
                            (0.25, "warped", "perturbed")):
        for n in (6, 12):
            m = HexMesh(n, skew=skew, seed=1, skew_mode=mode)
            yb = m.b_centre[:, 1]
            wall = (np.abs(yb) < 1e-12) | (np.abs(yb - 1.0) < 1e-12)
            d = wall_distance(m, wall)
            y = m.cell_centre[:, 1]
            err = np.abs(d - np.minimum(y, 1.0 - y)).max()
            db = wall_distance(m, wall, m.b_centre)
            err_b = np.abs(db - np.minimum(yb, 1.0 - yb)).max()
            passed = err <= 1e-13 and err_b <= 1e-13
            ok &= passed
            print(f"  {tag:<18} n={n:<3} max|d - d_exact| cells {err:.1e}, boundary faces "
                  f"{err_b:.1e}  {'PASS' if passed else 'FAIL'}", flush=True)
    return ok


# ------------------------------------------------------------ 2. manufactured
def _setup(name, variant, n, skew):
    m = HexMesh(n, skew=skew, seed=1, skew_mode="smooth")
    f = MS.numeric(name, variant)
    ok, line = MS.check(name, variant, m.cell_centre)
    return m, f, ok, line


def _model(m, f, variant):
    from sst import SstModel
    wall = np.abs(m.b_centre[:, 1]) < 1e-12           # y = 0
    model = SstModel(m, f["nu"], variant=variant, wall=wall)
    nb = len(m.b_cell)
    model.set_boundary(scalar_face_average(m, f["k"]), scalar_face_average(m, f["w"]),
                       np.zeros(nb, dtype=int))        # Dirichlet everywhere
    model.set_source(f["src_k"](m.cell_centre), f["src_w"](m.cell_centre))
    model.set_state(f["k"](m.cell_centre), f["w"](m.cell_centre))
    return model


def run_frozen(name, variant, n, skew, dt=0.5, max_steps=4000):
    m, f, props, line = _setup(name, variant, n, skew)
    model = _model(m, f, variant)
    u = f["u"](m.cell_centre)
    u_b = face_average(m, f["u"])
    F, Fb = face_flux_from_potential(m, f["a"])
    change, k = np.inf, 0
    for k in range(max_steps):
        kp, wp = model.k.copy(), model.w.copy()
        model.advance_frozen(u, u_b, F, Fb, dt)
        change = max(rel_change(model.k, kp), rel_change(model.w, wp))
        if change < TOL:
            break
    ek = l2(model.k - f["k"](m.cell_centre), m.cell_volume)
    ew = l2(model.w - f["w"](m.cell_centre), m.cell_volume)
    return dict(props=props, line=line, steps=k + 1, change=change, ek=ek, ew=ew,
                bounded=model.bounded_total)


def run_coupled(variant, n, skew, dt=0.5, max_steps=4000):
    from piso import PisoSolver
    m, f, props, line = _setup("A", variant, n, skew)
    model = _model(m, f, variant)
    solver = PisoSolver(m, f["nu"], dt, n_correctors=2, n_outer=3, turbulence=model)
    cc = m.cell_centre
    solver.u = f["u"](cc)
    solver.u_old = solver.u.copy()
    solver.u_old2 = solver.u.copy()
    p0 = f["p"](cc)
    solver.p = p0 - p0.mean()
    F, _ = face_flux_from_potential(m, f["a"])
    solver.F, solver.F_old, solver.F_old2 = F.copy(), F.copy(), F.copy()
    u_b = face_average(m, f["u"])
    Fb = ES.adjust_boundary_flux(m, integrate_face_flux(m, f["u"]))
    src = f["src_u"](cc)
    change, k = np.inf, 0
    for k in range(max_steps):
        prev = (solver.u.copy(), model.k.copy(), model.w.copy())
        solver.advance(u_b, Fb, src)
        change = max(rel_change(a, b) for a, b in zip((solver.u, model.k, model.w), prev))
        if change < TOL:
            break
    eu = l2(solver.u - f["u"](cc), m.cell_volume)
    ek = l2(model.k - f["k"](cc), m.cell_volume)
    ew = l2(model.w - f["w"](cc), m.cell_volume)
    return dict(props=props, line=line, steps=k + 1, change=change, eu=eu, ek=ek, ew=ew,
                bounded=model.bounded_total)


def _study(title, runs, fields):
    """runs(n, skew) -> result dict; fields: keys of the errors to judge."""
    print(title, flush=True)
    ok = True
    for skew, lo, hi, approaching, tag in FAMILIES:
        errs = {key: [] for key in fields}
        hs = []
        for n in GRIDS:
            t0 = time.time()
            r = runs(n, skew)
            if n == GRIDS[0]:
                print(f"  {tag}: {'properties hold' if r['props'] else 'PROPERTIES VIOLATED'}: "
                      f"{r['line']}", flush=True)
            ok &= r["props"] and r["change"] < TOL and r["bounded"] == 0
            for key in fields:
                errs[key].append(r[key])
            hs.append(1.0 / n)
            print(f"  {tag:<18} n={n:<3} steps {r['steps']:<5} "
                  + "  ".join(f"L2({key[1:]}) {r[key]:.6e}" for key in fields)
                  + f"  change {r['change']:.0e}  bounded {r['bounded']}  ({time.time() - t0:.0f} s)",
                  flush=True)
        for key in fields:
            ok &= verdict(f"{key[1:]}, {tag}", orders(errs[key], hs), lo, hi, approaching)
    return ok


def gate_frozen():
    ok = True
    for name in ("A", "B"):
        for variant in VARIANTS:
            ok &= _study(f"\n2a. frozen velocity, MS-{name}, SST-{variant}",
                         lambda n, skew: run_frozen(name, variant, n, skew), ("ek", "ew"))
    return ok


def gate_coupled():
    ok = True
    for variant in VARIANTS:
        ok &= _study(f"\n2b. coupled, MS-A, SST-{variant}",
                     lambda n, skew: run_coupled(variant, n, skew), ("eu", "ek", "ew"))
    return ok


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "1ab"
    gates = {"1": gate_wall_distance, "a": gate_frozen, "b": gate_coupled}
    ok = True
    for key in which:
        try:
            ok &= gates[key]()
        except (ImportError, TypeError, AttributeError) as e:
            print(f"  FAIL: the solver lacks what this gate needs ({type(e).__name__}: {e})")
            ok = False
    print("\nv2b SST GATE (Python): " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
