"""
v0 verification gate: manufactured-solution convergence of the diffusion operator.

Two manufactured solutions are required, not one:

  A) u = sin(pi x) sin(pi y) sin(pi z)        -- vanishes on the whole boundary
  B) u = exp(x + y/2) sin(pi z + 3/10)        -- non-zero on every boundary face

Case A alone is not a valid gate: its Dirichlet term contributes nothing, so a
broken boundary coefficient passes unnoticed. Case B exercises it.

Gate: observed L2 order must be 2.0 +/- 0.1 for BOTH solutions on BOTH an
orthogonal mesh and a skewed mesh. A drop toward 1.5 on the skewed mesh means
the non-orthogonal correction is wrong.
"""

import sys
import numpy as np
from mesh import HexMesh
from fvm import DiffusionOperator

PI = np.pi


class Manufactured:
    def __init__(self, name, u, f):
        self.name, self.u, self.f = name, u, f


def _sine(p):
    return np.sin(PI * p[:, 0]) * np.sin(PI * p[:, 1]) * np.sin(PI * p[:, 2])


def _expsin(p):
    return np.exp(p[:, 0] + 0.5 * p[:, 1]) * np.sin(PI * p[:, 2] + 0.3)


CASES = [
    # -laplacian(u) = 3 pi^2 u
    Manufactured("A: zero boundary", _sine, lambda p: 3.0 * PI ** 2 * _sine(p)),
    # laplacian(u) = (1 + 1/4 - pi^2) u  ->  -laplacian(u) = (pi^2 - 5/4) u
    Manufactured("B: non-zero boundary", _expsin,
                 lambda p: (PI ** 2 - 1.25) * _expsin(p)),
]


def run(case, n, skew, seed=1):
    m = HexMesh(n, skew=skew, seed=seed)
    op = DiffusionOperator(m, gamma=1.0)
    u, iters = op.solve(case.f(m.cell_centre), case.u(m.b_centre))
    ex = case.u(m.cell_centre)
    err = u - ex
    l2 = np.sqrt((err ** 2 * m.cell_volume).sum() / m.cell_volume.sum())
    return dict(n=n, h=1.0 / n, l2=l2, linf=np.abs(err).max(), iters=iters,
                umax=np.abs(ex).max(), nonortho=m.non_orthogonality())


def study(case, skew, grids):
    rows = [run(case, n, skew) for n in grids]
    for i in range(1, len(rows)):
        rows[i]["order"] = (np.log(rows[i - 1]["l2"] / rows[i]["l2"])
                            / np.log(rows[i - 1]["h"] / rows[i]["h"]))
    return rows


def report(label, rows, gate=(1.9, 2.1)):
    print(f"\n{label}   (max non-orthogonality {rows[0]['nonortho']:.2f} deg, "
          f"|u|max {rows[0]['umax']:.3f})")
    print(f"  {'N':>4} {'h':>9} {'L2 error':>19} {'Linf':>13} {'order':>8} {'corr it':>8}")
    for r in rows:
        o = f"{r['order']:8.3f}" if "order" in r else f"{'-':>8}"
        print(f"  {r['n']:4d} {r['h']:9.5f} {r['l2']:19.12e} {r['linf']:13.6e} {o} {r['iters']:8d}")
    last = rows[-1]["order"]
    ok = gate[0] <= last <= gate[1]
    print(f"  -> order {last:.3f} in [{gate[0]}, {gate[1]}]: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    grids = [int(a) for a in sys.argv[1:]] or [8, 16, 32]
    ok = True
    for case in CASES:
        for skew, tag in ((0.0, "orthogonal"), (0.25, "skewed")):
            ok &= report(f"{case.name} / {tag} mesh", study(case, skew, grids))
    print()
    print("v0 MMS GATE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
