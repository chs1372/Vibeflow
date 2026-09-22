"""
v1 gate: manufactured-solution convergence of the convection-diffusion operator.

    div(u phi) - div(gamma grad phi) = f

u comes from a vector potential, so the prescribed mass flux is divergence-free
to machine precision (see flux.py). Without that, the discrete divergence error
multiplies phi in the convective term and shows up as a loss of order that
looks like a bug in the scheme.

The manufactured solution is non-zero on every boundary, so the convective
boundary flux is exercised rather than silently zero.

Gate: second order on both meshes, at a cell Peclet number high enough that
convection actually dominates -- at gamma = 0.02 the coarse mesh runs around
Pe = 15, so a first-order upwind scheme would be plainly visible.
"""

import sys

import numpy as np

from mesh import HexMesh
from flux import face_flux_from_potential, discrete_divergence, SolenoidalField as SF
from transport import ConvectionDiffusion

PI = np.pi
GAMMA = 0.02


def phi_exact(p):
    return np.exp(p[:, 0] + 0.5 * p[:, 1]) * np.sin(PI * p[:, 2] + 0.3)


def grad_exact(p):
    e = np.exp(p[:, 0] + 0.5 * p[:, 1])
    s = np.sin(PI * p[:, 2] + 0.3)
    c = np.cos(PI * p[:, 2] + 0.3)
    return np.column_stack([e * s, 0.5 * e * s, PI * e * c])


def source(p):
    # div(u phi) = u . grad(phi) because div(u) = 0
    conv = np.einsum("ij,ij->i", SF.velocity(p), grad_exact(p))
    # laplacian(phi) = (1 + 1/4 - pi^2) phi
    return conv + GAMMA * (PI ** 2 - 1.25) * phi_exact(p)


def run(n, skew, seed=1, mode="smooth"):
    m = HexMesh(n, skew=skew, seed=seed, skew_mode=mode)
    fi, fb = face_flux_from_potential(m, SF.potential)
    div = np.abs(discrete_divergence(m, fi, fb)).max()
    op = ConvectionDiffusion(m, GAMMA, fi, fb)
    phi, iters, converged = op.solve(source(m.cell_centre), phi_exact(m.b_centre), relax=0.7)
    err = phi - phi_exact(m.cell_centre)
    l2 = np.sqrt((err ** 2 * m.cell_volume).sum() / m.cell_volume.sum())
    return dict(n=n, h=1.0 / n, l2=l2, linf=np.abs(err).max(), iters=iters,
                converged=converged, pe=op.peclet(), div=div,
                nonortho=m.non_orthogonality())


def study(skew, grids, mode="smooth"):
    rows = [run(n, skew, mode=mode) for n in grids]
    for i in range(1, len(rows)):
        rows[i]["order"] = (np.log(rows[i - 1]["l2"] / rows[i]["l2"])
                            / np.log(rows[i - 1]["h"] / rows[i]["h"]))
    return rows


def report(label, rows, gate=(1.85, 2.15)):
    # "non-orthogonality" contains the substring "orthogonal", which a naive
    # parser reads as the orthogonal case. Keep the word out of this line.
    print(f"\n{label}   (skewness {rows[0]['nonortho']:.2f} deg, "
          f"max discrete div {max(r['div'] for r in rows):.1e})")
    print(f"  {'N':>4} {'h':>9} {'L2 error':>19} {'Linf':>13} {'order':>8} "
          f"{'maxPe':>8} {'corr it':>8}")
    last = 0.0
    for i, r in enumerate(rows):
        o = f"{r['order']:8.3f}" if "order" in r else f"{'-':>8}"
        if "order" in r:
            last = r["order"]
        print(f"  {r['n']:4d} {r['h']:9.5f} {r['l2']:19.12e} {r['linf']:13.6e} {o} "
              f"{r['pe']:8.2f} {r['iters']:8d}{'' if r['converged'] else '  NOT CONVERGED'}")
    ok = gate[0] <= last <= gate[1] and all(r["converged"] for r in rows)
    print(f"  -> order {last:.3f} in [{gate[0]}, {gate[1]}]: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    grids = [int(a) for a in sys.argv[1:]] or [8, 16, 32]
    ok = True
    for skew, tag in ((0.0, "orthogonal"), (0.25, "distorted")):
        ok &= report(f"convection-diffusion (gamma={GAMMA}) / {tag} mesh", study(skew, grids))
    print()
    print("v1 convection MMS GATE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
