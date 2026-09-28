"""
ADR-039, Python side: slip walls, gated by the Taylor-Green vortex.

    u = ( sin(pi x) cos(pi y), -cos(pi x) sin(pi y), 0 ) exp(-2 pi^2 nu t)
    p = ( cos(2 pi x) + cos(2 pi y) ) / 4 * exp(-4 pi^2 nu t)

solves Navier-Stokes exactly and, in the unit cube, meets the slip condition
on every wall: zero normal velocity, zero normal derivative of the tangential
velocity. Slip on all six faces, zero boundary flux. As revised before the
result (ADR-039):

  1. on the smooth distortion -- walls planar and axis-aligned, the cells next
     to them not -- the observed order in [1.6, 2.3] and approaching 2;
  2. on both families, at every mesh, the slip error no more than 1.1 times
     the error of the same run with the exact wall velocity prescribed.

The first version also demanded [1.85, 2.15] on the orthogonal family. Exact
Dirichlet walls miss that band just as slip does (1.03 then 1.38): on a uniform
mesh this problem's interior error is tiny and a slower wall-layer error rules
it, whatever the boundary. That convergence is reported, not gated.

Velocity boundary codes, one per boundary face, as in the C++ solver:
0 Dirichlet, 1 zero gradient, 2 slip.
"""

import sys

import numpy as np

from flux import face_average
from mesh import HexMesh
from piso import PisoSolver

PI = np.pi
SLIP = 2


def velocity(q, t, nu):
    x, y = q[:, 0], q[:, 1]
    e = np.exp(-2.0 * PI * PI * nu * t)
    return np.column_stack([np.sin(PI * x) * np.cos(PI * y),
                            -np.cos(PI * x) * np.sin(PI * y),
                            np.zeros(len(q))]) * e


def pressure(q, t, nu):
    x, y = q[:, 0], q[:, 1]
    return 0.25 * (np.cos(2 * PI * x) + np.cos(2 * PI * y)) * np.exp(-4.0 * PI * PI * nu * t)


def run(n, skew, dt=2e-4, nsteps=2, nu=0.05, nouter=6, boundary="slip"):
    m = HexMesh(n, skew=skew, seed=1, skew_mode="smooth")
    solver = PisoSolver(m, nu, dt, n_correctors=2, n_outer=nouter)
    solver.u = velocity(m.cell_centre, 0.0, nu)
    solver.u_old = solver.u.copy()
    solver.u_old2 = solver.u.copy()
    p0 = pressure(m.cell_centre, 0.0, nu)
    solver.p = p0 - p0.mean()
    uf0 = (solver.w[:, None] * solver.u[m.owner]
           + (1 - solver.w)[:, None] * solver.u[m.neigh])
    solver.F = np.einsum("ij,ij->i", uf0, m.face_area)
    solver.F_old = solver.F.copy()
    solver.F_old2 = solver.F.copy()
    nb = len(m.b_cell)
    u_b = np.zeros((nb, 3))            # unused on a slip face
    Fb = np.zeros(nb)                  # nothing crosses a slip wall
    u_bc = np.full(nb, SLIP)
    src = np.zeros((m.nc, 3))
    for k in range(nsteps):
        if boundary == "slip":
            solver.advance(u_b, Fb, src, u_bc=u_bc)
        else:                           # the control: exact wall velocity
            t = (k + 1) * dt
            solver.advance(face_average(m, lambda q: velocity(q, t, nu)), Fb, src)
    err = solver.u - velocity(m.cell_centre, nsteps * dt, nu)
    return np.sqrt((np.einsum("ij,ij->i", err, err) * m.cell_volume).sum()
                   / m.cell_volume.sum())


def main():
    ok = True
    print("Taylor-Green vortex, slip walls on all six faces (nu = 0.05, dt = 2e-4, 2 steps)")
    for skew, tag, gated in ((0.0, "orthogonal", False), (0.25, "smooth distortion", True)):
        try:
            slip = [run(n, skew) for n in (6, 12, 24)]
        except TypeError as e:
            print(f"  FAIL: the solver has no slip boundary ({e})")
            ok = False
            continue
        exact = [run(n, skew, boundary="exact") for n in (6, 12, 24)]
        so = [np.log(slip[i - 1] / slip[i]) / np.log(2.0) for i in (1, 2)]
        eo = [np.log(exact[i - 1] / exact[i]) / np.log(2.0) for i in (1, 2)]
        for n, a, b in zip((6, 12, 24), slip, exact):
            ratio = a / b
            good = ratio <= 1.1
            ok &= good
            print(f"  {tag:<18} n={n:<3} L2(u) slip {a:.6e}   exact walls {b:.6e}   "
                  f"ratio {ratio:.3f} (<= 1.1) {'ok' if good else 'TOO LARGE'}")
        msg = (f"  -> {tag}: order slip {so[0]:.3f} then {so[1]:.3f}, "
               f"exact walls {eo[0]:.3f} then {eo[1]:.3f}")
        if gated:
            good = (1.6 <= so[1] <= 2.3
                    and abs(so[1] - 2.0) <= abs(so[0] - 2.0) + 0.02)
            ok &= good
            msg += f"; slip in [1.6, 2.3] and approaching 2: {'PASS' if good else 'FAIL'}"
        else:
            msg += "  (reported, not gated)"
        print(msg)
    print("\nv2a slip-wall GATE (Python): " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
