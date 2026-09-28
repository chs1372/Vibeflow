"""
ADR-039, Python side: slip walls, gated by the Taylor-Green vortex.

    u = ( sin(pi x) cos(pi y), -cos(pi x) sin(pi y), 0 ) exp(-2 pi^2 nu t)
    p = ( cos(2 pi x) + cos(2 pi y) ) / 4 * exp(-4 pi^2 nu t)

solves Navier-Stokes exactly and, in the unit cube, meets the slip condition
on every wall: zero normal velocity, zero normal derivative of the tangential
velocity. With slip on all six faces and zero boundary flux, the spatial order
must be second: [1.85, 2.15] on the orthogonal family, [1.6, 2.3] and rising on
the smooth distortion, whose walls stay planar and axis-aligned while the
cells next to them do not.

Velocity boundary codes, one per boundary face, as in the C++ solver:
0 Dirichlet, 1 zero gradient, 2 slip.
"""

import sys

import numpy as np

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


def run(n, skew, dt=2e-4, nsteps=2, nu=0.05, nouter=6):
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
    for _ in range(nsteps):
        solver.advance(u_b, Fb, src, u_bc=u_bc)
    err = solver.u - velocity(m.cell_centre, nsteps * dt, nu)
    return np.sqrt((np.einsum("ij,ij->i", err, err) * m.cell_volume).sum()
                   / m.cell_volume.sum())


def main():
    ok = True
    print("Taylor-Green vortex, slip walls on all six faces (nu = 0.05, dt = 2e-4, 2 steps)")
    for skew, lo, hi, rising, tag in ((0.0, 1.85, 2.15, False, "orthogonal"),
                                      (0.25, 1.6, 2.3, True, "smooth distortion")):
        try:
            errs = [run(n, skew) for n in (6, 12, 24)]
        except TypeError as e:
            print(f"  FAIL: the solver has no slip boundary ({e})")
            ok = False
            continue
        orders = [np.log(errs[i - 1] / errs[i]) / np.log(2.0) for i in (1, 2)]
        for n, e in zip((6, 12, 24), errs):
            print(f"  {tag:<18} n={n:<3} L2(u) {e:.6e}")
        good = lo <= orders[-1] <= hi
        msg = f"  -> {tag}: order {orders[-1]:.3f} in [{lo}, {hi}]"
        if rising:
            r = orders[1] >= orders[0] - 0.02
            good &= r
            msg += f", trend {'rising' if r else 'FALLING'}"
        print(msg + (": PASS" if good else ": FAIL"))
        ok &= good
    print("\nv2a slip-wall GATE (Python): " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
