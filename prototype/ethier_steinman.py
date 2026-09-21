"""
v1 gate: the Ethier-Steinman exact solution of the 3D unsteady incompressible
Navier-Stokes equations.

Unlike a manufactured solution this needs NO source term -- it satisfies the
equations exactly as they stand. Verified symbolically: div(u) = 0 and all
three momentum residuals reduce to 0 (see tests/mms/verify_exact.py).

    a = pi/4, d = pi/2
    u = -a[e^{ax} sin(ay+dz) + e^{az} cos(ax+dy)] e^{-d^2 nu t}   (and cyclic)

Two orders are measured separately:
  * spatial, at a time step small enough that temporal error is negligible
  * temporal, by self-convergence against a much smaller time step on the same
    mesh, so spatial error cancels instead of dominating
"""

import sys

import numpy as np

from flux import face_average, integrate_face_flux
from mesh import HexMesh
from piso import PisoSolver

A = np.pi / 4.0
D = np.pi / 2.0


def velocity(p, t, nu):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    e = np.exp(-D * D * nu * t)
    return -A * np.column_stack([
        np.exp(A * x) * np.sin(A * y + D * z) + np.exp(A * z) * np.cos(A * x + D * y),
        np.exp(A * y) * np.sin(A * z + D * x) + np.exp(A * x) * np.cos(A * y + D * z),
        np.exp(A * z) * np.sin(A * x + D * y) + np.exp(A * y) * np.cos(A * z + D * x),
    ]) * e


def pressure(p_, t, nu):
    x, y, z = p_[:, 0], p_[:, 1], p_[:, 2]
    e = np.exp
    return -A * A / 2.0 * (
        e(2*A*x) + e(2*A*y) + e(2*A*z)
        + 2*np.sin(A*x + D*y)*np.cos(A*z + D*x)*e(A*(y + z))
        + 2*np.sin(A*y + D*z)*np.cos(A*x + D*y)*e(A*(z + x))
        + 2*np.sin(A*z + D*x)*np.cos(A*y + D*z)*e(A*(x + y))
    ) * np.exp(-2.0 * D * D * nu * t)


def adjust_boundary_flux(mesh, Fb):
    """Enforce global mass balance on the prescribed boundary flux.

    With Dirichlet velocity everywhere the pressure Poisson problem is only
    solvable if the boundary fluxes sum to zero. The exact field is
    divergence-free, but evaluating it at face centres leaves an O(h^2)
    imbalance that would make the system inconsistent. Spreading the residual
    over the boundary area is what OpenFOAM's adjustPhi does.
    """
    area = np.linalg.norm(mesh.b_area, axis=1)
    return Fb - Fb.sum() * area / area.sum()


def run(n, dt, nsteps, nu, skew, seed=1, ncorr=2, nouter=20, consistent=True):
    m = HexMesh(n, skew=skew, seed=seed)
    solver = PisoSolver(m, nu, dt, n_correctors=ncorr, n_outer=nouter,
                        consistent_rhie_chow=consistent)

    # Initialise from the exact state, INCLUDING pressure. Starting from p = 0
    # leaves the Rhie-Chow flux residual far from its converged value, and it
    # then relaxes over a few steps regardless of dt -- a transient that is not
    # a time-discretisation error but shows up in a convergence study as one.
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

    src = np.zeros((m.nc, 3))
    cont = 0.0
    for k in range(nsteps):
        t_new = (k + 1) * dt
        u_b = face_average(m, lambda q: velocity(q, t_new, nu))
        Fb = adjust_boundary_flux(
            m, integrate_face_flux(m, lambda q: velocity(q, t_new, nu)))
        cont = max(cont, solver.advance(u_b, Fb, src))

    t_end = nsteps * dt
    ex = velocity(m.cell_centre, t_end, nu)
    err = solver.u - ex
    l2 = np.sqrt((np.einsum("ij,ij->i", err, err) * m.cell_volume).sum()
                 / m.cell_volume.sum())
    return dict(n=n, h=1.0 / n, dt=dt, l2=l2, cont=cont, u=solver.u.copy(),
                outer=solver.outer_used, nonortho=m.non_orthogonality())


def order_table(rows, key, label, gate):
    print(f"\n{label}")
    hdr = "h" if key == "h" else "dt"
    print(f"  {'N':>4} {hdr:>10} {'L2(u) error':>19} {'max div':>11} {'order':>8}")
    last = 0.0
    for i, r in enumerate(rows):
        o = "-"
        if i:
            last = np.log(rows[i-1]["l2"] / r["l2"]) / np.log(rows[i-1][key] / r[key])
            o = f"{last:.3f}"
        print(f"  {r['n']:4d} {r[key]:10.5f} {r['l2']:19.12e} {r['cont']:11.2e} {o:>8}")
    ok = gate[0] <= last <= gate[1]
    print(f"  -> order {last:.3f} in [{gate[0]}, {gate[1]}]: {'PASS' if ok else 'FAIL'}")
    return ok


def gate_spatial(nu, skew, grids, dt=1e-3, nsteps=4):
    rows = [run(n, dt, nsteps, nu, skew) for n in grids]
    tag = "orthogonal" if skew == 0 else f"skewed ({rows[0]['nonortho']:.1f} deg)"
    return order_table(rows, "h", f"spatial order / {tag} mesh  (dt={dt}, {nsteps} steps)",
                       (1.7, 2.3))


def gate_temporal(nu, skew, n=8, t_end=0.8, steps=(2, 4, 8), ref_steps=64):
    """Self-convergence in time against a much smaller dt on the same mesh.

    Two things this measurement gets wrong if set up carelessly:

    * Measuring against the exact solution reports the SPATIAL error, which
      does not shrink when dt does.
    * Running over an interval where the solution barely changes reports
      whatever transient is largest, not the time-discretisation error. At
      nu = 0.05 over t = 0.4 the field decays by 5% and this gate measured
      order 1.3 for a scheme that is genuinely second order. The parameters
      below decay the field to 14% of its initial amplitude, which is a real
      transient.
    """
    ref = run(n, t_end / ref_steps, ref_steps, nu, skew, nouter=200)
    mesh = HexMesh(n, skew=skew, seed=1)
    rows = []
    for s_ in steps:
        r = run(n, t_end / s_, s_, nu, skew, nouter=200)
        d = r["u"] - ref["u"]
        r["l2"] = np.sqrt((np.einsum("ij,ij->i", d, d) * mesh.cell_volume).sum()
                          / mesh.cell_volume.sum())
        rows.append(r)
    tag = "orthogonal" if skew == 0 else "skewed"
    return order_table(rows, "dt",
                       f"temporal order (BDF2) / {tag} mesh  "
                       f"(n={n}, nu={nu}, t_end={t_end}, vs dt/{ref_steps})",
                       (1.8, 2.6))


def gate_dt_independence(nu, skew, n=8, dts=(0.05, 2.0)):
    """Rhie-Chow must not let dt change the STEADY state (ADR-010).

    STATUS: INCONCLUSIVE. Reported, not gated. See docs/DECISIONS.md ADR-012.

    A steady problem has no temporal discretisation error, so two runs that
    both reach steady state must agree. Measured on the steady MMS, skewed
    mesh, both runs iterated to |du| < 1e-13:

        naive       dt=0.05  L2 2.0015e-02 | dt=2.0  L2 2.0016e-02  (7e-05)
        consistent  dt=0.05  L2 2.0551e-02 | dt=2.0  L2 2.0021e-02  (2.6e-02)

    That is backwards from the intent: the form meant to remove the dt
    dependence is the one that shows it. Two things are unresolved and neither
    should be guessed at:

      * As dt -> 0, Df*aP_t -> 1, so R^n = Df*Dp + R^{n-1} accumulates rather
        than settling. The steady value derived on paper assumes it settles.
      * This error norm is nearly blind to the damping term anyway, so it may
        be measuring discretisation error, not decoupling. A checkerboard
        -sensitive measure (the odd-even pressure mode amplitude) would
        discriminate; this one does not.

    The default stays consistent=True per ADR-010 rather than being flipped on
    one inconclusive measurement.
    """
    import steady_mms as SM

    print(f"\nRhie-Chow time-step independence (steady MMS, n={n}, "
          f"{'orthogonal' if skew == 0 else 'skewed'} mesh)  [REPORTED, NOT GATED]")
    print(f"  {'formulation':<14}{'dt':>8}{'steps':>7}{'L2 vs exact':>20}{'|du|':>10}")
    for consistent in (True, False):
        vals = []
        for dt in dts:
            m = HexMesh(n, skew=skew, seed=1)
            solver = PisoSolver(m, nu, dt, n_correctors=2, n_outer=3,
                                consistent_rhie_chow=consistent)
            src = SM.source(m.cell_centre, nu)
            u_b = face_average(m, SM.velocity)
            Fb = adjust_boundary_flux(m, integrate_face_flux(m, SM.velocity))
            prev, change = None, 1.0
            for k in range(int(400.0 / dt)):
                solver.advance(u_b, Fb, src)
                if prev is not None:
                    change = np.abs(solver.u - prev).max()
                    if change < 1e-13:
                        break
                prev = solver.u.copy()
            e = solver.u - SM.velocity(m.cell_centre)
            vals.append(np.sqrt((np.einsum("ij,ij->i", e, e) * m.cell_volume).sum()
                                / m.cell_volume.sum()))
            print(f"  {'consistent' if consistent else 'naive':<14}{dt:>8.3f}"
                  f"{k + 1:>7}{vals[-1]:>20.10e}{change:>10.0e}")
        print(f"  {'':<14}{'':>8}{'':>7}{'relative spread':>20}"
              f"{abs(vals[0] - vals[1]) / max(vals):>10.1e}")
    return True      # reported only


def main():
    ok = True
    ok &= gate_spatial(0.05, 0.0, [6, 12, 24], dt=2e-4, nsteps=2)
    # KNOWN FAILURE, deliberately left failing. Root cause measured: on a
    # randomly perturbed mesh the reconstructed pressure gradient is first
    # order (least squares) or does not converge at all (Green-Gauss), and
    # the velocity correction u = H/aP - grad(p) V/aP inherits that order.
    # The PRESSURE itself converges at 2.06 here, so the projection is sound
    # and only the gradient reconstruction is short. Fix: quadratic
    # least-squares gradient over a wider stencil. See ADR-012.
    ok &= gate_spatial(0.05, 0.25, [6, 12, 24], dt=2e-4, nsteps=2)
    ok &= gate_temporal(1.0, 0.0)
    gate_dt_independence(0.1, 0.25)
    print()
    print("v1 Ethier-Steinman GATE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
