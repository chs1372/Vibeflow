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


def run(n, dt, nsteps, nu, skew, seed=1, ncorr=2, nouter=20, consistent=True,
        skew_mode="planar"):
    m = HexMesh(n, skew=skew, seed=seed, skew_mode=skew_mode)
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


def order_table(rows, key, label, gate, require_rising=False):
    print(f"\n{label}")
    hdr = "h" if key == "h" else "dt"
    print(f"  {'N':>4} {hdr:>10} {'L2(u) error':>19} {'max div':>11} {'order':>8}")
    orders = []
    for i, r in enumerate(rows):
        o = "-"
        if i:
            orders.append(np.log(rows[i-1]["l2"] / r["l2"]) / np.log(rows[i-1][key] / r[key]))
            o = f"{orders[-1]:.3f}"
        print(f"  {r['n']:4d} {r[key]:10.5f} {r['l2']:19.12e} {r['cont']:11.2e} {o:>8}")
    last = orders[-1] if orders else 0.0
    ok = gate[0] <= last <= gate[1]
    msg = f"  -> order {last:.3f} in [{gate[0]}, {gate[1]}]"
    if require_rising and len(orders) > 1:
        # The mesh family is still approaching its limiting geometry at these
        # resolutions, so the order approaches 2 from below rather than
        # sitting on it. A REVERSAL is the regression signal that matters: it
        # means a new first-order term has appeared.
        rising = all(b >= a - 0.02 for a, b in zip(orders, orders[1:]))
        ok = ok and rising
        msg += f", trend {'rising' if rising else 'FALLING'}"
    print(msg + f": {'PASS' if ok else 'FAIL'}")
    return ok


def gate_spatial(nu, skew, grids, dt=2e-4, nsteps=2, mode="smooth",
                 gate=(1.7, 2.3), rising=False):
    rows = [run(n, dt, nsteps, nu, skew, skew_mode=mode, nouter=6) for n in grids]
    if skew:
        q = HexMesh(grids[-1], skew=skew, seed=1, skew_mode=mode)
        tag = f"{mode} distortion, {q.non_orthogonality():.0f} deg non-orth"
    else:
        tag = "orthogonal"
    return order_table(rows, "h",
                       f"spatial order / {tag}  (dt={dt}, {nsteps} steps)",
                       gate, require_rising=rising)


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


def steady_state(n, skew, nu, dt, with_pressure, tol=1e-13, max_time=400.0,
                 **solver_kw):
    """Run the steady MMS from rest until u and p stop changing (max |change|
    per step below tol). Returns the solver, the step count and the last
    change, which the caller checks against tol."""
    import steady_mms as SM
    m = HexMesh(n, skew=skew, seed=1)
    solver = PisoSolver(m, nu, dt, n_correctors=2, n_outer=3, **solver_kw)
    src = SM.source(m.cell_centre, nu, with_pressure)
    u_b = face_average(m, SM.velocity)
    Fb = adjust_boundary_flux(m, integrate_face_flux(m, SM.velocity))
    prev_u, prev_p, change, steps = None, None, np.inf, 0
    for k in range(int(round(max_time / dt))):
        solver.advance(u_b, Fb, src)
        steps = k + 1
        if prev_u is not None:
            change = max(np.abs(solver.u - prev_u).max(),
                         np.abs(solver.p - prev_p).max())
            if change < tol:
                break
        prev_u, prev_p = solver.u.copy(), solver.p.copy()
    return solver, steps, change


def gate_dt_independence(nu=0.1, n=8, dts=(0.02, 0.2, 2.0), bound=1e-6,
                         report_forms=(("v1 old-flux form", {"old_flux": "v1"}),
                                       ("naive, v1 structure",
                                        {"consistent_rhie_chow": False,
                                         "old_flux": "v1"}))):
    """Rhie-Chow must not let dt change the STEADY state (ADR-010, ADR-037).

    A steady problem has no temporal discretisation error: the BDF2 terms
    cancel exactly once u stops changing. Two runs of the same steady problem
    with different time steps must therefore reach the same discrete state,
    and whatever separates them is the time step leaking into the spatial
    discretisation through the face flux.

    Two steady manufactured problems -- constant pressure, and a smooth
    non-constant pressure that makes the pressure-damping part of the flux
    work -- on an orthogonal and a randomly skewed mesh, each run to
    max |du|, |dp| < 1e-13 at dt = 0.02, 0.2 and 2.0. The spread is the
    largest difference between two of those steady states, in the L2 norm,
    relative to the discretisation error of the dt = 2.0 run: for u against
    the exact velocity, for p (mean removed) against the exact pressure.

    Bound: 1e-6. A formulation whose steady equations contain no dt reaches
    the same state to the iteration tolerance, about 1e-10 here; one that
    leaks dt anywhere shows up far above it. Measured when the gate was
    written, before any fix (ADR-037), spread of u / spread of p:

                              constant p            grad p
      v1 old-flux form
        orthogonal         8.1e-04 / 9.9e-04    4.8e-03 / 7.1e-03
        skewed             6.9e-01 / 1.5e+00    6.6e-01 / 1.5e+00
      naive (no old flux)
        orthogonal         2.1e-02 / 3.6e-02    9.4e-02 / 1.5e-01
        skewed             6.7e-02 / 1.8e-01    1.1e-01 / 2.3e-01

    On the skewed mesh the v1 form moves the steady velocity by 69% of its
    own discretisation error between dt = 0.02 and 2.0.

    Only the default formulation is gated. The forms in report_forms are
    printed beside it for the record.
    """
    import steady_mms as SM

    def norm(v, vol):
        v = v if v.ndim == 2 else v[:, None]
        return np.sqrt((np.einsum("ij,ij->i", v, v) * vol).sum() / vol.sum())

    def spreads(states, m, with_pressure):
        vol = m.cell_volume
        u_ex = SM.velocity(m.cell_centre)
        p_ex = SM.pressure(m.cell_centre, with_pressure)
        centred = lambda p: p - (p * vol).sum() / vol.sum()
        ref_u, ref_p = states[-1]
        eu = norm(ref_u - u_ex, vol)
        ep = norm(centred(ref_p) - centred(p_ex), vol)
        su = sp_ = 0.0
        for a in range(len(states)):
            for b in range(a + 1, len(states)):
                su = max(su, norm(states[a][0] - states[b][0], vol) / eu)
                sp_ = max(sp_, norm(centred(states[a][1]) - centred(states[b][1]), vol) / ep)
        return su, sp_, eu, ep

    print(f"\nRhie-Chow steady state independent of dt (steady MMS, n={n}, nu={nu}, "
          f"dt = {', '.join(str(d) for d in dts)})")
    print(f"  {'problem':<16}{'mesh':<12}{'formulation':<28}{'steps':>18}"
          f"{'spread u':>11}{'spread p':>11}{'L2 u':>10}{'L2 p':>10}")
    ok = True
    forms = (("default", {}),) + tuple(report_forms)
    for with_pressure in (False, True):
        for skew in (0.0, 0.25):
            for name, kw in forms:
                states, steps, conv = [], [], True
                for dt in dts:
                    solver, k, change = steady_state(n, skew, nu, dt, with_pressure, **kw)
                    states.append((solver.u.copy(), solver.p.copy()))
                    steps.append(k)
                    conv &= change < 1e-13
                su, sp_, eu, ep = spreads(states, solver.m, with_pressure)
                gated = name == "default"
                verdict = ""
                if gated:
                    passed = conv and su <= bound and sp_ <= bound
                    ok &= passed
                    verdict = "  PASS" if passed else ("  FAIL" if conv else "  FAIL (not steady)")
                print(f"  {'grad p' if with_pressure else 'constant p':<16}"
                      f"{'skewed' if skew else 'orthogonal':<12}{name:<28}"
                      f"{'/'.join(str(k) for k in steps):>18}{su:>11.1e}{sp_:>11.1e}"
                      f"{eu:>10.2e}{ep:>10.2e}{verdict}")
    print(f"  -> default formulation, every spread <= {bound:.0e}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    full = "--full" in sys.argv
    ok = True
    ok &= gate_spatial(0.05, 0.0, [6, 12, 24], gate=(1.85, 2.15))

    # Distorted-mesh order is gated on the SMOOTH family. A randomly
    # perturbed mesh redraws its perturbation at every resolution, so the
    # meshes are independent samples rather than refinements of one another:
    # the mesh quality itself wanders (25.9 / 30.1 / 28.4 deg at n = 6/12/24)
    # and the measured order wanders with it. The smooth family refines
    # toward one geometry (24.3 / 32.0 / 34.1 / 34.7 deg, converging), so the
    # order it reports is the scheme's. See ADR-013.
    #
    # Measured on the smooth family (linear gradient, the default):
    #   6->12  1.694 | 12->24  1.816 | 8->16  1.695 | 16->32  1.895
    # Rising toward 2 as the family approaches its limiting geometry. The
    # same code on a randomly perturbed family reported 1.23; that number was
    # measuring the mesh, not the scheme.
    # --full runs 8/16/32 and demands 1.85; the default runs 6/12/24, which is
    # pre-asymptotic, so it demands a rising trend instead of a fixed value.
    if full:
        ok &= gate_spatial(0.05, 0.25, [8, 16, 32], gate=(1.85, 2.3))
    else:
        ok &= gate_spatial(0.05, 0.25, [6, 12, 24], gate=(1.6, 2.3), rising=True)

    # Reported, not gated: warped faces are a separate first-order error
    # source in 3D, and this family is not a valid refinement sequence.
    gate_spatial(0.05, 0.25, [6, 12, 24], mode="warped", gate=(0.0, 9.9))

    ok &= gate_temporal(1.0, 0.0)
    ok &= gate_dt_independence()
    print()
    print("v1 Ethier-Steinman GATE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
