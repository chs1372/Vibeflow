"""
v2a gates, Python side: the energy equation and Boussinesq buoyancy (ADR-038).

Written before the code they judge. Each runs the PisoSolver with the energy
equation switched on (kappa, beta_g, t_ref, t_ref_grad) and says PASS or FAIL;
run against a solver without it they fail with the reason.

  1. Temperature in an exact flow. The Ethier-Steinman velocity, which solves
     Navier-Stokes exactly, carries a manufactured temperature
         T = exp(-t) sin(pi x) sin(pi y) sin(pi z)
     with the source that makes it exact and no buoyancy. Spatial order on the
     orthogonal and the smooth-distortion families, BDF2 order in time.

  2. Boussinesq, steady, manufactured. The steady solenoidal velocity of the
     dt-independence gate, p = cos(pi x) cos(pi y) cos(pi z) and
     T = 1 + sin(pi x) sin(pi y) sin(pi z) / 2, with buoyancy on:
         f = -beta (T - T_ref) g,    beta g = (0, 0, -1),  T_ref = 0,
     and sources that make all three exact. Orders of u and T, and the same
     steady state at dt = 0.2 and 2.0.

  3. A stratified fluid at rest stays at rest. The Rayleigh-Benard conduction
     profile T = 1 - z between a hot bottom and a cold top, adiabatic sides,
     the reference stratification equal to it: 50 steps must leave u at zero
     and T linear, to 1e-12, on a Cartesian and a distorted mesh.
"""

import sys

import numpy as np
import sympy as sp

from flux import SolenoidalField as SF, face_average, integrate_face_flux
from mesh import HexMesh
from piso import PisoSolver
import ethier_steinman as ES

PI = np.pi


# ------------------------------------------------------------------ helpers
def l2(err, vol):
    err = err if err.ndim == 2 else err[:, None]
    return np.sqrt((np.einsum("ij,ij->i", err, err) * vol).sum() / vol.sum())


def scalar_face_average(m, fn):
    return face_average(m, lambda q: fn(q)[:, None], ncomp=1)[:, 0]


def orders(errs, hs):
    return [np.log(errs[i - 1] / errs[i]) / np.log(hs[i - 1] / hs[i])
            for i in range(1, len(errs))]


def verdict(label, orders_, lo, hi, rising=False):
    last = orders_[-1]
    ok = lo <= last <= hi
    msg = f"  -> {label}: order {last:.3f} in [{lo}, {hi}]"
    if rising and len(orders_) > 1:
        r = all(orders_[i] >= orders_[i - 1] - 0.02 for i in range(1, len(orders_)))
        ok &= r
        msg += f", trend {'rising' if r else 'FALLING'}"
    print(msg + (": PASS" if ok else ": FAIL"))
    return ok


# ------------------------------------------ 1. temperature in an exact flow
def t_exact(q, t):
    return np.exp(-t) * np.sin(PI * q[:, 0]) * np.sin(PI * q[:, 1]) * np.sin(PI * q[:, 2])


def t_source(q, t, nu, kappa):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    e = np.exp(-t)
    T = t_exact(q, t)
    gT = e * PI * np.column_stack([np.cos(PI * x) * np.sin(PI * y) * np.sin(PI * z),
                                   np.sin(PI * x) * np.cos(PI * y) * np.sin(PI * z),
                                   np.sin(PI * x) * np.sin(PI * y) * np.cos(PI * z)])
    u = ES.velocity(q, t, nu)
    return -T + np.einsum("ij,ij->i", u, gT) + 3.0 * PI * PI * kappa * T


def run_energy_exact_flow(n, dt, nsteps, nu, kappa, skew, mode="smooth", nouter=6):
    m = HexMesh(n, skew=skew, seed=1, skew_mode=mode)
    solver = PisoSolver(m, nu, dt, n_correctors=2, n_outer=nouter,
                        kappa=kappa, beta_g=np.zeros(3))
    solver.u = ES.velocity(m.cell_centre, 0.0, nu)
    solver.u_old = solver.u.copy()
    solver.u_old2 = solver.u.copy()
    p0 = ES.pressure(m.cell_centre, 0.0, nu)
    solver.p = p0 - p0.mean()
    uf0 = (solver.w[:, None] * solver.u[m.owner]
           + (1 - solver.w)[:, None] * solver.u[m.neigh])
    solver.F = np.einsum("ij,ij->i", uf0, m.face_area)
    solver.F_old = solver.F.copy()
    solver.F_old2 = solver.F.copy()
    solver.T = t_exact(m.cell_centre, 0.0)
    solver.T_old = solver.T.copy()
    solver.T_old2 = solver.T.copy()
    src = np.zeros((m.nc, 3))
    for k in range(nsteps):
        t_new = (k + 1) * dt
        u_b = face_average(m, lambda q: ES.velocity(q, t_new, nu))
        Fb = ES.adjust_boundary_flux(
            m, integrate_face_flux(m, lambda q: ES.velocity(q, t_new, nu)))
        T_b = scalar_face_average(m, lambda q: t_exact(q, t_new))
        solver.advance(u_b, Fb, src, T_b=T_b,
                       T_src=t_source(m.cell_centre, t_new, nu, kappa))
    return m, solver, nsteps * dt


def gate_energy_exact_flow():
    print("\n1. temperature in the exact Ethier-Steinman flow (beta = 0)")
    nu, kappa, ok = 0.05, 0.05, True
    for skew, lo, hi, rising, tag in ((0.0, 1.85, 2.15, False, "orthogonal"),
                                      (0.25, 1.6, 2.3, True, "smooth distortion")):
        errs, hs = [], []
        for n in (6, 12, 24):
            m, s, t_end = run_energy_exact_flow(n, 2e-4, 2, nu, kappa, skew)
            errs.append(l2(s.T - t_exact(m.cell_centre, t_end), m.cell_volume))
            hs.append(1.0 / n)
            print(f"  {tag:<18} n={n:<3} L2(T) {errs[-1]:.6e}")
        ok &= verdict(f"spatial, {tag}", orders(errs, hs), lo, hi, rising)
    # BDF2 in time, against dt/64 on the same mesh (as the Navier-Stokes gate)
    n, nu_t, t_end = 8, 1.0, 0.8
    ref = run_energy_exact_flow(n, t_end / 64, 64, nu_t, nu_t, 0.0, nouter=200)[1].T
    errs, dts = [], []
    for s_ in (2, 4, 8):
        m, s, _ = run_energy_exact_flow(n, t_end / s_, s_, nu_t, nu_t, 0.0, nouter=200)
        errs.append(l2(s.T - ref, m.cell_volume))
        dts.append(t_end / s_)
        print(f"  temporal   dt={t_end / s_:<6} L2(T - T_ref) {errs[-1]:.6e}")
    ok &= verdict("temporal, BDF2", orders(errs, dts), 1.8, 2.6)
    return ok


# --------------------------------------------- 2. steady Boussinesq MMS
_X = sp.symbols("x y z", real=True)


def _boussinesq_sources(nu, kappa):
    x, y, z = _X
    s, c, pi = sp.sin, sp.cos, sp.pi
    U = [s(pi*x)*(c(pi*y) - c(pi*z)), s(pi*y)*(c(pi*z) - c(pi*x)),
         s(pi*z)*(c(pi*x) - c(pi*y))]
    P = c(pi*x)*c(pi*y)*c(pi*z)
    T = 1 + s(pi*x)*s(pi*y)*s(pi*z)/2
    bg = (0, 0, -1)                       # beta * g, per unit temperature
    su = []
    for i in range(3):
        conv = sum(U[j]*sp.diff(U[i], _X[j]) for j in range(3))
        lap = sum(sp.diff(U[i], _X[j], 2) for j in range(3))
        f = -T * bg[i]                     # f = -beta (T - 0) g
        su.append(sp.simplify(conv - nu*lap + sp.diff(P, _X[i]) - f))
    sT = sp.simplify(sum(U[j]*sp.diff(T, _X[j]) for j in range(3))
                     - kappa*sum(sp.diff(T, _X[j], 2) for j in range(3)))
    fu = sp.lambdify(_X, su, "numpy")
    fT = sp.lambdify(_X, sT, "numpy")
    return (lambda q: np.column_stack([np.broadcast_to(np.asarray(v, float), (len(q),))
                                       for v in fu(q[:, 0], q[:, 1], q[:, 2])]),
            lambda q: np.broadcast_to(np.asarray(fT(q[:, 0], q[:, 1], q[:, 2]), float),
                                      (len(q),)))


def b_temperature(q):
    return 1 + 0.5 * np.sin(PI * q[:, 0]) * np.sin(PI * q[:, 1]) * np.sin(PI * q[:, 2])


def b_pressure(q):
    return np.cos(PI * q[:, 0]) * np.cos(PI * q[:, 1]) * np.cos(PI * q[:, 2])


def steady_boussinesq(n, skew, mode, dt, nu=0.1, kappa=0.1, tol=1e-13, max_time=400.0):
    m = HexMesh(n, skew=skew, seed=1, skew_mode=mode)
    solver = PisoSolver(m, nu, dt, n_correctors=2, n_outer=3, kappa=kappa,
                        beta_g=np.array([0.0, 0.0, -1.0]), t_ref=0.0)
    su, sT = _boussinesq_sources(nu, kappa)
    src = su(m.cell_centre)
    T_src = sT(m.cell_centre)
    u_b = face_average(m, SF.velocity)
    Fb = ES.adjust_boundary_flux(m, integrate_face_flux(m, SF.velocity))
    T_b = scalar_face_average(m, b_temperature)
    solver.T = np.ones(m.nc)
    solver.T_old = solver.T.copy()
    solver.T_old2 = solver.T.copy()
    prev, change, k = None, np.inf, 0
    for k in range(int(round(max_time / dt))):
        solver.advance(u_b, Fb, src, T_b=T_b, T_src=T_src)
        state = (solver.u.copy(), solver.p.copy(), solver.T.copy())
        if prev is not None:
            change = max(np.abs(a - b).max() for a, b in zip(state, prev))
            if change < tol:
                break
        prev = state
    return m, solver, k + 1, change


def gate_boussinesq_mms():
    print("\n2. steady manufactured Boussinesq flow (beta g = (0, 0, -1))")
    ok = True
    for skew, lo, hi, rising, tag in ((0.0, 1.85, 2.15, False, "orthogonal"),
                                      (0.25, 1.6, 2.3, True, "smooth distortion")):
        eu, eT, hs = [], [], []
        for n in (6, 12, 24):
            m, s, k, ch = steady_boussinesq(n, skew, "smooth", 2.0)
            eu.append(l2(s.u - SF.velocity(m.cell_centre), m.cell_volume))
            eT.append(l2(s.T - b_temperature(m.cell_centre), m.cell_volume))
            hs.append(1.0 / n)
            print(f"  {tag:<18} n={n:<3} steps {k:<4} L2(u) {eu[-1]:.6e}  "
                  f"L2(T) {eT[-1]:.6e}  last change {ch:.0e}")
            ok &= ch < 1e-13
        ok &= verdict(f"u, {tag}", orders(eu, hs), lo, hi, rising)
        ok &= verdict(f"T, {tag}", orders(eT, hs), lo, hi, rising)
    # The buoyancy coupling must not bring dt back into the steady state.
    states = []
    for dt in (0.2, 2.0):
        m, s, k, ch = steady_boussinesq(8, 0.25, "warped", dt)
        states.append((s.u.copy(), s.T.copy()))
        ok &= ch < 1e-13
    eu = l2(states[1][0] - SF.velocity(m.cell_centre), m.cell_volume)
    eT = l2(states[1][1] - b_temperature(m.cell_centre), m.cell_volume)
    su = l2(states[0][0] - states[1][0], m.cell_volume) / eu
    sT = l2(states[0][1] - states[1][1], m.cell_volume) / eT
    passed = su <= 1e-6 and sT <= 1e-6
    ok &= passed
    print(f"  dt = 0.2 vs 2.0, skewed n=8: spread u {su:.1e}, T {sT:.1e} "
          f"(bound 1e-6): {'PASS' if passed else 'FAIL'}")
    return ok


# ------------------------------------------------- 3. stratified rest state
def gate_rest_state():
    print("\n3. a fluid resting in its reference stratification stays at rest")
    ok = True
    for skew, tag in ((0.0, "Cartesian"), (0.25, "distorted")):
        m = HexMesh(8, skew=skew, seed=1, skew_mode="smooth")
        profile = lambda q: 1.0 - q[:, 2]
        solver = PisoSolver(m, 1.0, 0.01, n_correctors=2, n_outer=3, kappa=1.0,
                            beta_g=np.array([0.0, 0.0, -1700.0]),
                            t_ref=1.0, t_ref_grad=np.array([0.0, 0.0, -1.0]))
        solver.T = profile(m.cell_centre)
        solver.T_old = solver.T.copy()
        solver.T_old2 = solver.T.copy()
        # Hot bottom, cold top, adiabatic sides; walls everywhere.
        nb = len(m.b_cell)
        z = m.b_centre[:, 2]
        n_hat = m.b_area / np.linalg.norm(m.b_area, axis=1)[:, None]
        horizontal = np.abs(n_hat[:, 2]) > 0.5
        T_bc = np.where(horizontal, 0, 1)            # 0 fixed T, 1 fixed flux
        T_b = np.where(horizontal, 1.0 - z, 0.0)     # value, or zero heat flux
        u_b = np.zeros((nb, 3))
        Fb = np.zeros(nb)
        src = np.zeros((m.nc, 3))
        for _ in range(50):
            solver.advance(u_b, Fb, src, T_b=T_b, T_bc=T_bc)
        du = np.abs(solver.u).max()
        dT = np.abs(solver.T - profile(m.cell_centre)).max()
        passed = du <= 1e-12 and dT <= 1e-12
        ok &= passed
        print(f"  {tag:<10} max|u| {du:.1e}   max|T - (1 - z)| {dT:.1e}  "
              f"{'PASS' if passed else 'FAIL'}")
    return ok


def main():
    ok = True
    for gate in (gate_energy_exact_flow, gate_boussinesq_mms, gate_rest_state):
        try:
            ok &= gate()
        except TypeError as e:          # the solver does not take the arguments
            print(f"  FAIL: the solver has no energy equation ({e})")
            ok = False
    print("\nv2a heat-transfer GATE (Python): " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
