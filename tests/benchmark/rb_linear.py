#!/usr/bin/env python3
"""Linear growth rates of Rayleigh-Benard convection between rigid plates:
the reference for the transient-coupling gate (ADR-040).

In units of the gap d and the diffusion time d^2/kappa, a normal mode
w = W(z) cos(kx) e^{sigma t}, theta = Th(z) cos(kx) e^{sigma t} of the
Boussinesq equations linearised about the conduction profile T = 1 - z obeys

    sigma (D^2 - k^2) W = Pr (D^2 - k^2)^2 W - Pr Ra k^2 Th
    sigma Th            = W + (D^2 - k^2) Th

with W = DW = Th = 0 on both plates. This is the solver's problem exactly:
kappa = 1, nu = Pr, betaG = (0, 0, -Ra Pr).

Chebyshev collocation in the D^2 form of Dongarra, Straughan & Walker (1996):
V = (D^2 - k^2) W is a third unknown, so only second derivatives appear, and
DW = 0 replaces the V equation at the plates. The direct fourth-order form was
tried first: it agrees to 1e-9 on 24 and 32 points and then drifts with N by
round-off (3e-5 by N = 128), where this one holds to 6e-9 from N = 24 to 128.
The boundary rows leave spurious eigenvalues of order 1e13, discarded.

It checks itself, and fails if a check does: the rates agree to 1e-8 on 24 to
64 points; the neutral Rayleigh number at k_c = 3.117 is within 5e-3 of
1707.762 (Chandrasekhar); and the slope d sigma / d eps at onset is within
0.1% of the amplitude equation's 1 / tau_0, tau_0 = (Pr + 0.5117) / (19.65 Pr)
for rigid plates (Bodenschatz et al., arXiv patt-sol/9305001).

Run:  python3 rb_linear.py
"""
import numpy as np
from scipy.linalg import eig

K_C = 3.117
RA_C = 1707.762


def cheb(n):
    """Chebyshev points on [-1, 1] and the differentiation matrix (Trefethen)."""
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def growth_rate(Ra, k=K_C, Pr=1.0, n=48):
    D, _ = cheb(n)
    D = 2.0 * D                          # z = (x + 1) / 2 maps [-1, 1] to [0, 1]
    I = np.eye(n + 1)
    L = D @ D - k * k * I
    m = n + 1
    Z = np.zeros((m, m))
    # Unknowns [W, V, Th]; rows: V equation, V = L W, Th equation.
    A = np.block([[Z, Pr * L, -Pr * Ra * k * k * I],
                  [L, -I, Z],
                  [I, Z, L]])
    B = np.block([[Z, I, Z],
                  [Z, Z, Z],
                  [Z, Z, I]])
    for r, row in ((m, I[0]), (m + n, I[n])):     # W = 0 at the plates
        A[r, :] = 0.0
        B[r, :] = 0.0
        A[r, :m] = row
    for r, row in ((0, D[0]), (n, D[n])):         # DW = 0 at the plates
        A[r, :] = 0.0
        B[r, :] = 0.0
        A[r, :m] = row
    for r in (0, n):                              # Th = 0 at the plates
        A[2 * m + r, :] = 0.0
        B[2 * m + r, :] = 0.0
        A[2 * m + r, 2 * m + r] = 1.0
    w = eig(A, B, right=False)
    w = w[np.isfinite(w) & (np.abs(w) < 1e4)]
    return float(w[np.argmax(w.real)].real)


def main():
    ns = (24, 32, 48, 64)
    ok = True
    print("Rigid-rigid Rayleigh-Benard, Pr = 1, k = 3.117: leading growth rate")
    print(f"  {'Ra':>9}" + "".join(f"{f'N={n}':>18}" for n in ns))
    for Ra in (1600.0, 1700.0, 1800.0):
        s = [growth_rate(Ra, n=n) for n in ns]
        spread = max(s) - min(s)
        ok &= spread < 1e-8
        print(f"  {Ra:9.1f}" + "".join(f"{v:18.12f}" for v in s) + f"   spread {spread:.1e}")
    # Neutral Rayleigh number at k_c, by the secant method.
    a, b = 1700.0, 1720.0
    fa, fb = growth_rate(a), growth_rate(b)
    for _ in range(40):
        c = b - fb * (b - a) / (fb - fa)
        a, fa, b, fb = b, fb, c, growth_rate(c)
        if abs(b - a) < 1e-9:
            break
    good = abs(b - RA_C) < 5e-3
    ok &= good
    print(f"\n  neutral Ra at k = {K_C}: {b:.4f}   (Chandrasekhar {RA_C}, within 5e-3): "
          f"{'PASS' if good else 'FAIL'}")
    slope = (growth_rate(b + 1.0) - growth_rate(b - 1.0)) / 2.0 * b
    tau0 = (1.0 + 0.5117) / 19.65
    good = abs(slope * tau0 - 1.0) < 1e-3
    ok &= good
    print(f"  d sigma / d eps at onset: {slope:.3f}   (amplitude equation 1/tau_0 = "
          f"{1/tau0:.3f}, within 0.1%): {'PASS' if good else 'FAIL'}")
    print(f"\n  reference for the gate: sigma(Ra = 1800) = {growth_rate(1800.0):.9f}")
    print("linear-theory reference: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
