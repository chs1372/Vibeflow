#!/usr/bin/env python3
"""Symbolically verify the exact solutions the v1 gates are built on.

A typo in an exact solution does not announce itself: the gate still runs, the
solver still converges, and the measured order is quietly meaningless. This
checks with sympy that Ethier-Steinman really satisfies incompressible
Navier-Stokes and that the steady MMS source really makes its velocity field a
solution.
"""
import sys

import sympy as sp

x, y, z, t, nu = sp.symbols("x y z t nu", real=True)
X = [x, y, z]


def check_ethier_steinman():
    a, d = sp.pi / 4, sp.pi / 2
    E = sp.exp(-d**2 * nu * t)
    U = [-a * (sp.exp(a*x)*sp.sin(a*y + d*z) + sp.exp(a*z)*sp.cos(a*x + d*y)) * E,
         -a * (sp.exp(a*y)*sp.sin(a*z + d*x) + sp.exp(a*x)*sp.cos(a*y + d*z)) * E,
         -a * (sp.exp(a*z)*sp.sin(a*x + d*y) + sp.exp(a*y)*sp.cos(a*z + d*x)) * E]
    p = -a**2 / 2 * (
        sp.exp(2*a*x) + sp.exp(2*a*y) + sp.exp(2*a*z)
        + 2*sp.sin(a*x + d*y)*sp.cos(a*z + d*x)*sp.exp(a*(y + z))
        + 2*sp.sin(a*y + d*z)*sp.cos(a*x + d*y)*sp.exp(a*(z + x))
        + 2*sp.sin(a*z + d*x)*sp.cos(a*y + d*z)*sp.exp(a*(x + y))
    ) * sp.exp(-2*d**2*nu*t)

    ok = sp.simplify(sum(sp.diff(U[i], X[i]) for i in range(3))) == 0
    print(f"  Ethier-Steinman  div(u) = 0                 {'PASS' if ok else 'FAIL'}")
    good = ok
    for i, name in enumerate("uvw"):
        res = (sp.diff(U[i], t)
               + sum(U[j] * sp.diff(U[i], X[j]) for j in range(3))
               + sp.diff(p, X[i])
               - nu * sum(sp.diff(U[i], X[j], 2) for j in range(3)))
        r = sp.simplify(sp.expand(res)) == 0
        print(f"  Ethier-Steinman  momentum residual {name} = 0     {'PASS' if r else 'FAIL'}")
        good &= r
    return good


def check_steady_mms():
    s, c, pi = sp.sin, sp.cos, sp.pi
    U = [s(pi*x)*(c(pi*y) - c(pi*z)),
         s(pi*y)*(c(pi*z) - c(pi*x)),
         s(pi*z)*(c(pi*x) - c(pi*y))]
    ok = sp.simplify(sum(sp.diff(U[i], X[i]) for i in range(3))) == 0
    print(f"  steady MMS       div(u) = 0                 {'PASS' if ok else 'FAIL'}")

    # The source is defined as conv - nu*lap, so the residual is zero by
    # construction; what is worth checking is that the potential in flux.py
    # really curls to this field.
    A = [s(pi*y)*s(pi*z)/pi, s(pi*z)*s(pi*x)/pi, s(pi*x)*s(pi*y)/pi]
    curl = [sp.diff(A[2], y) - sp.diff(A[1], z),
            sp.diff(A[0], z) - sp.diff(A[2], x),
            sp.diff(A[1], x) - sp.diff(A[0], y)]
    match = all(sp.simplify(curl[i] - U[i]) == 0 for i in range(3))
    print(f"  steady MMS       u = curl(A) as flux.py assumes  "
          f"{'PASS' if match else 'FAIL'}")
    return ok and match


if __name__ == "__main__":
    print("symbolic verification of the exact solutions")
    good = check_ethier_steinman() & check_steady_mms()
    print(f"\nexact-solution verification: {'PASS' if good else 'FAIL'}")
    sys.exit(0 if good else 1)
