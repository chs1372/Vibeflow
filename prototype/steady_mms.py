"""
Steady manufactured solutions, used for the Rhie-Chow time-step independence
gate (ADR-010, ADR-037).

u is the solenoidal field from flux.py. The pressure is either constant or
the smooth field

    p = cos(pi x) cos(pi y) cos(pi z),

which has zero mean over the unit cube. The body force is whatever makes the
pair an exact steady solution:

    s = (u . grad) u - nu laplacian(u) + grad(p)

A steady state has no temporal discretisation error, so any difference between
runs using different time steps is the time step leaking into the spatial
discretisation -- which is exactly the Rhie-Chow trap ADR-010 describes. The
constant-pressure problem leaves the pressure-damping part of the face flux
almost idle; the second one makes it work.
"""

import numpy as np
import sympy as sp

from flux import SolenoidalField as SF

_x, _y, _z = sp.symbols("x y z", real=True)


def _pressure_expr():
    c, pi = sp.cos, sp.pi
    return c(pi*_x) * c(pi*_y) * c(pi*_z)


def _build_source(nu, with_pressure):
    s, c, pi = sp.sin, sp.cos, sp.pi
    U = [s(pi*_x)*(c(pi*_y) - c(pi*_z)),
         s(pi*_y)*(c(pi*_z) - c(pi*_x)),
         s(pi*_z)*(c(pi*_x) - c(pi*_y))]
    X = [_x, _y, _z]
    assert sp.simplify(sum(sp.diff(U[i], X[i]) for i in range(3))) == 0
    P = _pressure_expr() if with_pressure else sp.Integer(0)

    expr = []
    for i in range(3):
        conv = sum(U[j]*sp.diff(U[i], X[j]) for j in range(3))
        lap = sum(sp.diff(U[i], X[j], 2) for j in range(3))
        expr.append(sp.simplify(conv - nu*lap + sp.diff(P, X[i])))
    return sp.lambdify((_x, _y, _z), expr, "numpy")


_CACHE = {}


def source(p, nu, with_pressure=False):
    key = (nu, with_pressure)
    if key not in _CACHE:
        _CACHE[key] = _build_source(nu, with_pressure)
    out = _CACHE[key](p[:, 0], p[:, 1], p[:, 2])
    return np.column_stack([np.broadcast_to(np.asarray(o, dtype=float), (len(p),))
                            for o in out])


def velocity(p):
    return SF.velocity(p)


def pressure(p, with_pressure=False):
    if not with_pressure:
        return np.zeros(len(p))
    return np.cos(np.pi*p[:, 0]) * np.cos(np.pi*p[:, 1]) * np.cos(np.pi*p[:, 2])
