"""
Steady manufactured solution, used for the Rhie-Chow time-step independence gate.

u is the solenoidal field from flux.py, p is constant, and the body force is
whatever makes that pair an exact steady solution:

    s = (u . grad) u - nu laplacian(u)

A steady state has no temporal discretisation error, so any difference between
runs using different time steps is the time step leaking into the spatial
discretisation -- which is exactly the Rhie-Chow trap ADR-010 describes.
"""

import numpy as np
import sympy as sp

from flux import SolenoidalField as SF

_x, _y, _z = sp.symbols("x y z", real=True)


def _build_source(nu):
    s, c, pi = sp.sin, sp.cos, sp.pi
    U = [s(pi*_x)*(c(pi*_y) - c(pi*_z)),
         s(pi*_y)*(c(pi*_z) - c(pi*_x)),
         s(pi*_z)*(c(pi*_x) - c(pi*_y))]
    X = [_x, _y, _z]
    assert sp.simplify(sum(sp.diff(U[i], X[i]) for i in range(3))) == 0

    expr = []
    for i in range(3):
        conv = sum(U[j]*sp.diff(U[i], X[j]) for j in range(3))
        lap = sum(sp.diff(U[i], X[j], 2) for j in range(3))
        expr.append(sp.simplify(conv - nu*lap))
    return sp.lambdify((_x, _y, _z), expr, "numpy")


_CACHE = {}


def source(p, nu):
    if nu not in _CACHE:
        _CACHE[nu] = _build_source(nu)
    out = _CACHE[nu](p[:, 0], p[:, 1], p[:, 2])
    return np.column_stack([np.broadcast_to(np.asarray(o, dtype=float), (len(p),))
                            for o in out])


def velocity(p):
    return SF.velocity(p)
