"""
Face mass fluxes that are divergence-free to machine precision.

For verification of a convection scheme the prescribed velocity field must be
discretely divergence-free, not just analytically. Evaluating u at face centres
and dotting with the area vector leaves a residual divergence of order h, and
that residual multiplies phi in the convective term -- it would contaminate the
measured order and be mistaken for a bug in the scheme.

The fix is Stokes' theorem. With u = curl(A),

    F_f = integral_f (curl A) . dS = contour integral over the face boundary of A . dl

Each edge of a cell is shared by exactly two of its faces, which traverse it in
opposite directions, so the fluxes out of a closed cell cancel exactly. The
quadrature rule does not matter for that cancellation, only that both faces use
the same points on the shared edge -- which they do, since the points depend
only on the edge's two endpoints.
"""

import numpy as np

# Two-point Gauss-Legendre on [0, 1].
_G = 0.5 / np.sqrt(3.0)
GAUSS_T = np.array([0.5 - _G, 0.5 + _G])
GAUSS_W = np.array([0.5, 0.5])


def face_flux_from_potential(mesh, A):
    """Contour integral of A around every face, internal and boundary.

    A(points) -> (n, 3) vector potential evaluated at arbitrary points.
    Returns (F_internal, F_boundary), each signed by the face's own area vector.
    """
    verts = mesh.vert.reshape(-1, 3)

    def circulation(fverts):
        total = np.zeros(len(fverts))
        for e in range(4):
            a = verts[fverts[:, e]]
            b = verts[fverts[:, (e + 1) % 4]]
            d = b - a
            for t, w in zip(GAUSS_T, GAUSS_W):
                total += w * np.einsum("ij,ij->i", A(a + t * d), d)
        return total

    return circulation(mesh.fverts), circulation(mesh.b_fverts)


def discrete_divergence(mesh, f_int, f_bnd):
    """Sum of outward fluxes per cell. Machine zero for a potential-derived flux."""
    div = np.zeros(mesh.nc)
    np.add.at(div, mesh.owner, f_int)
    np.add.at(div, mesh.neigh, -f_int)
    np.add.at(div, mesh.b_cell, f_bnd)
    return div


class SolenoidalField:
    """A divergence-free velocity field and its vector potential.

        A = (sin(pi y) sin(pi z), sin(pi z) sin(pi x), sin(pi x) sin(pi y)) / pi
        u = curl A

    Non-trivial in all three components, zero divergence analytically, and
    machine-zero divergence discretely through the contour integral above.
    """

    @staticmethod
    def potential(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        s = np.sin
        return np.column_stack([s(np.pi * y) * s(np.pi * z),
                                s(np.pi * z) * s(np.pi * x),
                                s(np.pi * x) * s(np.pi * y)]) / np.pi

    @staticmethod
    def velocity(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        s, c = np.sin, np.cos
        return np.column_stack([
            s(np.pi * x) * (c(np.pi * y) - c(np.pi * z)),
            s(np.pi * y) * (c(np.pi * z) - c(np.pi * x)),
            s(np.pi * z) * (c(np.pi * x) - c(np.pi * y)),
        ])


def integrate_face_flux(mesh, u_func, boundary=True):
    """Integrate u . dS over faces with a quadrature that is exact for quadratics.

    Evaluating u at the face centre and multiplying by the area vector is only
    exact for linear fields. The resulting O(h^2) error per face leaves an
    O(h) spurious mass source per cell, and because pressure is elliptic that
    boundary error is carried into the interior -- the whole solution drops to
    first order, not just the cells next to the wall.

    Each quad is fanned into four triangles from the vertex average, and each
    triangle uses the three edge midpoints with equal weight: exact for
    quadratics, so the per-cell specific divergence error is O(h^2).
    """
    verts = mesh.vert.reshape(-1, 3)
    fverts = mesh.b_fverts if boundary else mesh.fverts
    pts = verts[fverts]                                   # (nf, 4, 3)
    avg = pts.mean(axis=1)

    total = np.zeros(len(pts))
    for t in range(4):
        a, b = pts[:, t, :], pts[:, (t + 1) % 4, :]
        tri = 0.5 * np.cross(a - avg, b - avg)            # triangle area vector
        mids = [(avg + a) / 2.0, (a + b) / 2.0, (b + avg) / 2.0]
        acc = np.zeros_like(tri)
        for mpt in mids:
            acc += u_func(mpt) / 3.0
        total += np.einsum("ij,ij->i", acc, tri)
    return total


def face_average(mesh, func, boundary=True, ncomp=3):
    """Area-weighted average of a field over each face, exact for quadratics.

    A finite-volume Dirichlet condition constrains the face AVERAGE, not the
    value at the face centre. The two differ by O(h^2); on an orthogonal mesh
    those errors cancel between opposite faces of a cell, on a skewed mesh
    they do not, and the solution loses an order.
    """
    verts = mesh.vert.reshape(-1, 3)
    fverts = mesh.b_fverts if boundary else mesh.fverts
    pts = verts[fverts]
    avg = pts.mean(axis=1)

    num = np.zeros((len(pts), ncomp))
    den = np.zeros(len(pts))
    for t in range(4):
        a, b = pts[:, t, :], pts[:, (t + 1) % 4, :]
        area = np.linalg.norm(0.5 * np.cross(a - avg, b - avg), axis=1)
        acc = np.zeros((len(pts), ncomp))
        for mpt in ((avg + a) / 2.0, (a + b) / 2.0, (b + avg) / 2.0):
            acc += np.atleast_2d(func(mpt)) / 3.0
        num += area[:, None] * acc
        den += area
    return num / den[:, None]
