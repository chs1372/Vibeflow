"""
Convection-diffusion operator.

Convection is implicit first-order upwind plus a deferred correction to the
second-order face value. Upwind alone is unconditionally stable but first
order; the correction recovers second order while leaving the matrix an
M-matrix, so the linear solve stays well behaved at any cell Peclet number.

The second-order face value carries a skewness correction:

    phi_f = [w phi_P + (1-w) phi_N] + grad(phi)_f . (x_f - x_lin)

where x_lin is the point the linear interpolation actually refers to. Without
that term the scheme drops toward first order on a skewed mesh -- which the
gate is built to catch.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fvm import LeastSquaresGradient, DiffusionOperator


class ConvectionDiffusion:
    def __init__(self, mesh, gamma, f_int, f_bnd):
        """f_int, f_bnd: face mass fluxes, signed by each face's area vector."""
        self.m = mesh
        self.gamma = gamma
        self.F = f_int
        self.Fb = f_bnd
        self.diff = DiffusionOperator(mesh, gamma)
        self.grad = self.diff.grad

        m = mesh
        # Linear-interpolation weight and the offset from the interpolated
        # point to the true face centre.
        dof = np.linalg.norm(m.face_centre - m.cell_centre[m.owner], axis=1)
        dnf = np.linalg.norm(m.face_centre - m.cell_centre[m.neigh], axis=1)
        self.w = dnf / (dof + dnf)
        x_lin = (self.w[:, None] * m.cell_centre[m.owner]
                 + (1.0 - self.w)[:, None] * m.cell_centre[m.neigh])
        self.skew_vec = m.face_centre - x_lin

        self.Fp = np.maximum(self.F, 0.0)
        self.Fn = np.maximum(-self.F, 0.0)
        self.A = self._assemble()
        self.lu = spla.splu(self.A.tocsc())

    def _assemble(self):
        m = self.m
        g = self.gamma
        a = self.diff.a_int
        rows, cols, vals = [], [], []

        # diffusion (over-relaxed implicit part)
        rows += [m.owner, m.owner, m.neigh, m.neigh, m.b_cell]
        cols += [m.owner, m.neigh, m.neigh, m.owner, m.b_cell]
        vals += [+g * a, -g * a, +g * a, -g * a, +g * self.diff.a_bnd]

        # convection (implicit upwind)
        rows += [m.owner, m.owner, m.neigh, m.neigh]
        cols += [m.owner, m.neigh, m.neigh, m.owner]
        vals += [self.Fp, -self.Fn, self.Fn, -self.Fp]

        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(m.nc, m.nc))
        return A.tocsr()

    def face_value_ho(self, phi, phi_b, grad):
        """Second-order face value with skewness correction."""
        m = self.m
        lin = self.w * phi[m.owner] + (1.0 - self.w) * phi[m.neigh]
        gf = (self.w[:, None] * grad[m.owner]
              + (1.0 - self.w)[:, None] * grad[m.neigh])
        return lin + np.einsum("ij,ij->i", gf, self.skew_vec)

    def rhs(self, source, phi_b, phi_prev):
        m = self.m
        g = self.gamma
        b = source * m.cell_volume
        grad = self.grad(phi_prev, phi_b)

        # --- diffusion: Dirichlet + non-orthogonal correction
        np.add.at(b, m.b_cell, g * self.diff.a_bnd * phi_b)
        gf = (self.diff.w_owner[:, None] * grad[m.owner]
              + (1.0 - self.diff.w_owner)[:, None] * grad[m.neigh])
        corr = g * np.einsum("ij,ij->i", self.diff.k_int, gf)
        np.add.at(b, m.owner, corr)
        np.add.at(b, m.neigh, -corr)
        np.add.at(b, m.b_cell, g * np.einsum("ij,ij->i", self.diff.k_bnd, grad[m.b_cell]))

        # --- convection: deferred correction from upwind to second order
        ho = self.face_value_ho(phi_prev, phi_b, grad)
        ud = np.where(self.F > 0.0, phi_prev[m.owner], phi_prev[m.neigh])
        dc = self.F * (ho - ud)
        np.add.at(b, m.owner, -dc)
        np.add.at(b, m.neigh, +dc)

        # Dirichlet boundary: the face value is known, so the whole convective
        # flux through it is a source term (OpenFOAM's fixedValue behaviour).
        np.add.at(b, m.b_cell, -self.Fb * phi_b)
        return b

    def solve(self, source, phi_b, n_corr=400, tol=1e-12, relax=1.0):
        """Returns (phi, iterations, converged).

        A truncated deferred-correction loop still returns a plausible field,
        so the caller is told whether the loop actually converged rather than
        having to infer it from the iteration count.
        """
        phi = np.zeros(self.m.nc)
        converged = False
        for it in range(n_corr):
            new = self.lu.solve(self.rhs(source, phi_b, phi))
            if relax < 1.0:
                new = phi + relax * (new - phi)
            delta = np.abs(new - phi).max()
            phi = new
            if delta < tol * max(1.0, np.abs(phi).max()):
                converged = True
                break
        return phi, it + 1, converged

    def peclet(self):
        """Max cell Peclet number, |F| d / (gamma |S|)."""
        m = self.m
        d = np.linalg.norm(m.cell_centre[m.neigh] - m.cell_centre[m.owner], axis=1)
        area = np.linalg.norm(m.face_area, axis=1)
        return (np.abs(self.F) * d / (self.gamma * area ** 2)).max() * area.max()
