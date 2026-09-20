"""
Cell-centred finite-volume diffusion operator with non-orthogonal correction.

Solves  -div(gamma grad u) = f  with Dirichlet boundaries.

Face flux uses the over-relaxed decomposition of the area vector:

    Sf = Delta + k,   Delta = d * |Sf|^2 / (d . Sf)

so the implicit part carries the full face area even on skewed meshes, and the
remainder k is handled by deferred correction against a least-squares cell
gradient. This is the scheme the C++ discretization layer must implement.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class LeastSquaresGradient:
    """Weighted least-squares cell gradient, including boundary faces."""

    def __init__(self, mesh):
        self.m = mesh
        m = mesh
        nc = m.nc
        A = np.zeros((nc, 3, 3))

        d_int = m.cell_centre[m.neigh] - m.cell_centre[m.owner]
        w_int = 1.0 / np.linalg.norm(d_int, axis=1) ** 2
        outer = w_int[:, None, None] * np.einsum("ij,ik->ijk", d_int, d_int)
        np.add.at(A, m.owner, outer)
        np.add.at(A, m.neigh, outer)

        d_bnd = m.b_centre - m.cell_centre[m.b_cell]
        w_bnd = 1.0 / np.linalg.norm(d_bnd, axis=1) ** 2
        outer_b = w_bnd[:, None, None] * np.einsum("ij,ik->ijk", d_bnd, d_bnd)
        np.add.at(A, m.b_cell, outer_b)

        self.Ainv = np.linalg.inv(A)
        self.d_int, self.w_int = d_int, w_int
        self.d_bnd, self.w_bnd = d_bnd, w_bnd

    def __call__(self, u, ub):
        m = self.m
        rhs = np.zeros((m.nc, 3))
        du = u[m.neigh] - u[m.owner]
        contrib = (self.w_int * du)[:, None] * self.d_int
        np.add.at(rhs, m.owner, contrib)
        np.add.at(rhs, m.neigh, contrib)

        dub = ub - u[m.b_cell]
        np.add.at(rhs, m.b_cell, (self.w_bnd * dub)[:, None] * self.d_bnd)

        return np.einsum("ijk,ik->ij", self.Ainv, rhs)


class DiffusionOperator:
    def __init__(self, mesh, gamma=1.0):
        self.m = mesh
        self.gamma = gamma
        m = mesh

        # --- internal faces: over-relaxed decomposition
        d = m.cell_centre[m.neigh] - m.cell_centre[m.owner]
        sf = m.face_area
        d_dot_s = np.einsum("ij,ij->i", d, sf)
        self.a_int = np.einsum("ij,ij->i", sf, sf) / d_dot_s      # |Sf|^2 / (d.Sf)
        self.k_int = sf - self.a_int[:, None] * d                  # non-orthogonal remainder
        self.d_int = d

        # linear interpolation weight for face gradient (owner side)
        dof = np.linalg.norm(m.face_centre - m.cell_centre[m.owner], axis=1)
        dnf = np.linalg.norm(m.face_centre - m.cell_centre[m.neigh], axis=1)
        self.w_owner = dnf / (dof + dnf)

        # --- boundary faces
        db = m.b_centre - m.cell_centre[m.b_cell]
        sb = m.b_area
        self.a_bnd = np.einsum("ij,ij->i", sb, sb) / np.einsum("ij,ij->i", db, sb)
        self.k_bnd = sb - self.a_bnd[:, None] * db

        self.grad = LeastSquaresGradient(mesh)
        self.A = self._assemble()
        self.lu = spla.splu(self.A.tocsc())

    def _assemble(self):
        m = self.m
        g = self.gamma
        rows, cols, vals = [], [], []

        rows.append(m.owner); cols.append(m.owner); vals.append(+g * self.a_int)
        rows.append(m.owner); cols.append(m.neigh); vals.append(-g * self.a_int)
        rows.append(m.neigh); cols.append(m.neigh); vals.append(+g * self.a_int)
        rows.append(m.neigh); cols.append(m.owner); vals.append(-g * self.a_int)
        rows.append(m.b_cell); cols.append(m.b_cell); vals.append(+g * self.a_bnd)

        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(m.nc, m.nc))
        return A.tocsr()

    def rhs(self, source, ub, u_prev):
        """Explicit side: volume source + Dirichlet + non-orthogonal correction."""
        m = self.m
        g = self.gamma
        b = source * m.cell_volume

        np.add.at(b, m.b_cell, g * self.a_bnd * ub)

        gr = self.grad(u_prev, ub)
        gf = (self.w_owner[:, None] * gr[m.owner]
              + (1.0 - self.w_owner)[:, None] * gr[m.neigh])
        corr = g * np.einsum("ij,ij->i", self.k_int, gf)
        np.add.at(b, m.owner, corr)
        np.add.at(b, m.neigh, -corr)

        corr_b = g * np.einsum("ij,ij->i", self.k_bnd, gr[m.b_cell])
        np.add.at(b, m.b_cell, corr_b)
        return b

    def solve(self, source, ub, n_corr=20, tol=1e-12):
        u = np.zeros(self.m.nc)
        for it in range(n_corr):
            u_new = self.lu.solve(self.rhs(source, ub, u))
            delta = np.abs(u_new - u).max()
            u = u_new
            if delta < tol * max(1.0, np.abs(u).max()):
                break
        return u, it + 1
