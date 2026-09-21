"""
Gradient reconstruction.

MEASURED, on the exact Ethier-Steinman pressure over a mesh with 30 deg maximum
non-orthogonality (see ADR-012):

    method                 n=6        n=12   order      n=24   order
    Green-Gauss       1.64e-01    1.47e-01    0.16   1.62e-01   -0.14
    LSQ linear        1.19e-01    5.49e-02    1.11   2.66e-02    1.05

Neither is second order, and the collocated velocity correction
u = H/aP - grad(p) V/aP uses the gradient DIRECTLY, so the velocity cannot beat
it. A linear least-squares fit reproduces linear fields exactly and therefore
has O(h) error on a quadratic one; to get a second-order gradient the fit has
to carry the quadratic terms too.

QuadraticLSQGradient fits

    phi(x) ~ phi_P + g . d + 1/2 d^T H d

over a two-ring stencil (face neighbours, plus their face neighbours) and keeps
g. Nine unknowns in 3D, against roughly 18-24 neighbours for a hex cell.
"""

import numpy as np


class QuadraticLSQGradient:
    """Second-order gradient by a quadratic least-squares fit."""

    N_TERMS = 9   # gx gy gz, then 1/2 dxx dyy dzz, then dxy dxz dyz

    def __init__(self, mesh, weight_power=2.0, rings=2):
        self.m = mesh
        self._build_stencil(rings)
        self._factor(weight_power)

    # ------------------------------------------------------------- stencil
    def _build_stencil(self, rings):
        m = self.m
        nc = m.nc
        adj = [[] for _ in range(nc)]
        for f in range(len(m.owner)):
            o, n = int(m.owner[f]), int(m.neigh[f])
            adj[o].append(n)
            adj[n].append(o)

        # Boundary faces act as extra samples: without them a corner cell has
        # too few neighbours for a nine-term fit.
        bfaces = [[] for _ in range(nc)]
        for f in range(len(m.b_cell)):
            bfaces[int(m.b_cell[f])].append(f)

        cells, bnds = [], []
        for c in range(nc):
            seen = {c}
            ring = set(adj[c])
            seen |= ring
            for _ in range(rings - 1):
                nxt = set()
                for r in ring:
                    nxt |= set(adj[r])
                nxt -= seen
                seen |= nxt
                ring = nxt
            lst = sorted(seen - {c})
            cells.append(lst)
            bl = list(bfaces[c])
            for nb in lst:
                bl.extend(bfaces[nb])
            bnds.append(sorted(set(bl)))

        self.maxc = max(len(c) for c in cells)
        self.maxb = max(len(b) for b in bnds) if any(bnds) else 0
        self.cell_idx = np.zeros((nc, self.maxc), dtype=np.int64)
        self.cell_ok = np.zeros((nc, self.maxc), dtype=bool)
        self.bnd_idx = np.zeros((nc, max(self.maxb, 1)), dtype=np.int64)
        self.bnd_ok = np.zeros((nc, max(self.maxb, 1)), dtype=bool)
        for c in range(nc):
            k = len(cells[c])
            self.cell_idx[c, :k] = cells[c]
            self.cell_ok[c, :k] = True
            j = len(bnds[c])
            self.bnd_idx[c, :j] = bnds[c]
            self.bnd_ok[c, :j] = True

    # -------------------------------------------------------------- fitting
    @staticmethod
    def _basis(d, h):
        """Nine polynomial terms, distances scaled by h so they are O(1).

        Without the scaling the quadratic columns are h^2 times the linear
        ones and the normal matrix is ill-conditioned on fine meshes.
        """
        e = d / h[:, :, None]
        x, y, z = e[..., 0], e[..., 1], e[..., 2]
        return np.stack([x, y, z,
                         0.5 * x * x, 0.5 * y * y, 0.5 * z * z,
                         x * y, x * z, y * z], axis=-1)

    def _factor(self, p):
        m = self.m
        nc = m.nc
        cc = m.cell_centre

        dc = cc[self.cell_idx] - cc[:, None, :]
        db = m.b_centre[self.bnd_idx] - cc[:, None, :]
        rc = np.linalg.norm(dc, axis=-1)
        rb = np.linalg.norm(db, axis=-1)

        # Local length scale for the non-dimensionalisation.
        h = np.where(self.cell_ok, rc, 0.0).sum(axis=1) / self.cell_ok.sum(axis=1)
        self.h = h
        hb = np.broadcast_to(h[:, None], rc.shape)

        Bc = self._basis(dc, hb)
        Bb = self._basis(db, np.broadcast_to(h[:, None], rb.shape))
        # Padded slots have r = 0; compute on a safe array, then mask.
        wc = np.where(self.cell_ok, np.where(self.cell_ok, rc, 1.0) ** -p, 0.0)
        wb = np.where(self.bnd_ok, np.where(self.bnd_ok, rb, 1.0) ** -p, 0.0)

        B = np.concatenate([Bc, Bb], axis=1)
        w = np.concatenate([wc, wb], axis=1)
        self.B, self.w = B, w
        self.ok = np.concatenate([self.cell_ok, self.bnd_ok], axis=1)

        # Normal equations, one 9x9 per cell, factored once.
        N = np.einsum("ijk,ij,ijl->ikl", B, w, B)
        # Tiny ridge: a cell whose stencil cannot resolve all nine terms then
        # degrades toward the linear fit instead of producing garbage.
        N += 1e-12 * np.trace(N, axis1=1, axis2=2)[:, None, None] * np.eye(self.N_TERMS)
        self.Ninv = np.linalg.inv(N)

    def __call__(self, phi, phi_b):
        dphi_c = np.where(self.cell_ok, phi[self.cell_idx] - phi[:, None], 0.0)
        dphi_b = np.where(self.bnd_ok, phi_b[self.bnd_idx] - phi[:, None], 0.0)
        rhs = np.einsum("ijk,ij,ij->ik", self.B, self.w,
                        np.concatenate([dphi_c, dphi_b], axis=1))
        a = np.einsum("ijk,ik->ij", self.Ninv, rhs)
        return a[:, :3] / self.h[:, None]     # undo the scaling
