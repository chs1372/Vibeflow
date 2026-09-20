"""
Non-orthogonal hexahedral mesh generator and finite-volume geometry.

This is the Python reference for the v0 geometry layer. The C++ implementation
in src/mesh must reproduce these quantities to machine precision on the same
input, so the arrays produced here double as unit-test fixtures.

Geometry follows the standard polyhedral decomposition:
  - face centroid / area vector : fan triangulation from the face vertex average
  - cell centroid / volume      : pyramid decomposition from the face centre average
"""

import numpy as np


class HexMesh:
    """Cell-centred FVM mesh on [0,1]^3, optionally skewed.

    Attributes
    ----------
    cell_centre : (nc, 3)
    cell_volume : (nc,)
    owner, neigh : (nif,)   internal face connectivity
    face_area   : (nif, 3)  area vector, points owner -> neighbour
    face_centre : (nif, 3)
    b_cell      : (nbf,)    boundary face owner cell
    b_area      : (nbf, 3)  outward area vector
    b_centre    : (nbf, 3)
    """

    def __init__(self, n, skew=0.0, seed=0):
        self.n = n
        self.skew = skew
        self._build_vertices(n, skew, seed)
        self._build_faces(n)
        self._compute_face_geometry()
        self._compute_cell_geometry()

    # ------------------------------------------------------------------ mesh
    def _build_vertices(self, n, skew, seed):
        nv = n + 1
        h = 1.0 / n
        g = np.linspace(0.0, 1.0, nv)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        v = np.stack([X, Y, Z], axis=-1)

        if skew > 0.0:
            rng = np.random.default_rng(seed)
            d = rng.uniform(-1.0, 1.0, size=v.shape) * skew * h
            # Interior vertices only: the domain boundary stays planar so that
            # Dirichlet faces carry the exact boundary value with no extra error.
            interior = np.zeros(v.shape[:3], dtype=bool)
            interior[1:-1, 1:-1, 1:-1] = True
            v[interior] += d[interior]

        self.vert = v
        self.nv = nv

    def _vid(self, i, j, k):
        nv = self.nv
        return (i * nv + j) * nv + k

    def _build_faces(self, n):
        """Enumerate internal and boundary faces of the structured hex grid."""
        nc = n ** 3
        cid = lambda i, j, k: (i * n + j) * n + k  # noqa: E731

        owner, neigh, fverts = [], [], []
        b_cell, b_fverts = [], []

        # A face normal to +x at index i separates cell (i-1,j,k) from (i,j,k).
        # Its four vertices, ordered so the area vector points along +x.
        for i in range(n + 1):
            for j in range(n):
                for k in range(n):
                    vs = [self._vid(i, j, k), self._vid(i, j + 1, k),
                          self._vid(i, j + 1, k + 1), self._vid(i, j, k + 1)]
                    if 0 < i < n:
                        owner.append(cid(i - 1, j, k)); neigh.append(cid(i, j, k))
                        fverts.append(vs)
                    elif i == 0:
                        b_cell.append(cid(0, j, k)); b_fverts.append(vs[::-1])
                    else:
                        b_cell.append(cid(n - 1, j, k)); b_fverts.append(vs)

        for j in range(n + 1):
            for i in range(n):
                for k in range(n):
                    vs = [self._vid(i, j, k), self._vid(i, j, k + 1),
                          self._vid(i + 1, j, k + 1), self._vid(i + 1, j, k)]
                    if 0 < j < n:
                        owner.append(cid(i, j - 1, k)); neigh.append(cid(i, j, k))
                        fverts.append(vs)
                    elif j == 0:
                        b_cell.append(cid(i, 0, k)); b_fverts.append(vs[::-1])
                    else:
                        b_cell.append(cid(i, n - 1, k)); b_fverts.append(vs)

        for k in range(n + 1):
            for i in range(n):
                for j in range(n):
                    vs = [self._vid(i, j, k), self._vid(i + 1, j, k),
                          self._vid(i + 1, j + 1, k), self._vid(i, j + 1, k)]
                    if 0 < k < n:
                        owner.append(cid(i, j, k - 1)); neigh.append(cid(i, j, k))
                        fverts.append(vs)
                    elif k == 0:
                        b_cell.append(cid(i, j, 0)); b_fverts.append(vs[::-1])
                    else:
                        b_cell.append(cid(i, j, n - 1)); b_fverts.append(vs)

        self.nc = nc
        self.owner = np.array(owner, dtype=np.int64)
        self.neigh = np.array(neigh, dtype=np.int64)
        self.fverts = np.array(fverts, dtype=np.int64)
        self.b_cell = np.array(b_cell, dtype=np.int64)
        self.b_fverts = np.array(b_fverts, dtype=np.int64)

    # -------------------------------------------------------------- geometry
    @staticmethod
    def _quad_geometry(pts):
        """Area vector and centroid of quads by fan triangulation.

        pts : (nf, 4, 3)
        """
        avg = pts.mean(axis=1)                       # (nf, 3)
        area = np.zeros_like(avg)
        cmom = np.zeros_like(avg)
        wsum = np.zeros(len(avg))
        for t in range(4):
            a = pts[:, t, :] - avg
            b = pts[:, (t + 1) % 4, :] - avg
            tri = 0.5 * np.cross(a, b)               # triangle area vector
            mag = np.linalg.norm(tri, axis=1)
            tc = (avg + pts[:, t, :] + pts[:, (t + 1) % 4, :]) / 3.0
            area += tri
            cmom += mag[:, None] * tc
            wsum += mag
        centre = cmom / wsum[:, None]
        return area, centre

    def _compute_face_geometry(self):
        flat = self.vert.reshape(-1, 3)
        self.face_area, self.face_centre = self._quad_geometry(flat[self.fverts])
        self.b_area, self.b_centre = self._quad_geometry(flat[self.b_fverts])

    def _compute_cell_geometry(self):
        nc = self.nc
        # Reference point: average of the centres of the faces of each cell.
        acc = np.zeros((nc, 3))
        cnt = np.zeros(nc)
        np.add.at(acc, self.owner, self.face_centre)
        np.add.at(cnt, self.owner, 1.0)
        np.add.at(acc, self.neigh, self.face_centre)
        np.add.at(cnt, self.neigh, 1.0)
        np.add.at(acc, self.b_cell, self.b_centre)
        np.add.at(cnt, self.b_cell, 1.0)
        xavg = acc / cnt[:, None]

        vol = np.zeros(nc)
        mom = np.zeros((nc, 3))

        def accumulate(cells, cf, sf, sign):
            """Pyramid from xavg[cell] over the face."""
            d = cf - xavg[cells]
            pv = sign * np.einsum("ij,ij->i", d, sf) / 3.0
            pc = 0.75 * cf + 0.25 * xavg[cells]
            np.add.at(vol, cells, pv)
            np.add.at(mom, cells, pv[:, None] * pc)

        accumulate(self.owner, self.face_centre, self.face_area, +1.0)
        accumulate(self.neigh, self.face_centre, self.face_area, -1.0)
        accumulate(self.b_cell, self.b_centre, self.b_area, +1.0)

        self.cell_volume = vol
        self.cell_centre = mom / vol[:, None]

    # ----------------------------------------------------------------- checks
    def closure_error(self):
        """Sum of outward area vectors per cell. Must be ~0 for a closed cell."""
        s = np.zeros((self.nc, 3))
        np.add.at(s, self.owner, self.face_area)
        np.add.at(s, self.neigh, -self.face_area)
        np.add.at(s, self.b_cell, self.b_area)
        return np.abs(s).max()

    def non_orthogonality(self):
        """Max angle in degrees between the owner->neighbour vector and the face normal."""
        d = self.cell_centre[self.neigh] - self.cell_centre[self.owner]
        sf = self.face_area
        cos = np.einsum("ij,ij->i", d, sf) / (
            np.linalg.norm(d, axis=1) * np.linalg.norm(sf, axis=1))
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max()
