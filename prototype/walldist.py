"""
Wall distance: the exact distance from a point to the nearest wall face
(ADR-042).

Each wall face is split into the four triangles its geometry uses (a fan from
the vertex average, as mesh._quad_geometry does), and the distance to a
triangle is the distance to its closest point, found by the region tests of
Ericson, Real-Time Collision Detection, section 5.1.5. Brute force over all
wall triangles: once per mesh, and the gates' meshes are small.
"""

import numpy as np


def wall_triangles(mesh, wall):
    """(nt, 3, 3): the triangles of the boundary faces flagged in wall."""
    verts = mesh.vert.reshape(-1, 3)
    pts = verts[mesh.b_fverts[np.asarray(wall, dtype=bool)]]      # (nw, 4, 3)
    avg = pts.mean(axis=1)
    tris = [np.stack([avg, pts[:, t], pts[:, (t + 1) % 4]], axis=1) for t in range(4)]
    return np.concatenate(tris) if len(pts) else np.zeros((0, 3, 3))


def _dot(u, v):
    return np.einsum("...k,...k->...", u, v)


def point_triangle_distance(p, tri):
    """Distances from points p (P, 3) to triangles tri (T, 3, 3): (P, T)."""
    p = p[:, None, :]
    a, b, c = tri[None, :, 0], tri[None, :, 1], tri[None, :, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    bp = p - b
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    cp = p - c
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    def safe(num, den):
        return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=den != 0)

    # Barycentric (v, w) of the closest point, region by region, in the order
    # Ericson tests them; np.select takes the first region that holds.
    in_a = (d1 <= 0) & (d2 <= 0)
    in_b = (d3 >= 0) & (d4 <= d3)
    in_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    in_c = (d6 >= 0) & (d5 <= d6)
    in_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    in_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    denom = va + vb + vc
    t_ab = safe(d1, d1 - d3)
    t_ac = safe(d2, d2 - d6)
    t_bc = safe(d4 - d3, (d4 - d3) + (d5 - d6))
    regions = [in_a, in_b, in_ab, in_c, in_ac, in_bc]
    v = np.select(regions, [0.0, 1.0, t_ab, 0.0, 0.0, 1.0 - t_bc], safe(vb, denom))
    w = np.select(regions, [0.0, 0.0, 0.0, 1.0, t_ac, t_bc], safe(vc, denom))
    q = a + v[..., None] * ab + w[..., None] * ac
    return np.linalg.norm(p - q, axis=-1)


def wall_distance(mesh, wall, points=None, chunk=512):
    """Distance from each point (default: every cell centre) to the nearest
    face flagged in wall (one flag per boundary face)."""
    pts = mesh.cell_centre if points is None else np.asarray(points, float)
    tri = wall_triangles(mesh, wall)
    if len(tri) == 0:
        return np.full(len(pts), np.inf)
    out = np.empty(len(pts))
    for s in range(0, len(pts), chunk):
        out[s:s + chunk] = point_triangle_distance(pts[s:s + chunk], tri).min(axis=1)
    return out
