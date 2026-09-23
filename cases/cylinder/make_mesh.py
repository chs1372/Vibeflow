#!/usr/bin/env python3
"""Hexahedral mesh around a circular cylinder, for the Re = 100 wake benchmark.

Structure: an O-grid ring bonded to the cylinder (so the boundary layer is
resolved by cells aligned with the wall) embedded in a rectangular outer
domain, all quads, then extruded one layer in z. Every cell is a hexahedron,
which is what the solver's face-matching topology builder expects.

Domain follows the usual convention for this benchmark: the cylinder sits
10 diameters from the inlet and 25 from the outlet, with 10 either side, so
the far field does not distort the shedding frequency.

Writes a plain text file of points and hex connectivity. The solver reads
that directly; the CGNS path is already covered by its own gate and adding a
conversion step here would only add something else to get wrong.
"""

import os
import sys
from pathlib import Path

import gmsh
import numpy as np


def _env(name, default):
    return float(os.environ.get(name, default))


D = 1.0                        # cylinder diameter
R = D / 2.0
DZ = 0.5 * D                   # one cell thick; slip on both z faces

# Domain and cell sizing, overridable so one script serves both the benchmark
# mesh and a small debug mesh. The defaults put the cylinder 10 diameters from
# the inlet and 25 from the outlet, which is far enough that the far field does
# not shift the shedding frequency.
X_IN = _env("CYL_XIN", -10.0) * D
X_OUT = _env("CYL_XOUT", 25.0) * D
Y_HALF = _env("CYL_YHALF", 10.0) * D
SIZE_MIN = _env("CYL_SIZEMIN", 0.02) * D
SIZE_MAX = _env("CYL_SIZEMAX", 1.2) * D


def build(out_path, show_stats=True):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cylinder")
    occ = gmsh.model.occ

    # Flow domain = rectangle minus the cylinder. An earlier version also
    # built a body-fitted ring and cut it in stages; the staged cuts left the
    # inner disk in the model and gmsh happily meshed the INSIDE of the
    # cylinder (nodes appeared at r = 0.003 where the wall is at r = 0.5).
    # One cut, then a check that nothing lies inside the wall.
    outer = occ.addRectangle(X_IN, -Y_HALF, 0.0, X_OUT - X_IN, 2 * Y_HALF)
    cyl = occ.addDisk(0.0, 0.0, 0.0, R, R)
    occ.cut([(2, outer)], [(2, cyl)])
    occ.synchronize()

    # Graded sizing: fine at the wall, coarse far away.
    gmsh.model.mesh.field.add("Distance", 1)
    # The cylinder wall is the only closed curve of length pi*D.
    target = np.pi * D
    cyl_curves = [c for (d, c) in gmsh.model.getEntities(1)
                  if abs(gmsh.model.occ.getMass(1, c) - target) < 1e-6]
    if not cyl_curves:
        raise RuntimeError("could not identify the cylinder wall curve")
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", cyl_curves)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 400)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", SIZE_MIN)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", SIZE_MAX)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5 * D)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 12.0 * D)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Quads in 2D, then extruded here rather than by gmsh. Mixing the OCC
    # geometry kernel with geo-kernel extrusion produced a mesh with no
    # hexahedra at all; doing the extrusion directly is a dozen lines and
    # leaves nothing to guess about node ordering.
    gmsh.option.setNumber("Mesh.Algorithm", 8)              # frontal-delaunay quads
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)   # force all-quad
    gmsh.model.mesh.generate(2)

    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(nodeTags)
    tags = np.asarray(nodeTags)[order]
    pts2 = np.asarray(coords).reshape(-1, 3)[order]
    remap = {int(t): i for i, t in enumerate(tags)}

    quads = []
    for dim, tag in gmsh.model.getEntities(2):
        etypes, _, enodes = gmsh.model.mesh.getElements(dim, tag)
        for et, en in zip(etypes, enodes):
            if et == 3:                       # 4-node quadrangle
                quads.append(np.vectorize(remap.get)(np.asarray(en).reshape(-1, 4)))
        if et == 2:
            pass
    tri_count = sum(len(np.asarray(en)) // 3
                    for dim, tag in gmsh.model.getEntities(2)
                    for et, en in zip(*gmsh.model.mesh.getElements(dim, tag)[::2])
                    if et == 2)
    gmsh.finalize()

    if not quads:
        raise RuntimeError("no quadrilaterals produced")
    quads = np.vstack(quads)
    if tri_count:
        raise RuntimeError(f"mesh still has {tri_count} triangles; "
                           "the solver's topology builder expects hexahedra only")

    # Extrude: layer 0 at z = 0, layer 1 at z = DZ. VTK hexahedron order is the
    # bottom face counter-clockwise then the top face, which is exactly a
    # gmsh quad followed by its copy.
    n2 = len(pts2)
    pts = np.vstack([np.column_stack([pts2[:, 0], pts2[:, 1], np.zeros(n2)]),
                     np.column_stack([pts2[:, 0], pts2[:, 1], np.full(n2, DZ)])])
    hexes = np.hstack([quads, quads + n2])

    with open(out_path, "w") as fh:
        fh.write(f"{len(pts)} {len(hexes)}\n")
        for p in pts:
            fh.write(f"{p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n")
        for h in hexes:
            fh.write(" ".join(str(int(v)) for v in h) + "\n")

    r = np.hypot(pts[:, 0], pts[:, 1])
    if r.min() < R - 1e-9:
        raise RuntimeError(f"mesh has nodes inside the cylinder: min radius "
                           f"{r.min():.5f} < R = {R}")
    if show_stats:
        print(f"wrote {out_path}")
        print(f"  points {len(pts)}   hexes {len(hexes)}")
        wall = r < R + 1e-9
        print(f"  min radius {r.min():.6f} (cylinder R = {R}); "
              f"{wall.sum()} nodes on the wall")
        print(f"  x range [{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]  "
              f"y range [{pts[:,1].min():.2f}, {pts[:,1].max():.2f}]  "
              f"z range [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
    return len(pts), len(hexes)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "cylinder.hex")
    build(out)
