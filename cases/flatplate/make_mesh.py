#!/usr/bin/env python3
"""The NASA TMR zero-pressure-gradient flat-plate grids as Vibeflow meshes
(ADR-042, gate 4).

Reads one of TMR's 2D PLOT3D grids (tmr/*.p2dfmt.gz, from
https://tmbwg.github.io/turbmodels/flatplate_grids.html) and writes the plain
.hex format PolyMesh reads: the grid extruded one cell in z, points in (i, j)
order on z = 0 then z = DZ, each hexahedron its quad (counter-clockwise in
x-y) followed by the quad's copy.

The grids span -1/3 <= x <= 2, 0 <= y <= 1; the plate is y = 0, x >= 0,
ahead of it a symmetry plane. x = 0.97008 -- the Cf station -- is a grid line
on every level. Each level is every other point of the next finer one.

Run:  python3 make_mesh.py [level ...]      levels: 35x25 69x49 137x97 273x193 545x385
      (default: all), each written to <level>.hex beside this script.
"""

import gzip
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEVELS = {"35x25": "flatplate_clust2_4levelsdown_35x25.p2dfmt.gz",
          "69x49": "flatplate_clust2_3levelsdown_69x49.p2dfmt.gz",
          "137x97": "flatplate_clust2_2levelsdown_137x97.p2dfmt.gz",
          "273x193": "flatplate_clust2_1leveldown_273x193.p2dfmt.gz",
          "545x385": "flatplate_clust2_0levelsdown_545x385.p2dfmt.gz"}
DZ = 0.01          # one cell thick; slip on both z faces, so any thickness


def read_p2d(path):
    with gzip.open(path, "rt") as fh:
        tok = fh.read().split()
    nblocks, ni, nj = int(tok[0]), int(tok[1]), int(tok[2])
    if nblocks != 1:
        raise ValueError(f"{path}: {nblocks} blocks, expected 1")
    d = np.array(tok[3:3 + 2 * ni * nj], dtype=float)
    x = d[:ni * nj].reshape(nj, ni)
    y = d[ni * nj:].reshape(nj, ni)
    return x, y


def write_hex(x, y, out):
    nj, ni = x.shape
    n2 = ni * nj
    pts = np.zeros((2 * n2, 3))
    for k in (0, 1):
        pts[k * n2:(k + 1) * n2, 0] = x.ravel()
        pts[k * n2:(k + 1) * n2, 1] = y.ravel()
        pts[k * n2:(k + 1) * n2, 2] = k * DZ
    idx = lambda i, j: j * ni + i  # noqa: E731
    hexes = []
    for j in range(nj - 1):
        for i in range(ni - 1):
            q = [idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
            hexes.append(q + [v + n2 for v in q])
    with open(out, "w") as fh:
        fh.write(f"{len(pts)} {len(hexes)}\n")
        for p in pts:
            fh.write(f"{p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n")
        for h in hexes:
            fh.write(" ".join(str(v) for v in h) + "\n")
    return len(pts), len(hexes)


def main():
    levels = sys.argv[1:] or list(LEVELS)
    for lv in levels:
        x, y = read_p2d(HERE / "tmr" / LEVELS[lv])
        # The x-lines are vertical and the y-lines horizontal: every cell is
        # a rectangle, and the plate's leading edge is a grid point.
        assert np.abs(np.diff(x, axis=0)).max() < 1e-12
        np_, nh = write_hex(x, y, HERE / f"{lv}.hex")
        print(f"{lv}: {np_} points, {nh} hexahedra -> {lv}.hex")


if __name__ == "__main__":
    main()
