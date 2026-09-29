#!/usr/bin/env python3
"""The NASA TMR backward-facing-step grids as Vibeflow meshes (ADR-045).

Reads one of TMR's four-zone 2D PLOT3D grids (tmr/backstep5_<n>levdn.p2dfmt.gz,
from https://tmbwg.github.io/turbmodels/backstep_grids.html), merges the
zones' interface points -- one-to-one, the same coordinates on both sides --
and writes the plain .hex format PolyMesh reads: the grid extruded one cell in
z, the points on z = 0 then z = DZ, each hexahedron its quad
(counter-clockwise in x-y) followed by the quad's copy.

The step height is 1. The channel runs from x = -130 to 50; upstream of the
step at x = 0 it spans 1 <= y <= 9, behind it 0 <= y <= 9. Zones: 1, x from
-130 to -4 above the upstream wall; 2, -4 to 0; 3, 0 to 8 behind the step; 4,
8 to 50. TMR's neutral map files (tmr/*.nmf) mark both walls slip for
x < -110 and no-slip after, and the step's face no-slip.

Gate 1 of ADR-045 is checked here, per level: the cell count is the zones'
sum, every interface point is merged (the number of distinct points is the
zones' total less the shared ones), and the cells' area is 1,490 exactly.

Run:  python3 make_mesh.py [level ...]      levels 1 2 3 4 (default 4 3 2 1),
      each written to L<level>.hex beside this script.
"""

import gzip
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DZ = 0.01          # one cell thick; the z faces are zero-gradient, so any thickness
AREA = 130.0 * 8.0 + 50.0 * 9.0


def read_p2d(path):
    """Zones as (x, y) arrays of shape (nj, ni), i along x and j along y."""
    with gzip.open(path, "rt") as fh:
        tok = fh.read().split()
    nb = int(tok[0])
    dims = [(int(tok[1 + 2 * b]), int(tok[2 + 2 * b])) for b in range(nb)]
    p = 1 + 2 * nb
    zones = []
    for ni, nj in dims:
        n = ni * nj
        x = np.array(tok[p:p + n], dtype=float).reshape(nj, ni); p += n
        y = np.array(tok[p:p + n], dtype=float).reshape(nj, ni); p += n
        zones.append((x, y))
    return zones


def merge(zones, tol=1e-9):
    """One point list for all zones and, per zone, each (j, i) node's index in
    it. Points closer than tol (the interfaces) become one."""
    pts, index = [], []
    key = {}
    for x, y in zones:
        idx = np.empty(x.shape, dtype=np.int64)
        for j in range(x.shape[0]):
            for i in range(x.shape[1]):
                k = (round(x[j, i] / tol), round(y[j, i] / tol))
                if k not in key:
                    key[k] = len(pts)
                    pts.append((x[j, i], y[j, i]))
                idx[j, i] = key[k]
        index.append(idx)
    return np.array(pts), index


def write_hex(pts2, index, out):
    n2 = len(pts2)
    hexes = []
    for idx in index:
        nj, ni = idx.shape
        for j in range(nj - 1):
            for i in range(ni - 1):
                q = [idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]]
                hexes.append(q + [v + n2 for v in q])
    with open(out, "w") as fh:
        fh.write(f"{2 * n2} {len(hexes)}\n")
        for k in (0, 1):
            for x, y in pts2:
                fh.write(f"{x:.17g} {y:.17g} {k * DZ:.17g}\n")
        for h in hexes:
            fh.write(" ".join(str(v) for v in h) + "\n")
    return len(hexes)


def check(zones, pts2, index):
    """Gate 1: the cells are the zones' cells, the interfaces merged, the area
    the channel's."""
    cells = sum((x.shape[0] - 1) * (x.shape[1] - 1) for x, _ in zones)
    total = sum(x.size for x, _ in zones)
    # Shared nodes: zone 1's last column is zone 2's first; zone 2's last
    # column is part of zone 3's first (above the step); zone 3's last column
    # is zone 4's first.
    shared = zones[0][0].shape[0] + zones[1][0].shape[0] + zones[2][0].shape[0]
    area = 0.0
    for idx in index:
        q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1)
        xq, yq = pts2[q, 0], pts2[q, 1]            # (nj-1, ni-1, 4)
        a = 0.5 * np.sum(xq * np.roll(yq, -1, axis=-1) - np.roll(xq, -1, axis=-1) * yq, axis=-1)
        if (a <= 0.0).any():
            raise ValueError("a cell is not counter-clockwise")
        area += a.sum()
    ok = len(pts2) == total - shared and abs(area - AREA) <= 1e-12 * AREA
    return cells, total - shared, len(pts2), area, ok


def main():
    levels = [int(a) for a in sys.argv[1:]] or [4, 3, 2, 1]
    allok = True
    for lv in levels:
        zones = read_p2d(HERE / "tmr" / f"backstep5_{lv}levdn.p2dfmt.gz")
        pts2, index = merge(zones)
        cells, want, got, area, ok = check(zones, pts2, index)
        n = write_hex(pts2, index, HERE / f"L{lv}.hex")
        assert n == cells
        allok &= ok
        print(f"level {lv}: {cells} cells, {got} points (zones less shared: {want}), "
              f"area {area:.15g} (channel {AREA:g}): {'PASS' if ok else 'FAIL'} -> L{lv}.hex")
    print(f"backstep mesh GATE: {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
