#!/usr/bin/env python3
"""Output gate: verify the .vtu written by C++ with an independent reader.

meshio parses the file the same way ParaView would. Checking our own writer with
our own reader would prove nothing; the point is that a third-party parser sees
the geometry and the field values we intended.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "build" / "tests" / "test_vtu"
FIX = ROOT / "tests" / "fixtures"
OUT = ROOT / "build" / "vtu_check"

TOL = 1e-13
fails = []


def check(label, ok, detail=""):
    print(f"  {label:<34}{'PASS' if ok else 'FAIL'}  {detail}")
    if not ok:
        fails.append(label)


def main():
    import meshio

    if not BIN.exists():
        print(f"test_vtu not built ({BIN}) -- skipping"); return 0
    r = subprocess.run([str(BIN), str(FIX), str(OUT)], cwd=ROOT,
                       capture_output=True, text=True,
                       env={"OMP_PROC_BIND": "false", "PATH": "/usr/bin:/bin"})
    if r.returncode != 0:
        print(r.stdout, r.stderr); print("writer FAILED"); return 1

    m = meshio.read(str(OUT) + ".vtu")
    verts = np.loadtxt(FIX / "vertices_n8_s25.txt", skiprows=1)

    print("meshio round-trip of the C++ .vtu")
    check("points count", len(m.points) == len(verts), f"{len(m.points)}")
    check("points exact", np.abs(m.points - verts).max() == 0.0,
          f"max|diff| {np.abs(m.points - verts).max():.1e}")

    cells = m.cells_dict.get("hexahedron")
    check("hexahedron cells", cells is not None and len(cells) == 512,
          f"{0 if cells is None else len(cells)}")

    cd = {k: np.asarray(v["hexahedron"]) for k, v in m.cell_data_dict.items()}
    check("fields present",
          set(cd) == {"scalar", "velocity", "volume", "centre"}, str(sorted(cd)))

    # Volumes must match the reference fixture: this catches a cell-ordering bug
    # in the writer, which a structural check alone would miss.
    volref = np.loadtxt(FIX / "cell_volume.txt", skiprows=1)
    check("volume matches fixture", np.abs(cd["volume"] - volref).max() < TOL,
          f"max|diff| {np.abs(cd['volume'] - volref).max():.1e}")

    ccref = np.loadtxt(FIX / "cell_centre.txt", skiprows=1)
    check("centre matches fixture", np.abs(cd["centre"] - ccref).max() < TOL,
          f"max|diff| {np.abs(cd['centre'] - ccref).max():.1e}")

    # Recompute the analytic fields from the reference centres.
    exp_s = np.sin(3 * ccref[:, 0]) * np.cos(2 * ccref[:, 1]) + ccref[:, 2]
    check("scalar field values", np.abs(cd["scalar"] - exp_s).max() < TOL,
          f"max|diff| {np.abs(cd['scalar'] - exp_s).max():.1e}")
    exp_v = np.column_stack([ccref[:, 1], -ccref[:, 0], 0.5 * ccref[:, 2]])
    check("vector field values", np.abs(cd["velocity"] - exp_v).max() < TOL,
          f"max|diff| {np.abs(cd['velocity'] - exp_v).max():.1e}")

    # Cell ordering: the centroid of each hex's 8 vertices must sit near the
    # cell centre it was written with.
    hexc = m.points[cells].mean(axis=1)
    check("cell order consistent", np.abs(hexc - ccref).max() < 0.02,
          f"max|diff| {np.abs(hexc - ccref).max():.1e}")

    # meshio has no .pvtu reader, so the index is validated structurally: it
    # must be well-formed XML, declare the fields, and every piece it names
    # must exist and parse as a .vtu.
    import xml.etree.ElementTree as ET

    pv = Path(str(OUT) + "_par.pvtu")
    check("pvtu written", pv.exists())
    root = ET.parse(pv).getroot()
    grid = root.find("PUnstructuredGrid")
    pieces = [e.get("Source") for e in grid.findall("Piece")]
    check("pvtu lists both pieces", pieces == ["vtu_check_par_0.vtu", "vtu_check_par_1.vtu"],
          str(pieces))
    names = [e.get("Name") for e in grid.find("PCellData").findall("PDataArray")]
    check("pvtu declares fields", names == ["scalar"], str(names))
    ok = True
    for src in pieces:
        f = pv.parent / src
        if not f.exists():
            ok = False; break
        mp = meshio.read(str(f))
        if len(mp.points) != len(verts) or "scalar" not in mp.cell_data_dict:
            ok = False; break
    check("every piece reads", ok, f"{len(pieces)} pieces")

    print("\nvtu output gate: " + ("PASS" if not fails else f"FAIL ({', '.join(fails)})"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
