"""
Dump the Python reference geometry and MMS solution as plain-text fixtures.

The C++ implementation must reproduce these to machine precision on the same
input. Text (17 significant digits) rather than binary so the fixtures are
diffable and endian-independent; they are small enough that this costs nothing.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mesh import HexMesh                       # noqa: E402
from fvm import DiffusionOperator              # noqa: E402
from mms_diffusion import CASES                # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def write(name, arr):
    arr = np.asarray(arr)
    path = OUT / name
    flat = arr.reshape(len(arr), -1)
    with path.open("w") as fh:
        fh.write(f"{flat.shape[0]} {flat.shape[1]}\n")
        for row in flat:
            fh.write(" ".join(f"{v:.17g}" for v in row) + "\n")
    return path.name, flat.shape


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n, skew, seed = 8, 0.25, 1
    m = HexMesh(n, skew=skew, seed=seed)

    manifest = [f"# nsflow v0 fixtures  n={n} skew={skew} seed={seed}",
                f"# cells={m.nc} internal_faces={len(m.owner)} boundary_faces={len(m.b_cell)}",
                f"# max_nonorthogonality_deg={m.non_orthogonality():.17g}",
                f"# closure_error={m.closure_error():.17g}",
                f"# volume_sum={m.cell_volume.sum():.17g}"]

    for name, arr in [
        ("vertices.txt", m.vert.reshape(-1, 3)),
        ("cell_centre.txt", m.cell_centre),
        ("cell_volume.txt", m.cell_volume[:, None]),
        ("owner.txt", m.owner[:, None]),
        ("neighbour.txt", m.neigh[:, None]),
        ("face_area.txt", m.face_area),
        ("face_centre.txt", m.face_centre),
        ("boundary_cell.txt", m.b_cell[:, None]),
        ("boundary_area.txt", m.b_area),
        ("boundary_centre.txt", m.b_centre),
    ]:
        fn, shape = write(name, arr)
        manifest.append(f"{fn} {shape[0]} {shape[1]}")

    # Vertex sets for every grid the C++ MMS gate runs. The C++ side never
    # regenerates the random perturbation -- reproducing numpy's PCG64 stream in
    # C++ would be pointless coupling. It reads vertices and rebuilds the
    # structured topology from n, exactly as it will later read a CGNS file.
    for nn in (8, 16, 32):
        for sk, tag in ((0.0, "s0"), (0.25, "s25")):
            mm = HexMesh(nn, skew=sk, seed=seed)
            fn, shape = write(f"vertices_n{nn}_{tag}.txt", mm.vert.reshape(-1, 3))
            manifest.append(f"{fn} {shape[0]} {shape[1]} n={nn} skew={sk}")

    # Reference solution of case A on this mesh.
    case = CASES[0]
    op = DiffusionOperator(m, gamma=1.0)
    u, _ = op.solve(case.f(m.cell_centre), case.u(m.b_centre))
    fn, shape = write("solution_caseA.txt", u[:, None])
    manifest.append(f"{fn} {shape[0]} {shape[1]}")
    err = u - case.u(m.cell_centre)
    l2 = np.sqrt((err ** 2 * m.cell_volume).sum() / m.cell_volume.sum())
    manifest.append(f"# solution_caseA_L2_error={l2:.17g}")

    (OUT / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print("\n".join(manifest))


if __name__ == "__main__":
    main()
