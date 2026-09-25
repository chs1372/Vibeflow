# Licensing

Everything in this repository is **Apache-2.0**; the full text is in
[`LICENSE`](../LICENSE) at the repository root.

Copyleft tools are kept outside the solver: nothing in `src/` links them, so
their licences do not propagate into it.

| Tool | Licence | How it is used |
| --- | --- | --- |
| gmsh | GPL-2.0+ | mesh generation: `cases/cylinder/make_mesh.py` imports its Python API; the solver never links it and reads only the `.hex` file that script writes |
| cfMesh | GPL-3.0 | boundary-layer hex meshing, planned, to be called as a subprocess |
| ParaView | BSD-3 | post-processing |

Linked dependencies are all permissive or LGPL: Kokkos (Apache-2.0),
PETSc (BSD-2), hypre (Apache-2.0 / MIT), CGNS (zlib-style), ADIOS2 (Apache-2.0),
VTK (BSD-3), OpenCASCADE (LGPL-2.1), preCICE (LGPL-3.0).

Before adding any dependency, check that linking it does not force a copyleft
licence onto the core.
