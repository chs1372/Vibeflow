# Licensing

Core code (`src/`, `tests/`, `prototype/`): **Apache-2.0**.

External tools are invoked as separate processes, never linked, so their
licences do not propagate into this codebase:

| Tool | Licence | How it is used |
| --- | --- | --- |
| gmsh | GPL-2.0+ | mesh generation, called as a subprocess / via its Python API in a separate script |
| cfMesh | GPL-3.0 | boundary-layer hex meshing, called as a subprocess |
| ParaView | BSD-3 | post-processing |

Linked dependencies are all permissive or LGPL: Kokkos (Apache-2.0),
PETSc (BSD-2), hypre (Apache-2.0 / MIT), CGNS (zlib-style), ADIOS2 (Apache-2.0),
VTK (BSD-3), OpenCASCADE (LGPL-2.1), preCICE (LGPL-3.0).

Before adding any dependency, check that linking it does not force a copyleft
licence onto the core.
