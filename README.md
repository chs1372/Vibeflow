# nsflow

3D unstructured finite-volume Navier–Stokes solver, assembled from open-source
libraries, targeting workstation CPU + GPU.

*(Name is a placeholder — rename before the first public push.)*

## Status

| Stage | Scope | State |
| --- | --- | --- |
| v0 | Mesh geometry, FVM diffusion operator, linear-solver plumbing | **Python reference passing; C++ interfaces fixed** |
| v1 | Incompressible laminar 3D (fractional step / SIMPLE) | not started |
| v2 | RANS turbulence + heat transfer + buoyancy | not started |
| v2.5 | GPU port | not started |
| v3 | Compressible (density-based path) | not started |
| v4 | Multiphase VOF | not started |

## The rule this project runs on

**A stage's verification gates are written before that stage's solver code, and
the next stage does not start until they pass.** CFD code that is wrong still
runs and still produces convincing pictures; the gate suite is the only thing
that separates the two.

Run the active gates:

```
python tests/mms/run_gates.py v0
```

### v0 gate — current result

Two manufactured solutions on `[0,1]^3` with Dirichlet boundaries, each run on
an orthogonal mesh and on a randomly skewed mesh with 25.65 deg maximum
non-orthogonality. Case B is required because case A vanishes on the whole
boundary and therefore never exercises the Dirichlet coefficient.

| Case | Mesh | N=8 | N=16 | N=32 | observed order |
| --- | --- | --- | --- | --- | --- |
| A `sin(pi x) sin(pi y) sin(pi z)` | orthogonal | 4.579e-03 | 1.138e-03 | 2.841e-04 | **2.002** |
| A | skewed 25.65 deg | 5.607e-03 | 1.338e-03 | 3.252e-04 | **2.041** |
| B `exp(x + y/2) sin(pi z + 3/10)` | orthogonal | 6.772e-03 | 1.719e-03 | 4.328e-04 | **1.990** |
| B | skewed 25.65 deg | 8.007e-03 | 2.065e-03 | 5.286e-04 | **1.966** |

Mesh closure error is at machine precision and cell volumes sum to exactly 1.0
on both meshes.

## Layout

```
prototype/    Python reference implementation -- defines correct behaviour
src/core/     Kokkos types, MPI, configuration
src/mesh/     unstructured mesh, geometry, CGNS reader
src/field/    field containers, boundary conditions
src/discretization/  gradients, flux schemes, non-orthogonal correction
src/linalg/   LinearSystem + PETSc/hypre/AmgX backends
src/physics/  transport equations, turbulence models
src/io/       CGNS, ADIOS2, ParaView Catalyst
tests/        unit tests and verification gates
```

CMake target dependencies enforce the layering: a layer may only link layers
below it.

## Why a Python prototype

Every numerical scheme is written in Python first, verified against its
manufactured solution, and only then implemented in C++ with the Python output
as the unit-test fixture. This is what makes it safe to let an AI assistant
rewrite solver internals: the reference and the gate both already exist.

## Build

```
spack env activate -d .
spack install
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

GPU build: see the commented specs at the bottom of `spack.yaml`, then
`-DNSFLOW_ENABLE_GPU=ON`.

## Licence

Apache-2.0 for the core. See `LICENSE.md` — GPL tools (gmsh, cfMesh) are
invoked as subprocesses, never linked.
