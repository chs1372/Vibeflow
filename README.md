# Vibeflow

3D unstructured finite-volume Navier–Stokes solver, assembled from open-source
libraries, targeting workstation CPU + GPU.

*(Name is a placeholder — rename before the first public push.)*

## Status

| Stage | Scope | State |
| --- | --- | --- |
| v0 | Mesh geometry, FVM diffusion, IO, MPI, linear-solver backends | **complete — 8 gates passing** |
| v1 | Incompressible laminar 3D (SIMPLE/PISO, Rhie-Chow, BDF2) | **Python and C++ passing** |
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

Four gates are active. The C++ ones are skipped with a notice if the build tree
is absent, so the suite still runs without Kokkos.

| Gate | What it proves |
| --- | --- |
| `python: MMS diffusion` | the scheme is second-order on orthogonal and skewed meshes |
| `c++: mesh geometry` | the C++ geometry reproduces the Python reference on identical vertices |
| `c++: MMS diffusion` | the C++ scheme is second-order, independently measured |
| `cross-check` | Python and C++ L2 errors agree to 1.6e-12 across all 12 combinations |
| `cgns` | a mesh read from CGNS gives identical geometry, with faces rediscovered by vertex matching |
| `vtu` | ParaView output round-trips through meshio, an independent parser |
| `mpi` | the answer is independent of the rank count (1/2/3/4 agree to 9e-14) |
| `backends` | native CG and four PETSc configurations give the same solution to 1.4e-13 |

v1 gates (`python tests/mms/run_gates.py v1`):

| Gate | State |
| --- | --- |
| `symbolic` | sympy confirms Ethier-Steinman satisfies Navier-Stokes exactly |
| `convection-diffusion` | **PASS** — 2.04 orthogonal, 2.02 distorted, at cell Peclet up to 30 |
| `NS spatial, orthogonal` | **PASS** — 1.98 |
| `NS spatial, distorted` | **PASS** — 1.82 and rising (1.90 at 16→32), 34 deg non-orthogonality |
| `NS temporal (BDF2)` | **PASS** — 2.43 |
| `NS spatial, warped faces` | reported, not gated — 1.25 |
| `Rhie-Chow dt independence` | reported, not gated — the measurement is inconclusive (ADR-012) |
| `c++: convection-diffusion` | **PASS** — 2.04 orthogonal, 2.04 distorted |
| `c++: Navier-Stokes` | **PASS** — 1.986 orthogonal and 1.895 distorted at 8/16/32 |
| `cross-check: convection` | Python and C++ agree to 4.6e-13 over six cases |
| `cross-check: Navier-Stokes` | agree to 2.0e-5 at six outer iterations, 1.5e-6 at forty |
| `benchmark: Ghia cavity` | **PASS** — rms 0.0016 (Re=100) and 0.0123 (Re=1000) at 64x64 |

The skewed-mesh failure is left failing on purpose. was mostly a broken
measurement, and finding that out took building the thing that was supposed to
fix it.

A randomly perturbed mesh redraws its perturbation at every resolution, so the
meshes are independent samples rather than refinements of one another. Their
non-orthogonality wanders (25.9, 30.1, 28.4, 41.7 deg) instead of converging,
and the measured order wanders with it. On a smoothly distorted family, which
refines toward one geometry (24.3, 32.0, 34.1, 34.7 deg), the same code
measures 1.82 and rising.

The quadratic least-squares gradient built to fix it IS a second-order
operator — 1.99, against 1.05 for linear least squares and no convergence at
all for Green-Gauss — and it still makes the solver slightly worse (1.75 vs
1.82) and diverges at nu = 1. It is kept in the tree, selectable, as recorded
negative evidence. See ADR-012 and ADR-013.

Warped faces are a separate first-order error source in 3D, so the gated
family extrudes a 2D distortion and keeps every face planar (measured warp
3e-17).

The C++ port reproduces all of this independently. At 8/16/32 it measures
1.977 → 1.986 on the orthogonal family and 1.695 → 1.895 on the distorted
one — the same orders as Python to three decimals, from a different linear
algebra stack.

Two of these carry most of the weight. The **cross-check** compares
implementations that share no code — numpy with a sparse direct solve against
Kokkos with a Jacobi-preconditioned CG — so agreement on twelve independent
cases is evidence one implementation cannot produce alone. The **mpi** gate is
the only thing that catches a wrong halo exchange: a missing exchange still
converges and still looks like a solution, it just quietly gives a different
answer on four ranks than on one.

Measured on 4096 cells, same system, every backend reaching the same solution:

| backend | iterations | L2 vs exact |
| --- | --- | --- |
| native-cg (jacobi) | 92 | 2.064615287e-03 |
| petsc cg+jacobi | 92 | 2.064615287e-03 |
| petsc cg+ilu | 33 | 2.064615287e-03 |
| petsc cg+hypre (BoomerAMG) | **7** | 2.064615287e-03 |
| petsc gmres+hypre | 7 | 2.064615287e-03 |

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
              and generates the fixtures the C++ unit tests check against
src/mesh/     geometry, generated mesh, CGNS reader, domain decomposition
src/field/    field containers, boundary conditions
src/discretization/  gradients, flux schemes, non-orthogonal correction
src/linalg/   LinearSystem, native CG, PETSc/hypre backend
src/physics/  PISO solver: momentum, Rhie-Chow, pressure correction, BDF2
src/io/       VTK XML output (.vtu/.pvtu) for ParaView
src/core/     Kokkos types, MPI communicator
tools/        fixture generators
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

Kokkos is required. MPI, CGNS and PETSc are each optional and detected at
configure time — without one, the corresponding gate is skipped and nothing
else changes.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKokkos_ROOT=<kokkos install>
cmake --build build -j
ctest --test-dir build
```

For the full dependency set (PETSc, hypre, CGNS, ADIOS2, ParaView):

```
spack env activate -d .
spack install
```

GPU build: swap the Kokkos spec at the bottom of `spack.yaml` for a CUDA or HIP
one. No solver code changes — that is what ADR-002 buys.

## Licence

Apache-2.0 for the core. See `LICENSE.md` — GPL tools (gmsh, cfMesh) are
invoked as subprocesses, never linked.
