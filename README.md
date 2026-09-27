# Vibeflow

A three-dimensional finite-volume Navier–Stokes solver for unstructured
meshes, in C++20 on [Kokkos](https://github.com/kokkos/kokkos), assembled
from open-source libraries rather than forked from an existing code: MPI for
domain decomposition, PETSc and hypre for the pressure solve, CGNS for mesh
input, gmsh for meshing, ParaView for post-processing.

It is built by vibe coding — an AI assistant writes the implementation — under
one rule that makes that acceptable for numerical code:

> **A stage's verification gates are written before the code they judge, and a
> gate is never loosened to let a defect through.**

CFD code that is wrong still runs and still produces convincing pictures. The
gate suite is the only thing that tells the two apart.

## Status

| Stage | Scope | State |
| --- | --- | --- |
| v0 | mesh geometry, FVM diffusion, CGNS input, ParaView output, MPI decomposition, linear-solver backends | **complete** — 8 gates |
| v1 | incompressible laminar flow: PISO/PIMPLE, Rhie–Chow, BDF2, inlet/outlet boundaries, MPI | **complete** — 13 gates |
| v2 | RANS turbulence, heat transfer, buoyancy | not started |
| v2.5 | GPU build | not started |
| v3 | compressible flow | not started |
| v4 | multiphase (VOF) | not started |

All 21 gates pass on the current commit, run from a fresh clone on Ubuntu
24.04 with two cores. CI runs the v0 gates on every push.

## What the gates show

**Order of accuracy against manufactured and exact solutions.** Observed
order from L2 errors on 8³, 16³ and 32³ cells (6³, 12³ and 24³ for the Python
Navier–Stokes study). "Distorted" is a random vertex perturbation reaching
25.7° of non-orthogonality for diffusion, and a smooth mapping reaching
29–34° for the rest.

| Gate | Orthogonal | Distorted |
| --- | --- | --- |
| diffusion, Python and C++ (two solutions each) | 2.002, 1.990 | 2.041, 1.966 |
| convection–diffusion at cell Péclet up to 12, Python and C++ | 2.042 | 2.040 |
| Navier–Stokes vs Ethier–Steinman, C++ | 1.991 | 1.964 |
| Navier–Stokes vs Ethier–Steinman, Python | 1.990 | 1.897 |
| BDF2 in time, Python | 2.428 | |

**Independent implementations agree.** The Python reference (numpy, sparse
direct solves) and the C++ solver (Kokkos, Krylov solvers) share no code.
Their L2 errors agree to 1.6e-12 for diffusion, 4.6e-13 for
convection–diffusion and 6.6e-11 for Navier–Stokes. Native CG and four PETSc
configurations solve the same system to the same answer within 1.4e-13;
BoomerAMG takes 7 iterations where Jacobi-preconditioned CG takes 92.

**Parallel runs give the serial answer.** Diffusion agrees across 1–4 ranks
to 4.5e-14 and Navier–Stokes to 1.7e-15. Each rank builds only its own
subdomain: on four ranks the busiest one constructs 0.288 of the serial mesh,
against a bar of 0.600 that a replicating partitioner (1.000) fails.

**Open boundaries conserve mass to round-off.** Uniform flow through a box
with an inlet and an outlet is an exact discrete fixed point, and the solver
holds it there to within 3e-15 on Cartesian and distorted meshes. With a
profiled, unsteady inflow the cell divergence stays below 2e-14 and the
global mass imbalance below 3e-15.

**Benchmarks against other people's numbers.**

Lid-driven cavity, 64×64, against Ghia, Ghia & Shin (1982):

| Re | rms(u) | rms(v) |
| --- | --- | --- |
| 100 | 0.0016 | 0.0044 |
| 1000 (opt-in, `--full`) | 0.0121 | 0.0126 |

Cylinder wake at Re = 100, the first unsteady case, on gmsh meshes:

| | 6,763 cells | 21,811 cells | literature |
| --- | --- | --- | --- |
| first cell at the wall | D/17 | D/33 | |
| Strouhal number | 0.1698 | 0.1699 | 0.164 (Williamson) |
| mean drag | 1.4233 | 1.3991 | 1.32–1.36 |
| lift amplitude | 0.3671 | 0.3491 | 0.30–0.35 |

The benchmark domain is small on purpose — inlet 10 D upstream, sides at
±10 D — and part of the excess is confinement. Widening the sides to ±40 D
lowers the drag by 0.018; moving the inlet to 20 D upstream lowers the
Strouhal number by 0.0013. Both are read against the scatter that node
placement alone produces at 6,763 cells, ±0.0006 in St and ±0.005 in drag
(one standard deviation), by repeating each run on independently placed
meshes.

One more gate runs the refined cylinder at a time step where an earlier form
of the Rhie–Chow flux let pressure and velocity decouple, and it judges the
fastest cell in the domain rather than a norm, because a norm is how that
defect went unnoticed.

## Known limits

- **The Strouhal number is 3–4% high on the benchmark domain, and the drag
  above the band.** Confinement explains part of it: the inlet ten
  diameters upstream is worth about 0.0026 of St (ADR-032) and the sides at
  ±10 D up to 0.0015 more and 0.022 of drag (ADR-029). Unconfined, St would
  be 1–2% above Williamson's value. Doubling the wall resolution lowers the
  drag by 0.024 without moving St (ADR-027); a three-level grid study is
  running (ADR-033).
- **The Rhie–Chow old-flux term misbehaves at small time steps.** At
  dt = 0.025 it lets a spurious velocity of up to 2.1 grow in the far wake
  and makes the wake values drift with dt; with it off, the drift nearly
  disappears (ADR-030, ADR-031). At the dt = 0.05 used everywhere else the
  flow stays physical. A dt-consistent form, with the dt-independence gate
  ADR-010 asked for written first, is pending.
- Hexahedral cells only.
- Written on Kokkos throughout, but no GPU build has been attempted yet.
- Every rank still reads the whole mesh description (points and
  connectivity), though it builds only its own part.
- The pressure stage is 85% of the cost of a time step (6,763-cell cylinder:
  0.6 s per step on two cores).

## Build

Tested configuration: Ubuntu 24.04, GCC 13.3, CMake 3.28, OpenMPI 4.1,
PETSc 3.19 with hypre 2.28, CGNS 3.4, Kokkos 5.2.2, Python 3.11 with
numpy 2.4 and gmsh 4.15.2. CI builds the same stack with Python 3.12.

```sh
sudo apt-get install -y build-essential cmake git pkg-config python3-venv \
    libopenmpi-dev openmpi-bin libcgns-dev libpetsc-real-dev libhypre-dev
python3 -m venv .venv && . .venv/bin/activate
pip install numpy scipy meshio h5py sympy gmsh

# Kokkos 5.2.2, serial and OpenMP back ends
git clone --depth 1 --branch 5.2.2 https://github.com/kokkos/kokkos.git
cmake -S kokkos -B kokkos-build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/kokkos-install \
    -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DCMAKE_CXX_STANDARD=20
cmake --build kokkos-build -j && cmake --install kokkos-build

# Vibeflow
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKokkos_ROOT=$HOME/kokkos-install
cmake --build build -j
```

Kokkos is the only hard dependency. MPI, CGNS and PETSc are detected at
configure time; without one, its code path is left out and the gates that
need it are reported as skipped. The gmsh wheel loads X11 and OpenGL
libraries even when nothing is drawn; a headless machine needs
`libgl1 libglu1-mesa libxcursor1 libxft2 libxinerama1`. `spack.yaml`
describes a fuller dependency set but has not been exercised yet.

## Run the gates

```sh
python3 tests/mms/run_gates.py v0     # about a minute
sh cases/cylinder/make_meshes.sh      # the two cylinder meshes, about 15 s
python3 tests/mms/run_gates.py v1     # just under an hour on two cores
```

The cylinder meshes are generated rather than stored. With gmsh 4.15.2 the
script reproduces, byte for byte, the meshes behind the numbers above.

## Run a case

```sh
build/tests/cylinder cases/cylinder/debug.hex 0.05 200    # mesh, dt, end time
```

It prints drag, lift, divergence and Courant number as it goes, then the
Strouhal number, mean drag and lift amplitude against the gate's bands,
and writes `cylinder_final.vtu` with velocity and pressure for ParaView. The
pressure solver is PETSc CG with BoomerAMG when PETSc is available
(`VIBEFLOW_PRESSURE=native` selects the built-in CG). The case is described
at the top of `tests/benchmark/cylinder.cpp`; its other settings are `CYL_*`
environment variables read in the same file.

## Layout

```
src/core/            Kokkos types, MPI communicator
src/mesh/            geometry, generated meshes, .hex and CGNS readers, partitioning
src/linalg/          linear systems, native CG and BiCGStab, PETSc/hypre back end
src/discretization/  gradients, convection and diffusion operators, face fluxes
src/physics/         PISO/PIMPLE solver: momentum, Rhie–Chow, pressure, BDF2
src/io/              VTK XML output (.vtu / .pvtu) for ParaView
prototype/           Python reference implementation of every scheme
tests/               unit tests, MMS gates, benchmarks, the gate runner
cases/cylinder/      gmsh mesh generator for the wake benchmark
tools/               fixture writers and a mesh-quality report
docs/                decision log, licensing
```

CMake target dependencies enforce the layering: a layer may link only the
layers below it.

## Decision log

The architecture, the stage plan and a record of each round of work, in
Korean, are in [`ROADMAP.md`](ROADMAP.md), a copy of the living roadmap
document kept in step with it.

[`docs/DECISIONS.md`](docs/DECISIONS.md) is append-only: 33 entries, each
saying what was decided, why, and what would reverse it. It keeps the wrong
turns too, marked where later entries corrected them. Two runs of entries are
worth reading as a story:

- **ADR-022 → ADR-026.** The refined cylinder mesh diverged. The divergence
  was put down to a Courant limit, then the deferred correction, the outer
  iteration, the diffusion correction and the Choi term, each ruled out by its
  own run. It was a checkerboard mode that the Rhie–Chow flux itself created,
  found by a probe that followed one cell term by term instead of a norm over
  the whole field.
- **ADR-012 and ADR-013.** A quadratic least-squares gradient was built to
  fix a disappointing order of accuracy. It is a second-order operator and it
  made the solver slightly worse; the real problem was that the mesh family
  was not a refinement of one geometry. The gradient stays in the tree as
  recorded negative evidence.

## How it is built

Every scheme is written in Python first (`prototype/`), verified against its
manufactured or exact solution, and only then written in C++, with the
Python output as the C++ test fixture. The cross-check gates keep the two
honest against each other.

The code is written with Claude (Anthropic) as the implementing assistant;
each commit carries a `Co-Authored-By` trailer. The rule at the top of this
page is why that works: the reference and the gate exist before the code
does.

## Licence

Apache-2.0 — see [`LICENSE`](LICENSE). How gmsh and the other GPL tools are
kept out of the solver is in [`docs/LICENSING.md`](docs/LICENSING.md).
