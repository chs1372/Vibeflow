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
| v1 | incompressible laminar flow: PISO/PIMPLE, Rhie–Chow, BDF2, inlet/outlet boundaries, MPI | **complete** — 15 gates |
| v2a | energy equation, Boussinesq buoyancy, slip walls | **complete** — 8 gates |
| v2b | RANS turbulence: k-ω SST, low-Reynolds wall treatment | not started |
| v2c | wall functions, conjugate heat transfer | not started |
| v2.5 | GPU build | not started |
| v3 | compressible flow | not started |
| v4 | multiphase (VOF) | not started |

All 31 gates pass on the current code, on Ubuntu 24.04 with two cores. CI
runs the v0 gates on every push.

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
| Navier–Stokes vs Ethier–Steinman, C++ | 1.991 | 2.034 |
| Navier–Stokes vs Ethier–Steinman, Python | 1.990 | 1.958 |
| BDF2 in time, Python | 2.428 | |

**Independent implementations agree.** The Python reference (numpy, sparse
direct solves) and the C++ solver (Kokkos, Krylov solvers) share no code.
Their L2 errors agree to 1.5e-12 for diffusion, 4.6e-13 for
convection–diffusion and 5.1e-10 for Navier–Stokes (3e-12 with the C++
non-orthogonal loop converged to 1e-12). Native CG and four PETSc
configurations solve the same system to the same answer within 1.4e-13;
BoomerAMG takes 7 iterations where Jacobi-preconditioned CG takes 92.

**Parallel runs give the serial answer.** Diffusion agrees across 1–4 ranks
to 4.5e-14 and Navier–Stokes to 7.6e-15. Each rank builds only its own
subdomain: on four ranks the busiest one constructs 0.288 of the serial mesh,
against a bar of 0.600 that a replicating partitioner (1.000) fails. Read
from the binary `.vmesh` format, each rank also holds only its share of the
mesh description: 0.403 of the file on four ranks, against the same bar.

**A steady state does not depend on the time step.** Two steady
manufactured problems, on an orthogonal and a skewed mesh, run to a steady
state at dt = 0.02, 0.2 and 2.0 must reach the same discrete state to 1e-6 of
the discretisation error; they agree to 1e-10. This is the trap ADR-010
named in the first week of v1. The first form of the Rhie–Chow old-flux term
fell into it — on the skewed mesh its steady velocity moved by 69% of its
own discretisation error across that range — and in the cylinder's far
wake it grew a spurious velocity at small time steps (ADR-037).

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

Cylinder wake at Re = 100, the first unsteady case, on gmsh meshes. The
gate runs the 6,763-cell mesh; the grid limit is a three-level Richardson
extrapolation over 10,216, 21,811 and 42,020 cells (ADR-033):

| | 6,763 cells, D/17 | grid limit | literature |
| --- | --- | --- | --- |
| Strouhal number | 0.1689 | 0.1697, converged within 0.0004 | 0.164 (Williamson) |
| mean drag | 1.4317 | 1.390 ± 0.004 | 1.32–1.36 |
| lift amplitude | 0.3681 | 0.344 ± 0.003 | 0.30–0.35 |

The grid study was run before the Rhie–Chow change of ADR-037, which moved
the 6,763-cell values by −0.0009 in St and +0.008 in drag; it has not been
repeated with the current flux. Halving the time step to 0.025 moves the
6,763-cell values by 0.0001.

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
defect went unnoticed. The wake gate itself also judges the fastest cell, at
every step of its statistics window, against 1.5: the physical peak beside
the cylinder is 1.36.

**Heat transfer and buoyancy (v2a).** The temperature is transported like a
velocity component and solved inside every outer iteration; the Boussinesq
force −β(T − T_ref)g joins the momentum equation. Orders, C++ (Python):

| Gate | Orthogonal | Distorted |
| --- | --- | --- |
| temperature carried by the exact Ethier–Steinman flow | 1.999 (1.997) | 1.977 (1.964) |
| steady manufactured Boussinesq flow, u | 2.021 (2.044) | 2.013 (2.031) |
| steady manufactured Boussinesq flow, T | 2.003 (2.005) | 2.024 (2.038) |
| Taylor–Green vortex between slip walls | reported | 2.045 (1.962) |
| BDF2 in time for T | 2.153 (2.153) | |

The steady Boussinesq state is the same at dt = 0.2 and 2.0 to 1e-10 of its
discretisation error, and a fluid resting in its reference stratification
stays at rest to within 3e-14. Python and C++ agree to 1e-10 on every v2a
quantity; two to four ranks give the serial answer to 2.3e-15.

The onset of Rayleigh–Bénard convection between rigid plates, from linear
growth rates on 16, 24 and 32 cells across the layer:

| | 16 | 24 | 32 | extrapolated | reference |
| --- | --- | --- | --- | --- | --- |
| critical Rayleigh number | 1678.630 | 1694.884 | 1700.563 | 1707.842 | 1707.762 |

The observed order is 2.005, and the extrapolation is 0.005% from
Chandrasekhar's value.

The differentially heated square cavity of de Vahl Davis (1983), air, run
to a steady state on 32², 64² and 128²:

| Ra | Nu, 128² | Nu, extrapolated | reference Nu | u_max / v_max, 128², vs de Vahl Davis |
| --- | --- | --- | --- | --- |
| 10³ | 1.11787 | 1.11779 | 1.1178 | +0.01% / +0.01% |
| 10⁴ | 2.24603 | 2.24482 | 2.2448 | +0.02% / +0.05% |
| 10⁵ | 4.53101 | 4.52161 | 4.5216 | +0.06% / +0.10% |
| 10⁶ | 8.88498 | 8.81945 | 8.8252 | +0.49% / +0.84% |

Observed orders are 1.90 to 2.01, and every extrapolation is within 0.07% of
the reference.

## Known limits

- **The Strouhal number is 2.8% high on the benchmark domain, and the drag
  above the band.** Most of it is the domain, which is small on purpose:
  the inlet ten diameters upstream is worth about 0.0026 of St (ADR-032) and
  the sides at ±10 D up to 0.0015 more and 0.022 of drag (ADR-029).
  Resolution is none of the Strouhal excess and 0.027 of the drag
  (ADR-033). Taken off, St is 0.3–1.2% above Williamson's value and drag
  about 1.365, just above the band. What remains of the Strouhal excess is
  not explained.
- Hexahedral cells only.
- Written on Kokkos throughout, but no GPU build has been attempted yet.
- The distributed mesh read needs the binary `.vmesh` format, which only
  code writes so far (`vmesh::write`). The text `.hex` and CGNS readers
  still read the whole file on every rank, and the cylinder cases use them.
- The pressure stage is 88% of the cost of a time step (6,763-cell cylinder:
  0.63 s per step on two cores). Restarting the boundary-pressure
  extrapolation cold is part of that price: a warm-started one saves its
  sweeps and doubles the non-orthogonal loop's (ADR-034).
- **Transient accuracy needs a converged outer loop.** The pressure
  correction sees only the momentum diagonal, so at a diffusion number
  dt·ν/h² near ten the PIMPLE loop contracts slowly, and a fixed handful of
  outer iterations advances the slow modes by a fraction of dt.
  Rayleigh–Bénard growth rates came out up to four times too small with four
  iterations and right with a hundred; the critical Rayleigh numbers did not
  move at all, since steady answers do not depend on it (ADR-038). No gate
  yet measures a transient coupling against an exact rate.
- **Hydrostatic balance is exact only for a matching reference.** Buoyancy
  enters as a cell force, not through the face flux as in a p_rgh
  formulation. A fluid resting in the reference stratification T_ref stays at
  rest to round-off; with a constant T_ref instead, the same resting fluid
  develops currents of 0.5 in units of κ/L within 50 steps on an 8³ mesh at a
  Rayleigh-level forcing of 1700.
- A fixed-heat-flux wall hands the gradient the cell's own temperature: exact
  for the adiabatic walls the gates use, first order in that boundary value
  where a non-zero flux is prescribed.
- A steady run on a distorted mesh wobbles from step to step at about 200
  times the non-orthogonal loop's tolerance, so it cannot be called steady
  below that (ADR-038).

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
python3 tests/mms/run_gates.py v1     # about an hour on two cores
python3 tests/mms/run_gates.py v2     # about two and a half hours
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
src/physics/         PISO/PIMPLE solver: momentum, Rhie–Chow, pressure, BDF2, energy, buoyancy
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

[`docs/DECISIONS.md`](docs/DECISIONS.md) is append-only: 39 entries, each
saying what was decided, why, and what would reverse it. It keeps the wrong
turns too, marked where later entries corrected them. Three runs of entries
are worth reading as a story:

- **ADR-022 → ADR-026.** The refined cylinder mesh diverged. The divergence
  was put down to a Courant limit, then the deferred correction, the outer
  iteration, the diffusion correction and the Choi term, each ruled out by its
  own run. It was a checkerboard mode that the Rhie–Chow flux itself created,
  found by a probe that followed one cell term by term instead of a norm over
  the whole field.
- **ADR-010 → ADR-030 → ADR-031 → ADR-037.** A trap named in the first
  week — the Rhie–Chow flux letting the time step into a steady state — got
  a report instead of a gate, the report pointed the wrong way and was left
  alone. Twenty entries later, halving the time step made the wake drift
  and a spurious velocity grow far downstream. The gate was finally written,
  failed as the report had, and the cause turned out to be two
  interpolations, each second order, that differed by a skewness term the
  old-flux term divided by a number proportional to dt.
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
