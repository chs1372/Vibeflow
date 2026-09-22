# Architecture decision log

Append-only. Each entry: what was decided, why, and what would reverse it.

## ADR-001 — Library assembly rather than forking OpenFOAM or code_saturne
**Decided.** Implement only the discretization and physics layers; take mesh,
linear algebra, IO and visualization from external libraries.
*Why:* OpenFOAM's architecture resists GPU porting; code_saturne carries Fortran
legacy and partial GPU support. A clean Kokkos-based core avoids both.
*Reverses if:* the v1 gate slips past ~9 months, at which point forking
code_saturne is the better trade.

## ADR-002 — Kokkos from v0, not from the GPU stage
**Decided.** All field and mesh storage uses Kokkos views from the first commit.
*Why:* retrofitting a memory layout is a rewrite, not a port.
*Reverses if:* nothing foreseeable. This is the load-bearing decision.

## ADR-003 — Over-relaxed non-orthogonal correction, deferred
**Decided.** Face area vector split as `Sf = Delta + k`, `Delta = d |Sf|^2/(d.Sf)`;
the `k` remainder is deferred-corrected against a least-squares cell gradient.
*Why:* keeps the full face area implicit, so the matrix stays well-conditioned
on skewed meshes. Verified: 2.041 order at 25.65 deg non-orthogonality.
*Reverses if:* corrector sweeps stop converging above ~60 deg, which would push
toward a fully implicit non-orthogonal treatment.

## ADR-004 — Pressure-based and density-based paths behind one FluxScheme interface
**Decided.** `src/discretization/FluxScheme.hpp` is the branch point; physics
and time integration are written against it.
*Why:* v3 compressible support is then an addition, not a rewrite.
*Open:* whether v3 uses a unified all-Mach pressure-based scheme or a separate
density-based path. Decide at the end of v1.

## ADR-005 — Apache-2.0 core, GPL tools as subprocesses
**Decided.** gmsh and cfMesh are called as separate processes.
*Why:* keeps the core relicensable and safe to embed.
*Reverses if:* the project commits to GPL, which would allow linking cfMesh
directly and simplify meshing automation.

## ADR-006 — v0 partitioner reads the whole mesh on every rank
**Decided.** `DistributedMesh` takes a global mesh that every rank already
holds, and keeps its slice. Partitioning is recursive coordinate bisection on
cell centroids — no ParMETIS dependency, deterministic.
*Why:* it makes the parallel-consistency gate possible immediately, and that
gate is what protects every later parallel change.
*Cost, stated plainly:* memory does not scale. A mesh that fits one rank is the
limit, which defeats the point of MPI for large cases.
*Reverses when:* before v2 on real geometry. Replace with a parallel CGNS read
plus ParMETIS/Zoltan2; the `PartitionMethod` enum and the `HaloExchange`
interface are the seams, so nothing above `mesh/` changes.

## ADR-007 — VTK XML output written directly, no VTK library
**Decided.** `io/VtuWriter` emits base64 `.vtu` and `.pvtu` itself.
*Why:* linking VTK for output alone adds a heavy build dependency. The XML
format is stable and ParaView reads it natively. Verified by reading our output
back with meshio, an independent parser.
*Reverses if:* in-situ Catalyst is adopted, which needs the VTK data model
anyway. At that point the writer becomes the file-output path and Catalyst the
in-memory one.

## ADR-008 — Linear-solver backend selected at runtime by string
**Decided.** `PetscSolver` takes "cg+hypre", "gmres+ilu" and so on from the
case; `NativeCG` stays as a dependency-free reference.
*Why:* the pressure solve dominates runtime, so the preconditioner must be
tunable per case without a rebuild, and the GPU path later is the same kind of
switch. Keeping NativeCG means every PETSc result has an independent check —
the backend-equivalence gate measured 1.4e-13 agreement across four PETSc
configurations.
*Measured:* BoomerAMG reaches the same solution in 7 iterations where
Jacobi-CG needs 92, on 4096 cells.

## ADR-009 — Pressure-velocity coupling: SIMPLE with PISO correctors
**Decided.** One momentum predictor followed by two or more pressure
correctors, with under-relaxation available so the same code runs steady
(SIMPLE) and transient (PISO) cases.
*Why:* it covers both regimes from one implementation, and the extra corrector
is what lets a transient run take a time step larger than a fractional-step
method tolerates.
*Cost:* more work per time step than fractional step, and the corrector count
is a tuning knob rather than a fixed property of the scheme.
*Reverses if:* v2 LES becomes the dominant use, where fractional step is
cheaper per step at the small time steps LES needs anyway.

## ADR-010 — Collocated variables with Rhie-Chow face interpolation
**Decided.** Velocity and pressure both at cell centres; the face mass flux
comes from Rhie-Chow interpolation rather than from interpolating velocity.
*Why:* staggered arrangements do not generalise to unstructured meshes. This is
what OpenFOAM, code_saturne and Nalu-Wind all do.
*Known trap, which the gates must cover:* the naive form makes the face flux
depend on the time step through the aP coefficient, because aP carries the
transient term V/dt. As dt shrinks the pressure-damping term vanishes and
checkerboarding returns; worse, a steady state reached with different dt is a
different steady state. The implementation must use the dt-consistent form
that carries the old-time flux, and a gate must solve the same steady problem
at several dt and require the same answer.

## ADR-011 — Second-order implicit time integration (BDF2)
**Decided.** BDF2, with a BDF1 start-up step.
*Why:* second order without the Crank-Nicolson oscillation at large time steps,
and it is L-stable, which matters once stiff source terms arrive with
turbulence models in v2.
*Cost:* one extra stored time level, and the start-up step is first order, so
the temporal order gate must measure over enough steps that the start-up
error does not dominate.
*Reverses if:* nothing foreseeable for v1. A higher-order or IMEX scheme is a
v3 concern.

## ADR-012 — Gradient reconstruction limits velocity accuracy on skewed meshes
**Measured, not yet decided.** The v1 solver reaches second order on an
orthogonal mesh (1.98) but only 1.17 on a randomly perturbed one, while the
pressure on that same run converges at 2.06.

The cause is the gradient reconstruction, measured directly against the
analytic gradient of the exact pressure field on a mesh with 30 deg maximum
non-orthogonality:

| cells per side | Green-Gauss | order | least squares | order |
| --- | --- | --- | --- | --- |
| 6 | 1.64e-01 | – | 1.19e-01 | – |
| 12 | 1.47e-01 | 0.16 | 5.49e-02 | 1.11 |
| 24 | 1.62e-01 | -0.14 | 2.66e-02 | 1.05 |

Green-Gauss does not converge at all on a perturbed mesh; least squares is
first order. The velocity correction `u = H/aP - grad(p) V/aP` uses this
gradient directly, so the velocity cannot be better than the gradient.

*Investigated, and the obvious fix does not work.* A quadratic least-squares
fit over a two-ring stencil (nine terms, about 24 neighbours per hex cell) is
genuinely a second-order gradient operator:

| cells per side | Green-Gauss | order | LSQ linear | order | LSQ quadratic | order |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 1.64e-01 | – | 1.19e-01 | – | 1.78e-01 | – |
| 12 | 1.47e-01 | 0.16 | 5.49e-02 | 1.11 | 4.86e-02 | 1.87 |
| 24 | 1.62e-01 | -0.14 | 2.66e-02 | 1.05 | 1.22e-02 | 1.99 |

It still does not make the solver more accurate. On the smooth mesh family
(ADR-013), velocity order is **1.816 with the linear gradient and 1.752 with
the quadratic one**, and at nu = 1 the quadratic version diverges outright.

Two reasons, both measured:

* A two-ring gradient in the velocity correction is inconsistent with the
  compact pressure Laplacian that produced that correction. The outer loop
  stops contracting: at nu = 1, n = 8, the velocity reaches 29 (the exact
  solution is order 1) after one step and the momentum matrix is singular at
  the next.
* What limits the velocity is not the reconstruction operator. The discrete
  pressure converges at 1.93 and the quadratic reconstruction of the EXACT
  pressure converges at 1.99, but the reconstruction of the SOLVER's pressure
  converges at only 1.32 — the pressure error field carries mesh-scale
  roughness, and differentiating rough data costs an order no matter how good
  the operator is.

*Decided:* weighted linear least squares stays the default. The quadratic
implementation is kept in `prototype/gradient.py` and selectable with
`PisoSolver(gradient="quadratic")` so the measurement above can be reproduced,
not because it is recommended.
*What the 1.23 originally reported actually was:* mostly a broken measurement.
See ADR-013 — on a valid refinement family the same code measures 1.82.
*Confirmed externally:* the ordering of the three operators matches the
literature — Green-Gauss with face averaging is "inconsistent on irregular
grids and fails to achieve first-order accuracy" while least-squares methods
are "at least first-order on arbitrary unstructured grids"
([Advances in Aerodynamics, 2019](https://aia.springeropen.com/articles/10.1186/s42774-019-0020-9)).
That paper also notes solution accuracy does not simply follow gradient
accuracy, which is exactly what happened here.

## ADR-013 — Order studies run on a smoothly distorted mesh, not a random one
**Decided.** The gated distorted-mesh order study uses a fixed analytic
distortion, applied in x-y and extruded in z so every face stays planar.
Randomly perturbed meshes are still run, but reported rather than gated.

*Why:* a randomly perturbed mesh redraws its perturbation at every
resolution, so the meshes are independent samples rather than refinements of
one another. The mesh quality itself wanders, and the measured order wanders
with it:

| cells per side | random perturbation | smooth distortion |
| --- | --- | --- |
| 6 | 25.9 deg | 24.3 deg |
| 12 | 30.1 deg | 32.0 deg |
| 24 | 28.4 deg | 34.1 deg |
| 48 | 41.7 deg | 34.7 deg |

The random column has no limit; the smooth one converges to about 35 deg. An
order measured against a geometry that is not converging is not measuring the
scheme.

*Second reason for planar faces:* in 3D, "simple flux integrations on
non-planar control volume faces lead to first-order solution errors"
([J. Comput. Phys. 230, 2011](https://www.sciencedirect.com/science/article/abs/pii/S0021999111003871)).
Randomly perturbing vertices in all three directions warps every face, and
the relative warp does not shrink under refinement — measured 0.72, 0.79,
0.87 at n = 6, 12, 24. Extruding a 2D distortion keeps every face planar
exactly (measured warp 3e-17), so the study isolates the scheme.

*Measured on the smooth family, linear gradient:*

| refinement | order |
| --- | --- |
| 6 -> 12 | 1.694 |
| 12 -> 24 | 1.816 |

Rising toward 2 as the family approaches its limiting geometry. The same code
on the random family reported 1.23, and on the planar-random family 1.23 as
well: those numbers were measuring the mesh, not the scheme.
*Gate:* the routine run uses 6/12/24 and requires a rising trend rather than a
fixed value, because those resolutions are pre-asymptotic; `--full` uses
8/16/32 and requires 1.8. *Condition to tighten to 1.9:* the C++ solver
reaching n = 64, which the Python prototype cannot do in reasonable time —
n = 32 already takes about ten minutes.

## ADR-014 — Pure-Neumann pressure: project the right-hand side, do not pin a cell
**Decided.** With Dirichlet velocity on every boundary the pressure operator is
singular, its null space the constants. The C++ solver removes the null-space
component from the right-hand side before each solve and from the solution
after it.

*Why not pin a cell:* setting one row to identity makes the matrix
non-symmetric, which rules out CG — and CG is what makes the pressure solve
affordable. The Python reference pins, because it uses a direct solve and does
not care.
*Measured:* without the projection, CG stalls on the null-space component. The
non-orthogonal corrector then chases linear-solver noise, uses all 40 sweeps
and still leaves a continuity residual of 8.9e-9; with it, the same case
converges in 20 sweeps to 1.3e-14.
*Related:* the corrector's own convergence threshold has to sit above the
linear solver's noise floor. At 1e-14 it never converged; 1e-12 does. The
correction is also under-relaxed at 0.7, because undamped it stalled on the
coarsest distorted mesh.

## ADR-015 — Two native Krylov solvers, chosen by symmetry
**Decided.** `NativeCG` for the pressure equation, `NativeBiCGStab` for
momentum.
*Why:* convection makes the momentum matrix non-symmetric. CG on a
non-symmetric system does not fail loudly — it converges to the wrong answer
while reporting a small residual, which is precisely the failure mode this
project's gates exist to prevent. The pressure Laplacian stays symmetric, so
it keeps CG.
*Both remain reference implementations,* not the production path: PETSc with
hypre is (ADR-008). They exist so every gate runs without a PETSc build and
every PETSc result has an independent check.
