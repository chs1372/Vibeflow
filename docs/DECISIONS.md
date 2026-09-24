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

## ADR-016 — The pressure cost was a stalling solver, not the discretisation
**Superseded by measurement. The original diagnosis in this slot was wrong and
is kept here because the way it was wrong is the point.**

What was recorded first: the distorted-mesh pressure solve costs 20-40x the
orthogonal one, and the cause is the deferred non-orthogonal correction needing
20-40 sweeps against one. The sweep count was right. The conclusion was not.

Profiling the solvers rather than counting sweeps:

| pressure backend | iterations, n=8 distorted | wall, n=8+16 |
| --- | --- | --- |
| native CG, as written | 2,745,371 | 354 s |
| PETSc CG + Jacobi | 31,688 | 6 s |
| native CG + null-space projection | 39,879 | 18 s |

Same algorithm, same preconditioner, 87x the iterations. The native CG was not
converging at all — it ran to its iteration cap on nearly every solve. The
pressure operator is pure Neumann, so its null space is the constants;
projecting that component out of the right-hand side once is not enough,
because round-off re-injects it every iteration and CG cannot reduce it. The
residual norm then stalls above any tight tolerance. Projecting inside the
iteration fixes it (ADR-017).

*What this cost:* a whole ADR arguing for an implicit non-orthogonal treatment
and an AMG preconditioner, to fix a bug in a reference solver. The lesson is
narrow and worth keeping: a sweep count is not a profile.

*What remains true:* the distorted mesh really does need 24-32 non-orthogonal
sweeps where the orthogonal one needs 1, and that is still the largest
structural cost. It is now roughly a 15x factor rather than 40x.

## ADR-017 — Preconditioner and backend choice for the pressure equation
**Decided.** PETSc is the production backend; the native CG stays as the
dependency-free reference, now with null-space projection.

Measured at n = 32 (32,768 cells), distorted, accumulated pressure-solve time
and iterations for the whole run:

| backend | iterations | pressure time |
| --- | --- | --- |
| native CG (projected) | 204,350 | 231 s |
| PETSc CG + Jacobi | 176,309 | 61 s |
| PETSc CG + BoomerAMG | 9,191 | 36-59 s |

BoomerAMG cuts iterations by 19x, and at this size that does NOT translate
into wall time: the pressure matrix changes at every PISO corrector, so the
AMG hierarchy is rebuilt about 24 times per run and the setup eats the gain.
Iteration counts are deterministic and reliable; the wall times on a
two-core sandbox are not — repeat runs of the same configuration varied by
50%, so no wall-clock claim finer than "about the same" is made here.

Reusing the preconditioner across matrix changes is implemented
(`pcRebuildInterval`) and did not clearly help: never rebuilding after the
first setup was fastest but moved the orthogonal answer by 1.5e-5 and left a
continuity residual of 1.4e-10, which means the solves stopped short of the
requested tolerance. Left at 1 (rebuild every change) until there is a case
large enough to measure it properly.

*Expectation, not yet measured:* AMG's advantage grows with problem size
because Krylov iteration counts grow and AMG's do not. The crossover is
somewhere above 32k cells. That is worth re-measuring on real hardware rather
than guessing at it here.


## ADR-018 — Benchmarks answer a different question than exact solutions
**Decided.** The gate suite carries both, and they are not interchangeable.

An exact or manufactured solution proves the discretisation converges to the
equations we wrote down. It cannot catch writing down the wrong equations, a
sign error in a term that the manufactured source silently absorbs, or a
boundary condition that is self-consistent but not what anyone else means by
"no-slip wall". A published benchmark catches exactly those, because the
reference numbers were produced by someone else's code from their reading of
the same physics.

First benchmark: the lid-driven cavity of Ghia, Ghia & Shin (1982), Tables I
and II. Ghia's case is 2D; it runs here as a one-cell-thick slab with
zero-gradient velocity on the two z faces, which needed a per-face velocity
boundary type (Dirichlet or zero-gradient) and an anisotropic box mesh — both
of which real cases need anyway.

Measured at 64x64 against Ghia's 129x129:

| Re | rms(u) | rms(v) | max abs diff | steps to steady |
| --- | --- | --- | --- | --- |
| 100 | 0.0016 | 0.0044 | 0.0088 | 893 |
| 1000 | 0.0123 | 0.0127 | 0.0223 | 4152 |

*The gate checks steadiness as well as agreement.* A profile read off a state
that is still evolving is not a steady-state result however well it happens to
match, and the first run of this benchmark hit its step cap at Re = 1000 with
du/dt still at 8e-5 — it would have "passed" on the profile alone.

*One artefact worth recording:* the first comparison reported the lid moving
at 0.77 instead of 1.0 and an rms of 0.06. Ghia's tables include the wall
points y = 0 and y = 1, and a profile built from cell centres stops half a
cell short of both, so the interpolation clamped to the nearest cell value.
The solver was right and the comparison was wrong — the same class of mistake
as the mesh-family and decay-interval artefacts in ADR-012 and ADR-013. Three
of the seven wrong answers this project has produced so far were in the
measurement, not the code.

*Re = 1000 is opt-in* (`--full`): it is 88% of the runtime.

## ADR-019 — When to add AMR
**Assessed, not scheduled. Recommendation: after v2, before v4, and only once
three things below are true.**

Adaptive mesh refinement is attractive for every case this solver is aimed at
— the cylinder wake wants cells where the vortices are, a compressible run
wants them at the shock, a VOF run wants them at the interface. It is also the
single change that breaks the most of what is already built, so the question
is not whether but when, and what has to be in place first.

### What AMR breaks here

| Piece | What changes |
| --- | --- |
| `Mesh` interface | Cell and face counts stop being constant. Every field allocation, every `View` sized at construction, and the `nCells`/`nGhost`/`nTotal` contract become time-dependent. |
| Hanging nodes | A refined face meets a coarse one. The face-based operators assume one owner and one neighbour per face; a 2:1 interface has one coarse face against four fine ones. Every flux assembly needs a second path. |
| `LinearSystem` | Face-based storage (diag + upper + lower) assumes a fixed sparsity pattern. Refinement changes it, so the matrix and any preconditioner are rebuilt from scratch, which ADR-017 already shows is the expensive part. |
| Conservation on adapt | Interpolating fields onto a new mesh must conserve mass, momentum and the face flux field simultaneously. A flux field that is divergence-free before adaptation is not afterwards unless the projection is done deliberately. |
| `DistributedMesh` | Refinement unbalances the partition, so AMR and load rebalancing arrive together, not separately. |
| Every gate | Order studies assume a fixed refinement family. An adaptive run has no single `h`, so the existing spatial gates cannot express what correctness means for it. |

### What has to exist first

1. **A parallel partitioner that can be re-run cheaply** (ADR-006 replacement).
   The current one reads the whole mesh on every rank; rebalancing on top of
   that is meaningless.
2. **A conservative field-transfer step with its own gate** — refine and
   coarsen a uniform flow and require the mass, momentum and discrete
   divergence to be unchanged to machine precision. That gate is cheap to
   write and would catch most of what goes wrong.
3. **A second-order treatment of hanging-node faces, verified on its own.**
   The natural test is the existing MMS diffusion gate run on a mesh with one
   refinement level in the middle: if it does not stay second order there, no
   amount of adaption logic will help.

### Why not sooner

* Before v2 the solver has no case whose cost AMR would actually relieve. The
  cylinder at 42k cells and the cavity at 4k both run on a laptop; adding AMR
  now optimises nothing and complicates everything.
* v2 (turbulence, heat transfer) adds transported scalars. Every one of them
  needs the same refine/coarsen transfer, so building the transfer once after
  the scalar set is settled is cheaper than rebuilding it per variable.
* v4 (VOF) is the case that genuinely needs it — an interface is a
  measure-zero feature and uniform refinement is hopeless. Arriving at v4
  without AMR would mean either accepting a badly resolved interface or
  building AMR under schedule pressure.

### The alternative worth pricing first

Static local refinement — a mesh refined once, by the mesher, where the
physics is known to be — gets a large part of the benefit for none of the
architectural cost, because gmsh already produces it and the solver already
reads it (the cylinder mesh spans three orders of magnitude in cell volume).
The honest comparison is "AMR versus a better static mesh", and for everything
through v3 the static mesh probably wins. That comparison should be made with
numbers on the first case that is actually too slow, not in the abstract.

### If AMR is adopted

Use AMReX rather than writing it. It is the block-structured AMR framework the
ExaWind stack builds on (ADR-003 already names it as a reference), it is
Kokkos-compatible, and it carries the rebalancing and the hanging-node
machinery. The cost is that block-structured AMR wants a Cartesian base grid,
which conflicts with the unstructured body-fitted meshes this solver reads —
so adopting AMReX probably means an overset or embedded-boundary approach for
geometry, which is its own large decision. Writing cell-based unstructured AMR
instead keeps the geometry path but is a multi-month project on its own.

That fork — block-structured with embedded boundaries, or unstructured
cell-based — is the real decision, and it should be made deliberately when
there is a case that forces it, not as a side effect of wanting finer cells in
a wake.

## ADR-020 — The open-domain path needed a gate of its own
**Decided.** Inlet/outlet boundaries get their own verification gate, separate
from the MMS and benchmark gates, and it demands machine precision.

Every gate up to this point ran a closed box: the cavity, both MMS families,
Ethier-Steinman. A closed box cannot see anything wrong with an outlet, and
worse, it cannot see anything wrong with the *structure* of the pressure
equation at a boundary, because when every boundary flux is prescribed the
term subtracted from the right-hand side and the term added back by the flux
correction are literally the same array. Open the domain and they become two
different arrays, and the whole class of defects below becomes reachable.

The first case run on that path — the cylinder wake — reported a continuity
residual of 5.2e-01 from its first step to its last, never decaying, while
producing a drag coefficient of 1.51 that sat close enough to the literature
value to look like a working solver for several hundred steps before it
diverged to NaN. That is the failure mode these gates exist to prevent: a
plausible answer from a broken scheme.

Four defects, all invisible to every existing gate:

1. **The pressure right-hand side used the wrong boundary flux.** It
   subtracted `Fb`, last step's solved outlet flux, while the correction step
   updated `FbStar`, the predicted one. Working through a boundary cell's
   balance, the leftover divergence comes out to exactly `Σ FbStar` — the
   entire outlet mass flow, constant from the first step. That is the 5.2e-01.

2. **The prescribed outlet pressure never reached the source.** The matrix
   carried the boundary diagonal `apb*Dfb` with no matching `apb*Dfb*p_out` on
   the right, so the solver silently imposed `p_out = 0` whatever the caller
   asked. Every case so far prescribed zero, so it gave the right answer for
   the wrong reason.

3. **The outlet face pressure was extrapolated, not prescribed.** The pressure
   equation drove the flux towards `p_out` through its boundary term while the
   gradient operator and the force integral read a value extrapolated from the
   interior, which need not equal it. Two different outlet conditions applied
   at once.

4. **The solved boundary flux was not part of the initial state.** On a
   FixedValue face the flux is a state variable — the caller does not supply
   it — and `setState` left it at zero. The first momentum assembly therefore
   saw an outlet cell whose fluxes did not sum to zero and picked up a
   spurious source worth the whole outlet mass flow. This one is subtle: it
   perturbs the field, the pressure solve redistributes the perturbation
   *conservatively*, and the continuity residual stays clean while the
   velocity is wrong.

Two more were found by reading the code the gate pointed at, and are wrong for
the same reason — a boundary value being used where the boundary condition
says something else:

5. `assembleMomentum` fed the caller's `uB` to the velocity gradient on every
   boundary face, including zero-gradient ones where `uB` is meaningless and
   is conventionally left at zero. That invents a velocity gradient of order
   `u/h` along every slip plane and outlet. It is identically invisible on a
   Cartesian mesh, because the corrections that gradient feeds — non-orthogonal
   and skewness — are zero there, which is why the cavity never saw it.

6. `boundaryForce` hard-coded the boundary velocity to zero, "no-slip wall".
   True for the cavity, where every boundary is one; false for any case with
   an inlet or an outlet, and it means the reported force is the force in a
   different velocity field from the one being solved.

### What the gate checks

Uniform flow through a box is an exact *discrete* fixed point, not merely an
exact solution of the differential equations: convection of a constant field
carries a factor of the cell's net flux, which is zero; diffusion is zero on
every face including the Dirichlet ones; the least-squares gradient of a
constant is identically zero; and the Rhie-Chow flux collapses to `u.S`. So the
gate starts *at* the uniform field and demands that nothing moves, to round-off,
on a distorted mesh as well as a Cartesian one.

Starting from rest instead would have measured a transient rather than the
scheme — the first draft of this gate did exactly that, reported a 5e-04
"failure" that was mostly an unconverged start-up, and had to be rewritten. A
gate that cannot say what the right answer is to machine precision is a gate
that will be argued with later.

Four checks, each catching something the others cannot:

| | check | catches |
| --- | --- | --- |
| A | uniform flow is a fixed point, `p_out = 0` | (1), (4), (5) |
| B | same with `p_out = 7`: same velocity, `p = 7` | (2) |
| C | profiled inflow, transient: zero divergence in every cell, and inflow equals outflow, every step | (1), conservation generally |
| D | outlet face pressure equals the prescribed value under a real pressure gradient | (3) |
| E | raising `p_out` by 7 shifts pressure by exactly 7 and leaves velocity untouched | (2), (3) |

C needs no exact solution: conservation is a property of the scheme, not of
the flow, so it holds from the first step. D and E are the reason B is not
enough — with a constant pressure field an extrapolation lands on the right
answer by luck.

All pass at 1e-14 to 1e-16 on both mesh families. On the cylinder the
continuity residual went from 5.2e-01 to 5e-13 and the run stopped diverging.

## ADR-021 — MPI for the PISO solver, and the gate that makes it checkable
**Decided.** Every cell field the solver reads at a ghost index is exchanged
explicitly, the linear solvers guarantee a valid halo on return, and a
rank-count-independence gate covers the full Navier-Stokes path, not just
diffusion.

v0 made the diffusion operator parallel and gated it. v1 added six more fields
that get read at ghost cells — the velocity, the momentum diagonal `aP`, `H/aP`,
the pressure, and the two gradient sets built from them — and shipped without
extending either. An order study run on four ranks would have reported a clean
second order around the wrong solution.

What was missing:

* **`aP` was a partial sum at ghosts.** Each rank accumulates only the faces it
  stores, so a ghost's diagonal is incomplete. `Df` interpolates `aP` to the
  face and Rhie-Chow reads it from both sides, so the partial value biases
  every rank-boundary flux.
* **`H/aP` likewise**, for the same reason and with the same consequence.
* **`u` and `p`** before any assembly that reads them at `nei(f)`.
* **The native Krylov solvers left the halo one update stale.** They exchange
  inside the matrix-vector product, and the final `x += alpha*p` happens after
  the last product. Callers read `x` at ghosts immediately afterwards — a
  gradient, a face flux — so `LinearSolver::solve` now carries an explicit
  contract that `x` has a valid halo on return, and the two native solvers
  exchange once more before returning. PETSc already did.

The velocity correction now runs over owned cells and exchanges, rather than
computing ghost values from ghost inputs. Both give the same answer when every
input is current, and only one of them keeps saying so when an input stops
being current.

### The gate

`tests/mms/mms_parallel_ns.cpp`: Ethier-Steinman on a *distorted* mesh — the
non-orthogonal and skewness corrections are the terms that read gradients at
ghosts, and on a Cartesian mesh they are identically zero, so a Cartesian gate
would prove nothing. The comparison is against the serial run to ten digits,
not against the exact solution, because the exact solution has discretisation
error thousands of times larger than the defect being hunted.

The sweep count is **fixed**, not converged, and the outer tolerance is zero.
With a tolerance the two runs take different numbers of sweeps whenever the
residual lands either side of it, and a difference in the answer could then be
blamed on that. Fixed, both runs perform literally the same sequence of
operations.

Measured: serial L2 = 1.36666184891033e-02; 2, 3 and 4 ranks agree to
0, 2.7e-15 and 3.8e-16 relative.

**The gate was verified to fail.** It was written after the fix rather than
before it, so it had to earn its place: commenting out the single `aP`
exchange gives L2 = 1.36048e-02, a 0.45% shift. That is the exact failure
profile the gate exists for — far too small for any accuracy study to notice,
and immediately visible as a disagreement between rank counts.

### Not done

The v0 partitioner still reads the whole mesh on every rank (ADR-006). That is
the next thing to change, and it is a memory limit rather than a correctness
one. Communication cost has not been tuned either: on 512 cells the exchanges
and allreduces dominate, which is expected at that size and says nothing about
scaling. Neither is measured yet, and neither should be claimed.

## ADR-022 — Cylinder wake at Re = 100, and the time-step limit it exposed
**Passed.** St = 0.1688, mean Cd = 1.4177, lift amplitude = 0.3636 over 13
shedding cycles, against Williamson's correlation St = 0.1643 — 2.7% high.

This is the first unsteady benchmark and the first on a body-fitted mesh from
an external generator, and it tests three things nothing before it could: time
accuracy against an external number (the cavity is steady, so it says nothing
about BDF2 beyond reaching the right fixed point), the open-domain path under
a real flow, and a mesh with 27 degrees of non-orthogonality and cells
spanning three orders of magnitude in volume.

| quantity | measured | accepted band | literature |
| --- | --- | --- | --- |
| Strouhal number | 0.1688 | 0.150 – 0.178 | 0.164 (Williamson) |
| mean drag | 1.4177 | 1.25 – 1.45 | 1.32 – 1.36 |
| lift amplitude | 0.3636 | 0.25 – 0.42 | 0.30 – 0.35 |

Drag and lift amplitude both sit at the top of their bands. That is what a
coarse near-wall mesh does — 6,763 cells with the first cell at D/17, where
the Re = 100 boundary layer is about D/10 thick — and it should improve with
refinement rather than be argued away. The bands are wide on purpose: published
values for this case move by several percent with domain size and blockage.

### The time-step limit

**This section was wrong. See ADR-024.** It is left standing rather than
edited away, because the way it was wrong is the point.

What it said: the finer mesh diverged because halving the near-wall cell took
the convective Courant number from 1.7 to 3.3, past the limit imposed by the
explicit deferred correction in the convection term.

Both numbers were arithmetic on cell size and free-stream speed, written up in
the voice of a measurement. When the solver was made to report the Courant
number it actually runs at, the coarse mesh turned out to be stable at 14.9
and the fine mesh unstable at 6 — so the quantity named as the cause does not
even order the two cases correctly. The mechanism was disproved separately:
with the deferred correction switched off entirely the fine mesh diverges just
the same, at the same rate.

The one part worth keeping is the instrument. The solver reports
`max_cells 0.5 dt sum|F_f| / V` every step, which is what made the error
visible. A number that is only wrong sometimes needs to be visible always.

### Cost

7,895 s for 4,000 steps at 6,763 cells on two cores — 2.0 s/step, of which the
pressure stage is 95%: 384,214 pressure solves against 36,000 momentum solves,
because each of the 3 outer iterations runs 2 correctors and each corrector
iterates the non-orthogonal loop 16 times. The boundary-pressure extrapolation
is the next item at 1,231 s, since it runs three least-squares gradient passes
every time the pressure gradient is taken, which is once per sweep.

Neither is addressed here. Both are recorded so the next person to ask "why is
this slow" starts from a measurement.

## ADR-023 — The partitioner builds only its own subdomain
**Decided.** `DistributedMesh` takes the raw mesh description — points and
8-node connectivity — and each rank builds the face table and geometry for its
own cells plus one ring of ghosts. It no longer takes a fully built `Mesh`.

ADR-006 accepted "every rank reads the whole mesh and keeps a slice" as the v0
trade: it made the parallel-consistency gate possible without a parallel
reader. The cost is that the largest mesh the solver can run is the largest
mesh that fits on one rank, whatever the rank count — adding nodes bought
speed and no capacity at all.

The built mesh is an order of magnitude larger than the description that
produces it. Faces outnumber cells three to one and each carries an area, a
centroid and four vertex ids; the ordered map of sorted vertex quads that
discovers them is larger still and is the transient peak. Separating the two
is therefore most of the win, and it needs no new file format:

| | serial | 2 ranks | 4 ranks |
| --- | --- | --- | --- |
| faces built, worst rank | 196,800 | 105,208 | 56,773 |
| fraction of serial | 1.000 | 0.535 | 0.288 |
| peak RSS | 61 MB | 42 MB | 31 MB |

(64,000 cells, RCB.)

### Three things this needed

**Ghosts are found by shared vertex, not shared face.** Face adjacency would
need the global face table, which is the thing being avoided. Two cells
sharing a face necessarily share vertices, so a vertex sweep gives a superset:
one bitmap over the points and two passes over the connectivity. The extra
cells — edge and corner neighbours — are dropped once the subdomain's own
faces reveal which ghosts are actually touched.

**Geometry is computed over the whole subset, before filtering.** A ghost's
volume and centroid need all six of its faces. The ring provides them: a face
of a ghost whose other side lies outside the subdomain is seen once by the
vertex hash and comes back as a subdomain boundary face, so the accumulation
still closes every cell. Filtering first would give ghosts volumes computed
from part of their surface — and nothing downstream would complain, because a
plausible volume produces a plausible answer.

**The halo schedule is derived from the local face list on both sides.**
Rank A's receive list from B is the ghosts A uses that B owns; B's send list
to A is the owned cells sitting across a face from one of A's ghosts. Face
adjacency is symmetric, so those are the same set seen from either end, and
sorting both by global cell id lines them up without either rank being told
the other's ordering.

### The gate

`tests/mms/mms_parallel_mem.cpp` counts what each rank *constructs* — cells
and faces, counted before the faces belonging to another rank are discarded,
because they were still built. Deterministic and allocator-independent, unlike
RSS, which the gate reports but does not judge.

The bar is `(1/P + 0.35)` of the serial build on the worst rank; the slack is
the ghost ring, which is real work that does not shrink as fast as the
interior. Measured 0.535 and 0.288 against bars of 0.850 and 0.600.

**Verified to fail.** Making the ghost search return every other cell — which
is precisely the old behaviour — scores 1.000 against a bar of 0.600. The gate
also refuses a rank that builds fewer cells than it owns, and checks the owned
cells still sum to the global mesh, so "build nothing" is not a way past it.

### Still replicated

The raw description. Distributing the read as well means each rank touching
only its own byte range of the file, which needs a format that says where the
cells are — the current one does not. Until then per-rank memory is
`O(raw) + O(built/P)`, and since raw is roughly a tenth of built, that ceiling
is an order of magnitude further out than it was. It is a ceiling all the
same, and this ADR does not claim otherwise.

## ADR-024 — The fine cylinder mesh is unstable, and five explanations are not the reason
**Open.** Recorded because the investigation ruled things out, not because it
finished.

The 21,811-cell cylinder mesh diverges where the 6,763-cell one runs 4,000
steps to a passing benchmark. ADR-022 attributed this to a convective Courant
limit and gave numbers for it. Those numbers were estimates presented as
measurements, and they are wrong; this ADR exists so the next person does not
start from them.

### What the failure actually looks like

Drag climbs smoothly and geometrically — 1.53, 1.56, 1.66, 1.77, 1.91, 2.37 —
while the maximum velocity in the domain grows at the same rate, and stays in
**one cell**, at (−0.16, 0.57), r = 0.59. That is the cylinder surface about
106 degrees round from the stagnation point, where the flow accelerates most.
The cell does not move as the mode grows: this is a stationary local mode, not
something convecting out of the wake.

### Ruled out, each by its own run

| Candidate | Test | Result |
| --- | --- | --- |
| Convective Courant limit | dt ladder on the coarse mesh | stable to **Co 14.9**; the fine mesh fails at 6 |
| Deferred correction (the ADR-022 mechanism) | first-order upwind, correction entirely off | grows identically |
| Explicit terms needing sub-iteration | outer iterations 3 → 10 | diverges **sooner** (t≈5 → t≈1) |
| Momentum non-orthogonal diffusion correction | switched off | grows identically |
| Rhie-Chow old-flux (Choi) term | naive formulation instead | grows identically |
| Mesh pathology | volumes, angles, skewness, LSQ conditioning | nothing: no non-positive volume, no face over 30°, worst gradient stencil 0.83 of isotropic |

The growth rate is *the same* in the upwind, no-non-orth and naive-Rhie-Chow
runs. A mode indifferent to all three is not caused by any of them.

That the outer loop diverges faster when iterated harder is the most
informative single result: it means the iteration is converging, and what it
converges to is unstable. So this is not a lagged term that needs more
sweeps — it is a property of the converged discrete system at this dt on this
mesh.

### What did change something

Dropping the wall pressure extrapolation to plain zero-gradient holds the drag
at 1.46 — a plausible value, near the coarse mesh's 1.41 — for as long as it
was run, while the local velocity spike still grows. So the extrapolation is
carrying the local mode into the global answer, and is not the source of it.
Zero-gradient wall pressure is not a fix: ADR-012 rejected it for dropping the
near-wall cells to first order.

It is also time-step dependent: at dt = 0.0125 the fine mesh is quiet, Courant
1.5, drag settling smoothly through the window where dt = 0.05 has already
turned.

### Where to look next

A stationary mode at one wall cell, indifferent to the convection scheme and
to every explicit correction, surviving outer-loop convergence, and sensitive
to dt. The remaining structural suspects are the wall treatment itself and the
pressure–velocity coupling at a Dirichlet-velocity, fixed-flux-pressure
boundary. The instrument to build first is a local one: the momentum and
continuity budget for that single cell and its neighbours, term by term, over
the steps in which the mode doubles. Every diagnostic so far has been a norm
over the whole field, and a norm is how five wrong answers survived this long.

The switches used here are kept — `deferredCorrection`, `diffusionNonOrth`,
`consistentRhieChow`, `pressureExtrapolation` on `PisoControls` — because each
one converted an argument into a run.
