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
refinement rather than be argued away.

**[Corrected by ADR-027]** It did not improve with refinement. Halving the
near-wall cell moved the drag by 1.7% and the Strouhal number not at all, so
near-wall resolution is not what holds them high. The bands are wide on purpose: published
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
**Closed by ADR-026**: a pressure-velocity checkerboard caused by the
Rhie-Chow flux formulation. The record below is kept as it was written,
because what it ruled out is part of the answer.

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

## ADR-025 — The pressure stage was 95% of the run, and most of it bought nothing
**Decided.** Three changes, each measured, together 2.06x on the cylinder
benchmark: 2.23 s/step to 1.08 s/step with the drag identical to four decimals.

The profile said the pressure stage was 95% of wall time — 384,214 pressure
solves against 36,000 momentum ones. What it did not say, and what mattered,
is that the solves were cheap individually (9.3 iterations each) and simply
too numerous: three outer iterations times two correctors times sixteen
non-orthogonality sweeps is ninety-six pressure solves per step.

### 1. The non-orthogonal correction restarted from zero every time

`solvePressure` allocated a fresh, zeroed correction field on each call and
re-converged it. It is the fixed point of a deferred-correction loop whose
answer moves only a little from one solve to the next, so the previous
answer is the best possible starting point and it was being thrown away a
hundred times a step. The same applied to the pressure itself, explicitly
zeroed before each solve.

Keeping both: 2.23 to 1.70 s/step, sweeps 16 to 12. This one is free in the
strict sense — the loop still converges to the same fixed point to the same
tolerance, only the path changes.

### 2. The loop tolerance was six orders tighter than the answer

1e-12, relative to the largest face flux. Measured on the Ethier-Steinman
order study, where loosening it would show first:

| tolerance | L2 error (distorted, n=16) | order | sweeps | pressure time |
| --- | --- | --- | --- | --- |
| 1e-12 | 2.171843814672e-03 | 1.695 | 24 | 14.7 s |
| 1e-8 | 2.171844960169e-03 | 1.695 | 11 | 7.8 s |
| 1e-6 | 2.171902983645e-03 | 1.695 | 4 | 2.9 s |

At 1e-8 the L2 error moves by 3e-7 relative — four orders below the
discretisation error the study exists to measure — and the pressure time
halves. Default is now 1e-8. 1e-6 is another 2.7x and still 300 times below
the error, and is left to cases that want it rather than taken as the
default, because the margin at 1e-8 will survive several more refinements and
the margin at 1e-6 will not.

### 3. Each sweep was solved to fourteen digits

An intermediate sweep of a fixed-point iteration solved to 1e-14 is fourteen
digits of an answer the next sweep changes. The constraint is a floor, not a
ceiling: looser than the loop's own threshold and the loop chases solver
noise, which is exactly the mistake behind ADR-016. Two orders tighter than
`nonOrthTol` satisfies both, so the default is 1e-10.

Measured: 12,511 linear iterations against 7,980, same drag to four decimals.
A case that loosens `nonOrthTol` must loosen this with it.

### Measured and rejected

**Rebuilding the AMG hierarchy less often.** The obvious suspect, and wrong.
Reusing the preconditioner across six matrix changes saves 1 s; reusing it
indefinitely *costs* 8 s, because a stale hierarchy needs 20,174 iterations
where a fresh one needs 12,511. Setup is not where the time goes.

**A fixed small sweep count instead of a tolerance.** Tempting: with the warm
start in place, one fixed sweep gives 0.35 s/step, six times faster than the
original, and the cylinder drag is unchanged to four decimals. Rejected as a
default because the benchmark is not where it would break — an order study on
a distorted mesh with two time steps has no history for the warm start to
carry, so the correction would never converge and the second-order term would
be the thing left out. A tolerance adapts to both; a count does not.

### Measured on the full benchmark

The cylinder wake gate, 4,000 steps, before and after:

| | before | after |
| --- | --- | --- |
| wall time | 7,895 s | **3,392 s** |
| per step | 2.0 s | 0.85 s |
| pressure solves | 384,214 | 170,141 |
| Strouhal | 0.1688 | 0.1688 |
| mean drag | 1.4177 | 1.4177 |
| lift amplitude | 0.3636 | 0.3636 |

2.33x, and the three measured quantities are unchanged to four decimals. That
is the claim worth making: not that it got faster, but that it got faster and
the answer did not move.

### Left on the table

The boundary-pressure extrapolation, now 19% of the run, still restarts cold
and runs three least-squares gradient passes on every pressure gradient. The
same warm start should take it to one. It was not done here because verifying
the three changes above mattered more than a fourth unverified one.

## ADR-026 — The instability was a checkerboard, made by the Rhie-Chow flux itself
**Fixed.** The predicted face flux is now `F* = (H/aP)_f · S` (plus the Choi
term), the standard form OpenFOAM's PISO uses. Both the Python reference and
the C++ solver had been adding `D (grad(p)_f · S − snGrad p_old)` to it. The
old form is kept behind `RhieChowForm::Interpolated` solely so the defect can
be shown on demand.

### What the probe saw

ADR-024 ended by asking for one instrument: every term of the budget for one
cell, every step. Built as `PisoSolver::probe()`, pointed at the cell the
growth lived in, it answered in a single run.

Cell 1255 — not a wall cell; three cells off the cylinder — and its four
neighbours, at t = 5.05:

| | pressure | velocity |
| --- | --- | --- |
| cell 1255 | +27.1 | (+7.3, +2.8) |
| neighbour 12532 | −39.4 | (**−10.1**, −6.0) |
| neighbour 1254 | −11.5 | |
| neighbour 1256 | +7.2 | |
| neighbour 21792 | −4.4 | |

Adjacent cells running to pressures of opposite sign, one of them flowing
backwards — it reversed at t ≈ 2.8 — against a free-stream dynamic pressure of
0.5. Face-flux continuity held to 1e-9 throughout. That is a pressure-velocity
checkerboard in its textbook form: a mode living in the cell-centred fields,
where the face-flux continuity equation cannot see it. It is precisely what
Rhie-Chow interpolation exists to prevent.

The decisive number was on the face between those two cells: the term
`D · snGrad(p)` evaluated on the pressure *before* the last solve was +0.126,
and on the pressure *after* it, −0.106. Same operator, consecutive solves,
opposite signs. It was already so at t = 2.05, long before anything grew.

### Why the flux made it

With `−D snGrad(p_old)` inside F* and the full pressure re-solved on every
corrector, the checkerboard part of the pressure equation reduces to

    p_new ≈ −p_old + forcing

because for a checkerboard the interpolated cell gradient `grad(p)_f` is nearly
blind and only the compact gradient survives. An eigenvalue near −1: every
decoupled mode flips sign on every pressure solve. On its own the flip damps
slightly (the measured ratio was 0.84), but the velocity corrector feeds the
checkerboard back through H/aP, and once that coupling tips the magnitude past
one the mode grows.

The form is a hybrid of two correct ones. Ferziger–Perić's incremental SIMPLE
carries `D (grad(p)_f − snGrad p_old)` but interpolates the *predicted*
velocity and solves for a pressure *correction*. OpenFOAM's PISO solves for
the full pressure from a *pressure-free* H/aP and has no such term. Taking the
term from the first and the unknowns from the second is what produced the −1.

### It explains everything ADR-024 ruled out

* **More outer iterations diverged sooner** — the result ADR-024 called the most
  informative. Each outer iteration is two more pressure solves, so two more
  flips per step. The iteration was converging, and what it converged to was
  a growing oscillation between solves.
* **Indifferent to the convection scheme, the diffusion correction and the
  Choi term** — none of them is in the pressure-correction loop.
* **Time-step dependent** — `D = V/aP` and the velocity feedback both scale with
  dt, which moves the loop gain across one.
* **Worse on the finer mesh** — the loop gain depends on the mesh; the coarse
  one happened to sit under one. Nothing about the coarse result was right for
  a better reason.
* **Six solves per step is even**, so the sign came back each step and the
  per-step pressure looked monotonic. The flip was invisible to anything that
  sampled once a step.

### Evidence that the fix is the fix

Same mesh, same dt = 0.05, only the flux form changed:

| | interpolated | standard |
| --- | --- | --- |
| drag at t = 5.3 | 2.93, climbing | 1.35, settling |
| fastest cell | 12.9 and growing | 1.34 |
| p(cell) − mean(neighbours) | 39.1 | 0.009 |
| non-orthogonality sweeps | 19–20 | 3–4 |
| cost | 5.3 s/step | 2.9 s/step |

And the time step the refined mesh tolerates, 150 steps each:

| dt | Courant | interpolated | standard |
| --- | --- | --- | --- |
| 0.05 | ~6 | diverges by step 120 | stable |
| 0.1 | ~11 | NaN in 5 steps | stable (pre-shedding drag 1.30–1.36) |
| 0.2 | ~21–31 | — | stable, full shedding |

### Side effects, all measured

**The Python–C++ cross-check went from 2.0e-5 to 6.6e-11.** That 2e-5 gap had
been put down to the two implementations' linear algebra. It was the
near-neutral mode amplifying round-off differences between them; with the mode
gone they agree to round-off.

**The order study came out better, not worse.** At the coarsest meshes the
standard form's error is larger on the distorted family, and the first
version of this ADR reported that alone ("about 9% larger"). The full C++
study shows the gap closing with refinement and the order rising:

| n | interpolated L2 | standard L2 | gap |
| --- | --- | --- | --- |
| 8 | 7.031e-3 | 7.835e-3 | +11% |
| 16 | 2.172e-3 | 2.310e-3 | +6% |
| 32 | 5.838e-4 | 5.922e-4 | +1.4% |
| order, 16→32 | 1.895 | **1.964** | |

Orthogonal: 1.986 → 1.991. The old form's smaller coarse-mesh error was a
lower-order term working in its favour on a smooth manufactured solution, not
an asymptotic advantage. The non-orthogonality loop on the distorted study now
converges in **one** sweep; the flipping mode had been making it fight.

**Every benchmark predating this ADR used the defective form**, so the whole
suite was re-run after the change — all v0 and v1 gates pass:

| | before | after |
| --- | --- | --- |
| cylinder Strouhal | 0.1688 | 0.1698 |
| cylinder mean drag | 1.4177 | 1.4233 |
| cylinder lift amplitude | 0.3636 | 0.3671 |
| cylinder run, 4,000 steps | 3,392 s | 2,897 s |
| Ghia cavity rms (u, v) | 0.0016, 0.0044 | 0.0016, 0.0044 |

The coarse cylinder moves by well under one percent and stays in band. Its
drag is still at the top of the band; this ADR first attributed that to its
near-wall resolution, citing 1.30–1.36 from the refined mesh at dt = 0.1. That
figure was the transient before shedding developed, and the saturated
refined-mesh drag is 1.399 — see ADR-027, which also shows resolution is not
the cause. The
cavity does not move at all: on a uniform orthogonal mesh the interpolated and
compact gradients nearly coincide, which is also why no earlier gate could
have told the two forms apart.

The boundary flux at a FixedValue face was already built in the standard form
(`FbStar = H/aP · S`), so the internal faces now agree with it — an
inconsistency nobody had noticed, fixed as a side effect.

### Why it survived this long

Both forms are second order, so no order study could separate them. The
reference implementation shared the defect, so the cross-check agreed with it.
The coarse benchmark mesh kept the eigenvalue under one, so the benchmarks
passed. And the one case that exposed it was explained away five times —
beginning with ADR-022's Courant limit, which was an estimate written up as a
measurement.

What found it was not cleverness but resolution: an instrument that looked at
one cell, term by term, instead of a norm over forty thousand.

### The gate

`tests/benchmark/checkerboard.py` runs the case that failed — the 21,811-cell
mesh at dt = 0.1 for 100 steps — and judges the fastest cell in the domain
against a bound of 3 (potential flow peaks at 2, the viscous flow at about
1.5). Verified both ways: the defective form reaches |u| = 11,443 and a drag
of 4.7 million before stopping at step 27; the corrected form holds 1.34 and
a drag of 1.2996 for all 100 steps.

The cylinder case now stops at the first non-finite force. A diverged run
otherwise grinds every remaining linear solve to its iteration cap, and a gate
that can hang the suite is not a gate.

`tests/benchmark/courant_limit.py` is removed rather than finished. It gated a
Courant limit attributed to the deferred correction; the limit was never
measured and the mechanism is disproved. The thing it was reaching for — the
refined mesh surviving a practical time step — is what this gate checks.

## ADR-027 — The refined cylinder: resolution is not why drag is high
**Measured.** With the checkerboard fixed (ADR-026) the 21,811-cell mesh runs
the full benchmark for the first time. Same dt, same domain, same statistics
window as the 6,763-cell run — only the mesh differs, so any change belongs
to the mesh.

| | 6,763 cells | 21,811 cells | literature |
| --- | --- | --- | --- |
| first cell at the wall | D/17 | D/33 | |
| Strouhal | 0.1698 | 0.1699 | 0.164 (Williamson) |
| mean drag | 1.4233 | 1.3991 | 1.32 – 1.36 |
| lift amplitude | 0.3671 | 0.3491 | 0.30 – 0.35 |
| 4,000 steps | 2,897 s | 7,853 s | |

Lift amplitude comes into the literature band. Drag and Strouhal barely move:
the Strouhal number is unchanged in the fourth digit and the drag falls 1.7%.
A two-grid Richardson estimate, assuming second order and taking the
near-wall ratio of two, puts the grid-independent drag near 1.39 — still
above the band. Two grids cannot confirm the order, so treat that as an
estimate; but no plausible order turns a 1.7% step into the 4–6% needed.

**[Qualified by ADR-029]** Node placement alone scatters the Strouhal number
by 0.0006 and the drag by 0.005 (one standard deviation) at 6,763 cells, so
"unchanged in the fourth digit" means unchanged within that scatter, and the
agreement to the fourth digit was luck. The conclusion stands: the drag drop
with refinement is three times the scatter of a difference, and the Strouhal
excess over Williamson is ten times it.

So the explanation this record gave twice — ADR-022 and the first version of
ADR-026 — that the high drag was a coarse near-wall mesh, is wrong. The second
of those also quoted 1.30–1.36 from the refined mesh at dt = 0.1, which was
the transient before shedding developed; saturated shedding raises the mean
drag, and the value the benchmark actually measures is 1.399.

What the two runs share is the domain: lateral boundaries at ±10 D (5%
blockage), inlet 10 D upstream, outlet 25 D downstream, and free-stream
velocity imposed on the lateral boundaries. Blockage raises both the
Strouhal number and the drag, and both sit 3–4% high on both meshes. That is
a hypothesis, recorded as one: it is tested by one run with the lateral
boundaries at ±20 D, and it should be tested before any further refinement,
because refinement has just been shown not to be the lever.

## ADR-028 — Published as Vibeflow
**Decided.** The project is public, under the name Vibeflow, at
github.com/chs1372/Vibeflow. "nsflow" was a working name, marked in the README
as a placeholder from the first commit. The rename is complete and has no
compatibility aliases, because nothing outside this repository ever used the
old names:

| | before | after |
| --- | --- | --- |
| C++ namespace | `nsflow` | `vibeflow` |
| CMake project, targets | `nsflow`, `nsflow_core`, … | `Vibeflow`, `vibeflow_core`, … |
| CMake option | `NSFLOW_BUILD_TESTS` | `VIBEFLOW_BUILD_TESTS` |
| compile definitions | `NSFLOW_HAVE_MPI` / `_PETSC` / `_CGNS` | `VIBEFLOW_HAVE_…` |
| environment knobs | `NSFLOW_PRESSURE`, `NSFLOW_OUTER`, … | `VIBEFLOW_PRESSURE`, `VIBEFLOW_OUTER`, … |

Commits before the rename keep the old name. They are the record, and this
log refers to them as they were.

*Why the name:* the solver is written by vibe coding — an AI assistant writes
the implementation — and the gate-first rule is what makes that acceptable
for numerical code. The name says how it was made; the gates say whether it
works.

### Found by reading the repository as a stranger would

None of these could have been caught by a gate, because none of them is in
the solver. Each is something a new user would have hit.

* **A build option that could only break the build.** `NSFLOW_WITH_PETSC`
  defaulted to off and did nothing, because `src/linalg` detects PETSc through
  pkg-config regardless; switching it on called `find_package(PETSc
  REQUIRED)`, which fails against the pkg-config install that every PETSc run
  in this log actually used. Removed. MPI, CGNS and PETSc are each detected
  where they are used.
* **A CI workflow that had never run.** It installed Kokkos with
  `-DCMAKE_INSTALL_PREFIX=~/kokkos-install`; neither bash nor CMake expands a
  tilde in the middle of a word, so Kokkos would have gone into a directory
  literally named `~` and the next step would not have found it. Now `$HOME`,
  pinned to `ubuntu-24.04`, and replayed step for step before the first push:
  a fresh clone, Kokkos 5.2.2 cloned from its tag and built with the
  workflow's own flags, fixtures regenerated with no diff, every v0 gate
  passing — 147 s end to end on two cores. Its actions also moved to their
  current major versions, which run on Node 24; the replay covers the run
  steps, not the actions themselves, so the first run on GitHub is still the
  first real test of those.
* **A benchmark mesh nobody could regenerate.** The cylinder meshes are
  generated, not stored, and the settings of the 6,763-cell one were never
  written down. Recovered by search: `CYL_SIZEMIN=0.06 CYL_SIZEMAX=1.0`
  reproduces it byte for byte with gmsh 4.15.2, as the defaults reproduce the
  21,811-cell one. `cases/cylinder/make_meshes.sh` now makes both and records
  their checksums, and the wake gate skips with that instruction rather than
  failing on a missing file.
* **A licence GitHub could not read.** The policy note on GPL tools stood in
  for the licence. `LICENSE` is now the canonical Apache-2.0 text and the note
  is `docs/LICENSING.md`.
* **A dependency file that had never been used.** `spack.yaml` named Kokkos
  4.4 while everything was built against 5.2.2. It now names 5.2.2 and says
  it has not been exercised; the README gives the configuration that has.

Re-run from that fresh clone under the new name, the whole suite passes: the
8 v0 gates inside the 147 s replay, then the 13 v1 gates in 54 minutes on the
meshes the new script generated. The coarse cylinder gives Strouhal 0.1698,
drag 1.4233 and lift amplitude 0.3671, the same in every printed digit as
ADR-026; a rename that changed a number would have been a defect. The opt-in
Re = 1000 cavity had not been run since the Rhie-Chow change of ADR-026 and
was run now for the README: rms 0.0121 and 0.0126 against Ghia, from 0.0123
and 0.0127 in ADR-018.

The authorship of every commit is the publishing account's GitHub no-reply
address. The AI co-author trailers are unchanged.

*Reverses if:* nothing technical. A rename costs one commit only until other
people depend on the names, which from this commit on they may.

## ADR-029 — Is the high drag blockage? Stated before the run, answered after
**Question.** ADR-027 left the Strouhal number 3.5% and the mean drag 5–8%
above the literature on both meshes, ruled out resolution, and named
blockage as the working hypothesis: the lateral boundaries sit at ±10 D, a
blockage ratio B = D/H of 5%, and carry the free-stream velocity. This entry
is written and committed before the test runs, so that the result cannot
choose its own yardstick.

**Test.** One change at a time: the lateral boundaries move out, everything
else stays. Same generator settings as the 6,763-cell mesh (first cell D/17,
the same distance-based size field), dt = 0.05, 200 time units, statistics
over t > 120. The added region lies ten diameters or more from the cylinder,
where the size field is already coarse, so the mesh near the body keeps its
resolution. Two widened domains, run in this order:

| lateral boundaries | B | role |
| --- | --- | --- |
| ±10 D | 5% | existing result: St 0.1698, Cd 1.4233 |
| ±20 D | 2.5% | the test |
| ±40 D | 1.25% | a third point, to check that the trend is linear in B before extrapolating it |

**What blockage alone would predict.** A classical first-order estimate for
a 2D bluff body — wake blockage ε = (D/4H)·Cd plus solid blockage
(π²/12)(D/H)² — gives an effective speed-up of 2.0% at B = 5%. Taken at face
value it predicts St 0.1680 and Cd 1.390 at ±20 D, St 0.1672 and Cd 1.376 at
±40 D, and St 0.1665 and Cd 1.364 unconfined. That estimate assumes distant
slip walls, not a prescribed velocity, so it is a magnitude, not a target.
Note what it already says: even with blockage removed entirely, St would sit
1.5% above Williamson's 0.164.

**Decision rule.** Extrapolate linearly in B through the runs to B = 0.

- *Blockage explains the excess* if the extrapolated values reach St ≤ 0.166
  and Cd ≤ 1.36: within 1% of Williamson, inside the reported drag band.
- *Blockage is ruled out* if the ±20 D run moves both by less than a mesh of
  the same resolution could: |ΔSt| < 0.0003 and |ΔCd| < 0.005. Doubling the
  resolution moved St by 0.0001, so node-placement noise at fixed resolution
  should sit below these.
- *Anything in between* means blockage is part of the cause. The entry then
  records what fraction of each excess it accounts for, and the next suspect
  — the inlet ten diameters upstream, which confines the flow from the front
  — gets the same treatment.
- A shift within twice the noise thresholds is not read either way until a
  control run (a ±10 D mesh with its nodes placed differently) measures the
  noise directly.

### Result

Everything above this heading was committed before the first run.

| mesh | lateral boundaries | cells | St | Cd | lift amplitude |
| --- | --- | --- | --- | --- | --- |
| `debug.hex` | ±10 D | 6,763 | 0.1698 | 1.4233 | 0.3671 |
| control A (lateral at ±9.999 D) | ±10 D | 6,784 | 0.1709 | 1.4157 | 0.3656 |
| control B (inlet at −10.001 D) | ±10 D | 6,781 | 0.1699 | 1.4130 | 0.3599 |
| ±20 D | ±20 D | 7,897 | 0.1692 | 1.4095 | 0.3666 |
| ±40 D | ±40 D | 9,353 | 0.1692 | 1.3989 | 0.3595 |

Every run: dt = 0.05, 4,000 steps, 13 shedding cycles after t = 120, maximum
Courant number 2.7–3.1. Each mesh is `cases/cylinder/make_mesh.py` with
`CYL_SIZEMIN=0.06 CYL_SIZEMAX=1.0` and one of `CYL_YHALF=20`, `CYL_YHALF=40`,
`CYL_YHALF=9.999` or `CYL_XIN=-10.001`; with gmsh 4.15.2 that reproduces them.

**The first finding was the noise.** The ±20 D run moved St by 0.0006, inside
twice the noise threshold assumed above, so under the rule the control runs
came before any reading — and they overturned the assumption. Three ±10 D
meshes that differ only because one boundary moved by a thousandth of a
diameter give St from 0.1698 to 0.1709 and Cd from 1.4130 to 1.4233. At this
resolution node placement alone scatters St with a standard deviation of
0.0006 and Cd with 0.005: twice and once the thresholds written above. The
assumption rested on the refined mesh moving St by 0.0001, and that
agreement was luck.

**Read against that scatter:**

- *Drag.* The ±40 D run sits 0.018 below the ±10 D mean, three standard
  deviations. A straight line in B through all five runs extrapolates to
  Cd = 1.395 ± 0.006 unconfined: blockage is worth 0.022 of drag, about 28%
  of the excess over the middle of the reported band.
- *Strouhal number.* Both wider domains give 0.1692, 1.4 standard deviations
  below the ±10 D mean; the line extrapolates to 0.1687 ± 0.0007. Whatever
  blockage does to St is at most 0.0015 — a quarter of the excess — and it
  cannot be told from zero at this scatter.
- *Lift amplitude.* No trend beyond the scatter.
- The classical estimate predicted shifts two to three times these. Why is
  not settled here.

**Verdict under the rule.** Neither "explains it" — the extrapolated values
miss St ≤ 0.166 and Cd ≤ 1.36 by 0.0027 and 0.035 — nor "ruled out", since
the drag trend is real. Blockage is part of the drag excess and at most a
small part of the Strouhal excess. The Strouhal number now sits at
0.169–0.171 through doubled resolution, a fourfold wider domain and three node
placements: 3–4% above Williamson, cause unknown.

**What this changes.**

- One run on one mesh at this resolution cannot resolve a difference smaller
  than about 0.002 in St or 0.015 in Cd (two standard deviations of a
  difference). ADR-027 is qualified accordingly; its conclusion stands.
- The next test should not change the mesh. The time step is the obvious
  one: St is a frequency, and neither space nor domain has moved it. Halving
  dt on `debug.hex` carries no node-placement scatter at all. The inlet
  distance, the suspect named above, changes the mesh, so it needs repeated
  runs to be read at all.

## ADR-030 — Is the Strouhal excess the time step? Stated before the run, answered after
**Question.** After ADR-027 and ADR-029 the Strouhal number sits at
0.169–0.171 whatever the resolution, the domain width or the node placement:
3–4% above Williamson's 0.164. It is a frequency, and the one discretisation
parameter not yet varied is the time step. Written and committed before the
runs, as ADR-029 was.

**Test.** The wake gate's own case on its own mesh, `debug.hex`, with only
dt changed, and the same end time (200) and statistics window (t > 120):

| dt | steps | role |
| --- | --- | --- |
| 0.1 | 2,000 | third point, for the observed order |
| 0.05 | 4,000 | existing result: St 0.1698, Cd 1.4233, lift 0.3671 |
| 0.025 | 8,000 | the test |

Because the mesh does not change, ADR-029's node-placement scatter does not
apply. Runs on one mesh are deterministic here — the gate reproduced 0.1698,
1.4233 and 0.3671 exactly from two separate builds, one of them a fresh
clone — so any difference is the time step's, down to the fourth printed
digit.

**What BDF2 alone would predict.** Applied to a pure oscillation at this
frequency, BDF2 lowers the frequency by 0.37% at dt = 0.1, 0.095% at 0.05
and 0.024% at 0.025. The wake is a nonlinear oscillator, not a linear one, so
that is a magnitude and a sign, not a target: St should *rise* by about
0.0005 from dt = 0.1 to 0.05, by about 0.0001 from 0.05 to 0.025, and by
0.0002 in all as dt goes to zero. If that holds, the time step works against
the excess and cannot be its cause. A drop of 0.001 or more as dt shrinks
would mean something first-order in dt dominates instead — the PISO
splitting, or a lagged correction — and that is the finding to look for.

**Decision rule.**

- *The time step explains the excess* if St, extrapolated to dt = 0 from the
  three runs, falls at least 0.003 below the dt = 0.05 value — half the
  excess.
- *The time step is ruled out* if St(0.025) differs from St(0.05) by no more
  than 0.0003, or moves up.
- *Anything in between* is part of the cause, and the observed order says
  which part: near two, the time integration; near one, the splitting,
  which the next test would then isolate by converging the outer iteration
  at a fixed dt.

### Result

Everything above this heading was committed before the first run.

| dt | steps | max Courant | St | Cd | lift amplitude | wall time |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 2,000 | 5.68 | 0.1688 | 1.4277 | 0.3675 | 1,660 s |
| 0.05 | 4,000 | 2.84 | 0.1698 | 1.4233 | 0.3671 | 2,395 s |
| 0.025 | 8,000 | 1.42 | 0.1707 | 1.4141 | 0.3711 | 4,043 s |

**Verdict under the rule: ruled out.** St moves *up* as dt shrinks, by 0.0009
from 0.05 to 0.025. The time step works against the excess, as BDF2
predicted in sign, and extrapolating to dt = 0 can only raise St further.

**But the values do not converge the way BDF2 says they should.** BDF2
predicted changes shrinking fourfold per halving, about +0.0005 and then
+0.0001. The measured changes are +0.0010 and +0.0009, an observed order near
0.2. The drag moves the other way from convergence: −0.0044, then −0.0092, the
change doubling as dt halves. The lift amplitude changes by −0.0004, then
+0.0040. Over a range where each shedding period gets 60 to 240 steps, nothing
here behaves like a second-order time error.

**The likeliest reason is already on record.** ADR-010 warned that
Rhie-Chow makes the face flux depend on dt through aP, required the
dt-consistent form that carries the old-time flux, and required a gate that
solves one steady problem at several dt and demands one answer. That gate was
never made one. The Python suite still reports it as "not gated", and on
today's code it says the consistent form is the one that shows the
dependence: the same steady problem at dt = 0.05 and 2.0 differs by 3.9% in
the error norm, against 0.22% for the naive form. The notes beside that
report suspect that as dt → 0 the old-flux term accumulates instead of
settling. A steady solution that
depends on dt can only get that from the face flux, and the drift here grows
as dt shrinks, which is what such an accumulation would do. That is a
hypothesis, not yet a finding.

**What this changes.**

- The Strouhal excess is not the time step.
- At a fixed mesh the wake values drift by about 0.001 in St and 0.005–0.01
  in drag per halving of dt, with no sign of settling over this range. They
  are not yet "the answer on this mesh", and every comparison in ADR-027 and
  ADR-029 was made at one dt, so it holds at that dt only.
- The next test is the one ADR-010 asked for: the same mesh and the same two
  time steps with the old-flux term switched off (`CYL_NAIVE_RC`). If the
  drift goes, the old-flux term is the source, and ADR-010's
  dt-independence gate is written and made to fail before the formulation
  is changed.

## ADR-031 — Is the time-step drift the Rhie-Chow old-flux term? Stated before the run, answered after
**Question.** ADR-030 found the wake values drifting with dt on a fixed mesh
— St up by 0.0010 and 0.0009 over two halvings, drag down by 0.0044 and
0.0092 — where BDF2 alone predicts changes that shrink fourfold. It named
the old-flux (Choi) term of the Rhie-Chow flux as the suspect: it is the only
part of the face flux that carries dt, and on the steady manufactured
problem it is the form that shows a dt dependence (3.9% against 0.22%).
Written and committed before the runs.

**Test.** ADR-030's case with one switch: `CYL_NAIVE_RC`, which drops the
old-flux term and leaves everything else as it was. Same mesh (`debug.hex`),
dt = 0.05 and 0.025, same end time and statistics window. Same mesh, so no
node-placement scatter.

**What a guilty old-flux term predicts.** With it off, the St change from
dt = 0.05 to 0.025 should shrink towards BDF2's own +0.0001, and the drag
change to a few thousandths. The naive form has its own known weakness,
which the old-flux term exists to cure: its pressure damping scales with
V/aP, which shrinks with dt, so at small dt pressure and velocity can
decouple. The runs report the fastest cell, the instrument that caught the
checkerboard of ADR-026, and a decoupled run is read as such, not as data.

**Decision rule**, on the changes from dt = 0.05 to 0.025:

- *The old-flux term is the source of the drift* if, with it off,
  |ΔSt| ≤ 0.0003 and |ΔCd| ≤ 0.003 (with it on: +0.0009 and −0.0092).
- *It is not the source* if, with it off, either change is at least half
  of what it was with it on: |ΔSt| ≥ 0.00045 or |ΔCd| ≥ 0.0046.
- *Anything in between* makes it part of the source.
- *Inconclusive* if a naive run decouples — the fastest cell above 3 or a
  non-finite force. That would say the damping the old-flux term provides
  is needed, and the fix is a form that keeps the damping without the
  drift, not dropping the term.

### Result

Everything above this heading was committed before the first run.

| old-flux term | dt | St | Cd | lift amplitude | fastest cell, t > 120 |
| --- | --- | --- | --- | --- | --- |
| on (ADR-030) | 0.05 | 0.1698 | 1.4233 | 0.3671 | 1.36, beside the cylinder |
| on (ADR-030) | 0.025 | 0.1707 | 1.4141 | 0.3711 | **2.13, 17.5 D downstream** |
| off | 0.05 | 0.1694 | 1.4364 | 0.3726 | 1.36, beside the cylinder |
| off | 0.025 | 0.1697 | 1.4379 | 0.3736 | 1.36, beside the cylinder |

Changes from dt = 0.05 to 0.025:

| old-flux term | ΔSt | ΔCd | Δlift |
| --- | --- | --- | --- |
| on | +0.0009 | −0.0092 | +0.0040 |
| off | +0.0003 | +0.0015 | +0.0010 |

**Verdict under the rule: the old-flux term is the source of the drift.** With
it off, both changes are within the thresholds — St exactly at 0.0003, drag
at half of 0.003 — and the drag change has reversed sign and shrunk
sixfold. Neither naive run decoupled.

**The fastest cell says why, and it was not in the rule.** At dt = 0.05 every
run's fastest cell sits beside the cylinder at 1.36, where the physical flow
peaks. With the old-flux term on and dt = 0.025 it moves 17–19 diameters
downstream and reaches 2.13 in a wake whose velocity should be close to
one: six of the ten report lines after t = 120 exceed 1.6. The far
wake has the largest cells and the weakest convective coupling, which is
where D·aP_t — the fraction of the old Rhie-Chow residual carried into each
new step — comes closest to one. The residual accumulates there faster than
it decays, a local pressure-velocity mode grows until the flow limits it,
and the forces feel it. It stayed under the decoupling gate's bound of 3,
which is why only this instrument, not the gate, showed it.

**The term also moves the answer itself.** At dt = 0.05, switching it off
raises the drag by 0.013 and lowers St by 0.0004: as much as a fourfold wider
domain. Neither form is the reference. The naive one is what ADR-010 warned
about — its damping shrinks with dt — and it only looks well behaved here
because dt = 0.025 is not yet small enough to show that.

**What this changes.**

- BDF2 is not the problem; the drift belongs to the face-flux formulation.
- The consistent form as written is defective at small dt. A form that
  keeps the damping without accumulating is needed, and ADR-010's gate comes
  first: one steady problem at several dt, one answer, made to fail on the
  current form before any change. Candidates: the old-flux term built from
  BDF2's own two time levels (the current one uses aP_t = 1.5/dt on the
  latest level only, which agrees with BDF2 only at steady state), and a
  limited coupling coefficient of the kind OpenFOAM's `ddtCorr` applies.
  **[Corrected by ADR-037]** The first candidate had already been measured
  and rejected: `prototype/piso.py` records that its recursion has a root at
  exactly one, so the initial residual never decays and the converged state
  depends on the path (2.6% apart at dt 0.05 and 2.0). ADR-037 finds the
  cause elsewhere.
- The wake gate should judge the fastest cell over the statistics window,
  with a bound near the physical peak, rather than 3 over any 100 steps.
- Results at dt = 0.05 are unaffected in the sense that matters for the
  comparisons already made and the ones queued: at that dt the fastest cell
  is physical, and every comparison is like for like. The Strouhal excess is
  not the old-flux term either: St is 0.169–0.170 with it on or off.

## ADR-032 — Does the inlet ten diameters upstream raise the Strouhal number? Stated before the run, answered after
**Question.** ADR-029 named the inlet as the next suspect: the velocity is
prescribed ten diameters upstream, which confines the flow from the front
the way the lateral boundaries confine it from the sides. Written and
committed before the runs.

**Test.** The inlet moves from −10 D to −20 D; everything else is as in
ADR-029's ±10 D runs — lateral boundaries at ±10 D, outlet at +25 D, first
cell D/17, dt = 0.05, the consistent Rhie-Chow form (at this dt the fastest
cell is physical, ADR-031). The mesh changes, so ADR-029's node-placement
scatter applies, and the test is built to be read through it: three meshes
with the inlet at −20 D whose nodes differ (inlet at −20, at −20.001, and at
−20 with the lateral boundaries at ±9.999 D), compared as a mean with the
mean of ADR-029's three ±10 D meshes (St 0.1702, Cd 1.4173). The standard
deviation of a difference of two three-run means is 0.0005 in St and 0.0043
in Cd.

**What the inlet alone would predict.** A body with drag sends a source-like
flow upstream, of strength U·Cd·D/2, which ten diameters ahead slows an
unbounded free stream by about 1.1%. Prescribing U∞ there removes that
slowdown, as if the body met a slightly faster stream. If the whole 1.1%
reached the body, moving the inlet to 20 D would halve it: St down by about
0.0009 and Cd by about 0.014. The lateral boundaries already carry part of
that flux, so the real effect should be smaller; take those as upper bounds.

**Decision rule**, on the mean at −20 D minus the mean at −10 D:

- *The inlet is a significant part of the excess* if St falls by at least
  0.0010, two standard deviations of the difference. Its whole effect is
  then about twice the measured shift, since an effect that decays like
  1/L leaves as much again at 20 D.
- *It is ruled out* if both St and Cd move by less than one standard
  deviation (0.0005 and 0.0043).
- *Anything in between* bounds it: at most twice the measured shift.

### Result

Everything above this heading was committed before the first run. One
sample was replaced during the test; the replacement is described below and
was chosen before it ran.

| mesh | inlet | lateral boundaries | cells | St | Cd | lift amplitude |
| --- | --- | --- | --- | --- | --- | --- |
| A | −20 D | ±10 D | 6,956 | 0.1693 | 1.4201 | 0.3708 |
| B — not independent of A, dropped | −20.001 D | ±10 D | 6,952 | 0.1693 | 1.4199 | 0.3708 |
| C | −20 D | ±9.999 D | 6,972 | 0.1688 | 1.4075 | 0.3610 |
| D — B's replacement | −20 D | ±10.001 D | 6,984 | 0.1686 | 1.4086 | 0.3591 |

Every run: dt = 0.05, 4,000 steps, 13 shedding cycles after t = 120, maximum
Courant number 2.8–3.2, fastest cell after t = 120 at 1.35–1.36 beside the
cylinder, which is physical. Each mesh is `cases/cylinder/make_mesh.py` with
`CYL_SIZEMIN=0.06 CYL_SIZEMAX=1.0 CYL_XIN=-20` plus nothing (A),
`CYL_YHALF=9.999` (C) or `CYL_YHALF=10.001` (D); B had `CYL_XIN=-20.001`.

**The deviation.** B was meant to be a third node placement and is not one.
It reproduced A to the fourth digit, and comparing the meshes showed why:
its nodes within 2 D of the cylinder sit a median 0.0026 D from A's, where
every other pair — here and in ADR-029 — differs by 0.025–0.028 D. Moving
the inlet by a thousandth of a diameter re-placed the far field and left the
body's neighbourhood almost as it was. B was dropped as a sample. By then A
and B had finished and C was running. Four candidate meshes were generated,
and the one whose smaller node offset from A and C was largest (0.026 D from
both) was run as D, chosen on geometry alone. The statistics below are over
A, C and D. Had B been counted instead, the St shift would be −0.0011: the
same verdict, from a sample that is really one mesh twice.

| | St | Cd | lift amplitude |
| --- | --- | --- | --- |
| mean, inlet at −10 D (ADR-029's three meshes) | 0.1702 | 1.4173 | 0.3642 |
| mean, inlet at −20 D (A, C, D) | 0.1689 | 1.4121 | 0.3636 |
| shift | −0.0013 | −0.0053 | −0.0006 |
| shift in standard deviations of the difference (0.0005, 0.0043, 0.0031) | −2.6 | −1.2 | −0.2 |

**Verdict under the rule: the inlet is a significant part of the Strouhal
excess.** St fell by 0.0013, past the 0.0010 threshold. The inlet's whole
effect at −10 D is then about twice that: St −0.0026, ±0.0010 with the
shift's standard deviation doubled. The drag shift is 1.2 standard
deviations, which the test does not resolve; if it is real, the whole effect
is about −0.011 ± 0.009.

**The two shifts do not fit the estimate above well.** The St shift exceeds
its stated upper bound (−0.0009) by less than one standard deviation, which
alone would be unremarkable. But the picture behind the bound — the body
meeting a slightly faster stream — ties the two together, drag moving about
fifteen times as much as St (0.014 against 0.0009). Measured, it moved four
times as much. Matching the St shift at face value would need a drag shift
near −0.02, more than three standard deviations from the −0.0053 measured. Either St
landed on the high side of its noise and drag on the low side, or the inlet
reaches the shedding frequency by some route other than the approach
velocity. The verdict does not depend on which, but the whole-effect figures
are rough and should be read with their error bars.

**Combined picture at D/17 and dt = 0.05**, taking the confinement effects
as additive, to first order:

- *Strouhal number.* 0.1702 on the −10 D, ±10 D domain; the inlet's whole
  effect −0.0026 ± 0.0010; the lateral boundaries' at most −0.0015
  (ADR-029). An unconfined value of about 0.166–0.168: 1–2% above
  Williamson's 0.1643 instead of 3.6%, and the lower end reaches the 0.166
  that ADR-029 would have counted as explained. The inlet is about 40% of
  the excess (0.0026 of 0.0059), and with the lateral boundaries up to 70%.
  Resolution barely moves St (ADR-027); the time step and the Rhie-Chow form
  each move it by a few ten-thousandths (ADR-030, ADR-031).
- *Drag.* 1.4173 on the same domain; lateral −0.022 (ADR-029); inlet about
  −0.011, unresolved; resolution −0.018 to −0.024 from D/17 to D/33
  (ADR-027 — the range is whether the D/17 value is `debug.hex` or the
  three-mesh mean). About 1.36–1.37, the top of the literature band, before
  ADR-033 adds the finer levels.

**What this changes.**

- A control is independent only if it re-places the nodes near the body,
  and moving a boundary by a thousandth of a diameter does not guarantee
  that. From here, a mesh counts as a separate sample only after its median
  near-body node offset from every other sample has been measured at 0.02 D
  or more. ADR-029's three meshes pass (0.025–0.027 D).
- The wake gate's case keeps its −10 D inlet and ±10 D sides, so that its
  history stays comparable and ADR-033 compares like with like. A comparison
  with the literature needs a larger domain than the gate's; by these
  estimates even −20 D and ±20 D would leave about 1% of confinement in St
  (0.0013 from the inlet, 0.0005 from the sides).
- What is left of the Strouhal excess after confinement, 0.002–0.003 with
  about 0.001 of uncertainty from the two extrapolations, is probably real
  but no longer large. ADR-033 says whether resolution holds any of it; the
  outlet at +25 D is the one boundary not yet moved.

## ADR-033 — Three-level grid convergence of the wake. Stated before the run, answered after
**Question.** ADR-027 had two meshes, which can show that resolution moves
the drag but cannot give a grid-independent value: Richardson extrapolation
needs a third level to measure the order instead of assuming it. Written and
committed before the runs.

**Test.** A family refined by √2 in every size the generator takes —
the wall size, the far-field size, and so the cell count by about two per
level — with the domain, dt = 0.05 and the consistent Rhie-Chow form as in
every earlier comparison:

| level | sizeMin | sizeMax | cells | first cell |
| --- | --- | --- | --- | --- |
| C | 0.0424 | 0.849 | 10,216 | D/24 |
| M | 0.03 | 0.6 | 21,811 | D/33 — the refined mesh of ADR-027, already run |
| F | 0.0212 | 0.424 | 42,020 | D/47 |

`debug.hex` is not in the family — its far-field size is not scaled with its
wall size — so it takes no part in the extrapolation.

**What to expect, and what can be read.** If ADR-027's drag drop between
D/17 and D/33 (0.024) is second-order discretisation error, the family's
drag differences should be about 0.008 between C and M and 0.004 between M
and F, converging to about 1.39. St barely moved with resolution before,
so its level-to-level differences should be tiny. Against that, ADR-029
measured node-placement scatter at D/17 of 0.0006 in St and 0.005 in drag,
and each level here is one mesh. The fine-level difference is below that
scatter unless the scatter shrinks with the cells, which it should but which
has not been measured.

**Decision rule.**

- With both drag differences of one sign and an observed order between
  0.5 and 4, report the extrapolated value with Roache's grid convergence
  index (safety factor 1.25).
- Otherwise the family does not resolve an order at this scatter: report
  the F value with an uncertainty equal to the larger level-to-level
  difference, and say so.
- For St, the same; if |St(C) − St(F)| < 0.0017 (two standard deviations
  of a difference at the D/17 scatter), St is grid-converged within the
  scatter and no extrapolation is attempted.

Whatever comes out is combined with ADR-029's blockage extrapolation
(drag −0.022 as B → 0) to say how far a grid-converged, unconfined value
sits from the literature.

### Result

Everything above this heading was committed before the first run. The two
runs were started once, lost when the machine running them was reclaimed,
and started again from scratch with the same binary; nothing was read from
the lost attempt, whose output had not yet been flushed.

| level | cells | first cell | St | Cd | lift amplitude | cost |
| --- | --- | --- | --- | --- | --- | --- |
| C | 10,216 | D/24 | 0.1695 | 1.4121 | 0.3582 | 1.15 s/step |
| M | 21,811 | D/33 | 0.1699 | 1.3991 | 0.3491 | ADR-027 |
| F | 42,020 | D/47 | 0.1697 | 1.3939 | 0.3457 | 4.34 s/step |

All three: dt = 0.05, 4,000 steps, 13 shedding cycles after t = 120, the
fastest cell 1.36–1.37 beside the cylinder.

**Drag.** The differences are 0.0130 (C to M) and 0.0052 (M to F), of one
sign, for an observed order of 2.64 — inside the rule's 0.5 to 4, and close
to the scheme's nominal two. Richardson extrapolation gives **Cd = 1.390**,
with a grid convergence index of 0.31% on the fine level (±0.004). The
prediction written above was differences of about 0.008 and 0.004 converging
to about 1.39; the differences came out larger, the limit where predicted.

**Strouhal number.** 0.1695, 0.1699, 0.1697: the differences change sign and
St(C) − St(F) is 0.0002, far inside the 0.0017 the rule allows. St is
grid-converged within the scatter, at 0.1697; no extrapolation.

**Lift amplitude**, not in the rule but read the same way: differences
0.0091 and 0.0034, observed order 2.84, extrapolated 0.344 (±0.003).

**The caveat the entry above named still stands.** Each level is one mesh,
and ADR-029 measured node-placement scatter of 0.005 in drag at D/17; the
M-to-F difference is the size of that scatter. The GCI does not include it.
That the observed order lands near two, and the limit where the two-grid
estimate of ADR-027 put it, is the evidence that the scatter shrinks with
the cells, not a measurement of it.

**Combined with the domain.** ADR-029's blockage extrapolation moves drag by
−0.022 and ADR-032's inlet estimate by about −0.011 (unresolved, ±0.009).
Grid-converged and unconfined, drag is then about **1.357**, the top of the
1.32–1.36 band; with the blockage alone, 1.368. The Strouhal number, 0.1697
on every level, less 0.0026 for the inlet and up to 0.0015 for the sides, is
0.166–0.167: 1–2% above Williamson's 0.1643. Resolution is not part of the
Strouhal excess at all; confinement is most of it.

**What this changes.**

- The benchmark's drag excess is accounted for, to within the band. From
  the D/17 mean of 1.417 (ADR-029's three meshes) to the grid limit of 1.390
  is 0.027 of resolution; confinement is about 0.033 more (blockage 0.022,
  inlet about 0.011).
- The Strouhal number is not a resolution problem. What is left of its
  excess after confinement, about 0.002–0.003, is not explained here.
- These are v1-flux numbers, as the entry specified. ADR-037 changes the
  Rhie-Chow flux; how much that moves the cylinder is measured there, on
  the benchmark mesh, and the grid study is not repeated unless it is large.

## ADR-034 — The boundary-pressure extrapolation: warm start and gradient caching. Stated before the runs, answered after
**Question.** ADR-025 left the boundary-pressure extrapolation at 19% of the
cylinder run and proposed the warm start that had just paid off for the
non-orthogonal correction: start each extrapolation from the boundary values
the previous one produced and sweep once, instead of three times from zero
normal gradient. Two changes are tested — that warm start, and handing over
the pressure gradient wherever it is recomputed from a pressure that has not
changed. Written and committed before the cylinder runs.

**Already known**, from runs made while the code was written, before this
entry; the rule below is written with them in view.

- *Caching.* Of the 42.5 pressure-gradient evaluations per step on the
  cylinder (169,902 in 4,000 steps, ADR-030's gate log), nine take the
  gradient of an unchanged pressure: one at the start of each of the three
  outer iterations, and one after each of the six correctors, whose pressure
  solve has just computed it. Handing it over is exact by construction. On
  Ethier–Steinman, at the same thread count, the errors, outer iterations,
  non-orthogonal sweeps and pressure iterations match the main build's in
  every printed digit. (Across thread counts the errors differ in the
  thirteenth digit, from the order of the reductions.)
- *Warm start, on Ethier–Steinman.* It changes the answer and costs more:
  orthogonal error at n = 8 goes from 5.5635e-3 to 5.6290e-3, a difference
  that shrinks faster than the error (3.5356e-4 against 3.5381e-4 at
  n = 32); the orthogonal runs need 6 outer iterations instead of 4, and the
  distorted n = 32 run 24,012 pressure iterations instead of 22,068. The
  reading at the time: three cold sweeps are a truncated fixed-point
  iteration, so a warm start, which keeps iterating across calls, computes a
  different boundary pressure, closer to the fixed point, and trails the
  current pressure by a sweep, which slows the outer loop.
- *The extrapolation's own convergence.* On the cylinder mesh each sweep
  halves the change the next one makes (a five-step instrument check:
  4.5e-2, 2.2e-2, 1.0e-2, 5.0e-3 of max|p| for sweeps 1–4, 6e-5 at sweep
  10). That is what the 1/d² least-squares weights predict for a wall cell,
  where the boundary face and the opposite neighbour carry equal weight: the
  error contracts by one half per sweep. Three sweeps leave about an eighth
  of the first change; ten leave about a thousandth.

Two things could make the cylinder differ from Ethier–Steinman. Its outer
loop is capped at three iterations and always uses all three (36,000
momentum solves in 4,000 steps), so a slower outer loop cannot cost
iterations there; it can only leave each step less converged. And its
boundary pressure changes little from one step to the next, which is where
ADR-025's warm start paid off.

**Test.** The development build, `debug.hex`, dt = 0.05, 4,000 steps, the
usual statistics window, one run at a time:

| run | extrapolation | role |
| --- | --- | --- |
| R1 | 3 cold sweeps, gradient cached | the proposed default; must reproduce the main build's 0.1698, 1.4233, 0.3671 |
| R2 | warm start, gradient cached | the candidate |
| R3 | 10 cold sweeps, gradient cached | the extrapolation close to its fixed point: the reference R1 and R2 are measured against |

and Ethier–Steinman (8, 16, 32, orthogonal and distorted) with the same
three settings. Every cylinder run prints, for its final field, the
largest change per sweep of a cold extrapolation for sweeps 1 to 30.

Cost is judged on counts, not seconds: identical runs of the gate's case
took 2,395 s in one suite and 2,897 s in another. Each run's modeled cost is
R1's wall time plus the difference in its counts — pressure iterations,
momentum iterations, extrapolation sweeps, gradient evaluations — priced at
R1's own seconds per unit. Wall time is reported beside it.

**Decision rule.**

- *Caching is kept* if R1 reproduces the main build's Strouhal number, drag
  and lift amplitude in every printed digit and its gradient evaluations
  fall by nine per step. Anything else is a defect, and it comes out.
- *The warm start is adopted* only if all three hold: (a) R2's modeled cost
  is at least 5% below R1's; (b) R2 is at least as close to R3 as R1 is,
  within St 0.0002 and Cd 0.002, i.e. |R2 − R3| ≤ |R1 − R3| plus that
  margin; (c) the C++ order gates pass with it. Adopting it means writing
  it into the Python reference first, because the cross-check compares the
  two implementations to 1e-10. Otherwise it stays off, as recorded
  evidence.
- *Truncation on its own terms.* If R3 differs from R1 by more than St
  0.0003 or Cd 0.003 — the thresholds ADR-031 used for a real change on one
  mesh — three cold sweeps are a discretisation error of their own on the
  benchmark, and an extrapolation swept to a tolerance (Python first)
  becomes the next change whatever happens to the warm start. Below that,
  three sweeps stand.
- R3 is a reference only if its final-field history shows the tenth sweep
  changing the boundary values by less than 1% of what the first did. If
  not, (b) is not judged and the entry says so.

### Result

Everything above this heading was committed before the runs below.

**Ethier–Steinman**, one thread, n = 32 (orders from 16 to 32):

| extrapolation | orthogonal L2 | order | outer iterations, last step | distorted L2 | order | pressure iterations, distorted |
| --- | --- | --- | --- | --- | --- | --- |
| 3 cold sweeps (default) | 3.53558e-4 | 1.991 | 4 | 5.92178e-4 | 1.964 | 22,068 |
| warm start | 3.53838e-4 | 1.998 | 6 | 5.94305e-4 | 1.969 | 24,144 |
| 10 cold sweeps | 3.53803e-4 | 1.998 | 4 | 5.94287e-4 | 1.969 | 22,101 |
| 20 cold sweeps | 3.53808e-4 | 1.998 | 4 | 5.94305e-4 | 1.969 | 22,100 |

The warm start converges to the extrapolation's fixed point: it matches 20
cold sweeps to 5e-6 at n = 8 and 3e-8 at n = 32. Converging the
extrapolation cold costs no outer iterations; the warm start's extra ones
come from trailing the pressure by a sweep, not from the answer it reaches.
(From committed code the distorted run takes 24,144 pressure iterations; the
smoke build quoted above took 24,012.)

**The cylinder**, `debug.hex`, dt = 0.05, 4,000 steps, two threads:

| run | St | Cd | lift | pressure solves / iterations | extrapolation sweeps | gradient evaluations | modeled cost | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main build (ADR-030's log) | 0.1698 | 1.4233 | 0.3671 | 133,902 / 742,895 | 521,706 (3 per call) | 169,902 | — | 2,395 s |
| R1: 3 cold, cached | 0.1698 | 1.4233 | 0.3671 | 133,902 / 742,895 | 413,709 | 133,903 | 2,512 s | 2,512 s |
| R2: warm start | 0.1698 | 1.4234 | 0.3669 | 266,838 / 1,407,702 | 270,841 | 266,839 | +56% | +52% |
| R3: 10 cold | 0.1698 | 1.4234 | 0.3669 | 136,487 / 754,006 | 1,404,880 | 136,488 | +39% | +37% |

Every run: fastest cell 1.35–1.36 beside the cylinder. R3's final-field
history — 5.6e-2, 2.7e-2, 1.3e-2, 6.2e-3 of max|p| for sweeps 1–4, 7.6e-5 at
sweep 10 — makes it a valid reference (the tenth sweep 0.14% of the first).

**Verdicts under the rule.**

- *Caching is kept.* R1 reproduces the main build's Strouhal number, drag
  and lift amplitude, and in fact every line of its force history; the
  gradient evaluations fall by exactly 9.00 per step, worth 5.1% of the
  uncached run at R1's own prices. (The momentum iterations differ by two in
  425,132: with two threads the atomic accumulations sum in a different
  order, which the history does not see.)
- *The warm start is rejected.* It passes (b) — R2 equals R3 in every
  printed digit — and fails (a) by a wide margin: +56% modeled, +52% wall.
  On the cylinder it is not the outer loop that pays but the non-orthogonal
  loop, which calls the extrapolation once per sweep: a boundary pressure
  that trails by a sweep makes each sweep's correction lag, and the loop
  needs twice the sweeps to converge. It stays off, as recorded evidence.
- *Three sweeps stand.* R3 differs from R1 by 0.0000 in St, 0.0001 in drag
  and 0.0002 in lift, against thresholds of 0.0003 and 0.003. Each sweep
  halves what the next one changes, three leave about 1% of max|p| at the
  worst boundary face, and the forces do not notice.

**What this changes.** The boundary-pressure extrapolation keeps its three
cold sweeps and now runs 43.5 − 9 = 34.5 times per step instead of 43.5. The
19% ADR-025 left on the table is not recoverable this way: it is the price of
a fixed-point iteration that has to restart, because the loop that calls it
cannot tolerate a lagging answer.

## ADR-035 — Each rank reads only its share of the mesh file
**Decided.** A binary mesh description, `.vmesh`, and a `DistributedMesh`
constructor that reads it in parts, so that no rank ever holds the whole
description. ADR-023 made each rank build only its own subdomain and named
what it left replicated: every rank still read all the points and all the
connectivity, so the largest mesh was still the largest one a single rank
could hold as a description.

**The format.** An eight-byte tag, `VFMESH01`, the point and cell counts as
64-bit integers, then three doubles per point and eight 64-bit vertex ids per
cell. Records have a fixed size, so any block of points or cells is one seek
and one read. The text `.hex` format cannot be read in parts, which is why a
new format was needed at all.

**The read**, in order, each rank:

1. reads one contiguous block of cells and one of points; holding a point
   block makes it the directory for those ids;
2. fetches the coordinates its cells need from their directory ranks and
   computes centroids;
3. sorts cells by the Morton key of their centroids, with a sample sort, and
   cuts the sorted order into exactly equal parts;
4. finds candidate ghosts — every cell sharing a vertex with an owned one, as
   on the RawMesh path — through the point directory: each rank registers
   its cells' vertices with their directory ranks, which tell it which other
   ranks' cells share them;
5. fetches the ghosts' connectivity from their owners and every coordinate it
   still lacks, and hands the subset to the same build as the RawMesh path.

**The gate**, `mms_parallel_read`, measures what each rank held: the peak
bytes of description, points at 24 bytes and cells at 64 as in the file, on
the worst rank, against the build-memory gate's bar of 1/P + 0.35 of the
file. A reader that skipped cells would pass that easily, so the same run
solves `mms_parallel`'s diffusion problem on the mesh it read, and the L2
error must match the serial run to 1e-10 — although the partition is Morton
order, not RCB, so it differs from every other gate's.

| mesh | ranks | description held, worst rank | bar | L2 against serial |
| --- | --- | --- | --- | --- |
| 16³ skewed fixture | 2 | 0.726 | 0.850 | within 7e-14 |
| | 3 | 0.607 | 0.683 | within 7e-14 |
| | 4 | 0.403 | 0.600 | within 7e-14 |
| 40³ smooth | 2 | 0.672 | 0.850 | |
| | 4 | 0.354 | 0.600 | |

**Verified to fail.** The replicated path, kept in the gate for this, holds
1.000 of the file on every rank and fails. The earlier parallel gates are
unchanged: `mms_parallel` within 4e-14, the build-memory gate at 0.288.

**Not done.** The `.hex` and CGNS readers still read the whole file on every
rank; a mesh reaches the distributed path by being written as `.vmesh`,
which for now only code does (`vmesh::write`). The solver cases still read
`.hex`.

**Merge condition.** The work sits on the development branch with ADR-034's
gradient caching and ADR-036's sweep switch. It merges after the full suite
passes on that branch, recorded below.

### Full suite

The suite ran on the branch that carries this work plus ADR-037's flux
change (and ADR-034's caching and ADR-036's switch), not on this work alone:
the machine time went to one run of a superset instead of two. All 23 gates
pass — the 8 of v0 and 15 of v1, this entry's `mms_parallel_read` among them
(0.726 / 0.607 / 0.403 of the file on 2 / 3 / 4 ranks, the L2 error within
9.3e-14 of serial). Had the superset failed on ADR-037's account, this work
would have been run alone before merging.

## ADR-036 — Momentum predictor sweeps for the deferred correction. Stated before the runs, answered after
**Question.** The roadmap's last v1 item was to make the deferred correction
implicit. Convection is upwind in the matrix plus a deferred correction
evaluated from the previous iterate, so with one momentum solve per outer
iteration the correction lags a whole outer iteration. What was built is
the conservative version: up to N momentum solves per outer iteration, each
re-evaluating the correction from the last (`convectionSweeps`, default 1).
The matrix, and so aP, is unchanged, so the converged answer is the same one
and only the route to it changes. A truly implicit central matrix was not
built: aP would change, can vanish at cell Péclet numbers above two, and
enters the Rhie-Chow flux through D = V/aP, so it would be a different
scheme with a different answer. Written and committed before the runs.

**Test.**

1. Ethier–Steinman, 8/16/32 on both meshes, native pressure solver, one
   thread, with the outer-iteration cap raised from 6 to 20 so that every
   run converges its outer loop (the distorted n = 32 run reaches the cap of
   6 today): 1, 2 and 3 sweeps.
2. The cylinder, `debug.hex`, dt = 0.05, 4,000 steps, 2 sweeps, against
   ADR-034's R1: the same binary with 1 sweep.

Cost is modeled from counts as in ADR-034: the 1-sweep run's seconds per
momentum iteration and per pressure iteration (and, on the cylinder, per
extrapolation sweep and gradient evaluation), times each run's counts.

**What to expect.** Sweeps remove only the deferred correction's lag. The
outer loop also converges the convecting flux and the pressure-velocity
coupling, which sweeps do not touch, so the outer count should fall by one
at most, while every sweep adds three momentum solves. On the cylinder the
outer loop is capped at three and always uses all three; there, sweeps
cannot save outer iterations unless they bring the loop under its tolerance
(1e-7) sooner, and whatever they do to the answer measures how much the
capped loop leaves the correction lagging.

**Decision rule.**

- *Consistency.* On Ethier–Steinman the errors with 2 and 3 sweeps must
  match the 1-sweep errors to 1e-8 relative. A larger difference means the
  sweeps moved the fixed point, which is a defect to find, not a result.
- *A new default* only if it lowers the modeled cost on both cases — by at
  least 10% at n = 32 on both Ethier–Steinman meshes, and at all on the
  cylinder — while the cylinder values stay within St 0.0003 and Cd 0.003 of
  R1. Otherwise the switch stays, off.
- *A finding either way.* If the cylinder values with 2 sweeps move beyond
  St 0.0003 or Cd 0.003, the benchmark's capped outer loop leaves a lag of
  that size; it is recorded as a known limit of every cylinder number so
  far, whatever the default.

### Result

Everything above this heading was committed before the runs below.

**Ethier–Steinman**, outer cap 20, one thread, whole-run work at n = 32
(modeled cost at the 1-sweep run's prices per momentum and pressure
iteration):

| sweeps | orthogonal: outer / momentum it. / pressure it. | modeled cost | distorted: outer / momentum it. / pressure it. | modeled cost |
| --- | --- | --- | --- | --- |
| 1 | 9 / 111 / 3,497 | 1.00 | 12 / 140 / 22,068 | 1.00 |
| 2 | 8 / 186 / 3,109 | 0.93 | 12 / 254 / 22,079 | 1.01 |
| 3 | 8 / 258 / 3,110 | 0.96 | 12 / 350 / 22,079 | 1.02 |

The errors with 2 and 3 sweeps match the 1-sweep errors to within 1.7e-10
relative on every mesh: the fixed point is the one it was. One statement
above was wrong: the distorted n = 32 run does not reach the cap of 6 today.
With the cap at 20 it still takes 6 outer iterations in its last step — it
had converged at exactly 6.

**The cylinder**, 2 sweeps, against R1 of ADR-034 (the same binary with 1):

| sweeps | St | Cd | lift | outer per step | momentum solves / iterations | pressure iterations | modeled cost | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (R1) | 0.1698 | 1.4233 | 0.3671 | 3.00 | 36,000 / 425,130 | 742,895 | 2,512 s | 2,512 s |
| 2 | 0.1698 | 1.4233 | 0.3671 | 3.00 | 72,000 / 881,726 | 738,111 | +6.0% | +9.8% |

**Verdicts under the rule.**

- *Consistency holds.*
- *No new default.* The saving is 7% on the orthogonal mesh and −1% on the
  distorted one, against a bar of 10% on both, and the cylinder costs 6%
  more. The switch stays, off.
- *No finding against the benchmark.* With the correction re-evaluated
  inside every outer iteration the cylinder's values do not move in any
  printed digit, and its outer loop still uses all three iterations: what
  keeps it from reaching 1e-7 is not the deferred correction's lag. Every
  cylinder number so far stands as it was.

**What this changes.** The roadmap item "make the deferred correction
implicit" is closed. The conservative version exists, is verified to reach
the same answer, and buys nothing on these cases; a truly implicit
higher-order matrix would change aP and the Rhie-Chow flux with it, and is a
different scheme, to be argued for on its own if a case ever needs it.

## ADR-037 — A dt-consistent Rhie-Chow flux, ADR-010's gate first. Stated before the change, answered after
**Question.** ADR-031 traced the wake's dt drift, and a spurious velocity
growing in the far wake at dt = 0.025, to the old-flux (Choi) term of the
Rhie-Chow flux, and asked for ADR-010's gate before any change. This entry
states the gate, the diagnosis, the candidate and what decides it. Written
and committed before the flux is changed.

**The gate** (`prototype/ethier_steinman.py`, `gate_dt_independence`,
committed failing on the development line before any change to the flux).
Two steady manufactured problems — the existing one with constant pressure,
and a new one with p = cos πx cos πy cos πz so that the pressure-damping part
of the flux has work to do — on an orthogonal and a randomly skewed 8³ mesh,
each run from rest until u and p stop changing by 1e-13 per step, at
dt = 0.02, 0.2 and 2.0. A steady state has no temporal error — the BDF2 terms
cancel exactly — so the three states must coincide. The spread is the
largest L2 difference between two of them, relative to the discretisation
error of the dt = 2.0 run, for u and for p. Bound: 1e-6; a formulation with no
dt in its steady equations meets it at the iteration tolerance, about 1e-10.
The current form fails, and so does the naive one:

| form | mesh | spread u / p, constant p | spread u / p, grad p |
| --- | --- | --- | --- |
| v1 old-flux | orthogonal | 8.1e-4 / 9.9e-4 | 4.8e-3 / 7.1e-3 |
| v1 old-flux | skewed | 6.9e-1 / 1.5 | 6.6e-1 / 1.5 |
| naive | orthogonal | 2.1e-2 / 3.6e-2 | 9.4e-2 / 1.5e-1 |
| naive | skewed | 6.7e-2 / 1.8e-1 | 1.1e-1 / 2.3e-1 |

On the skewed mesh the current form moves the steady velocity by 69% of its
own discretisation error between dt = 0.02 and 2.0. One exploratory run came
before the gate and pointed at the cause: the old reporting check, which
compared L2 values rather than fields, gave 9.2e-5 on the orthogonal mesh
against 3.9e-2 on the skewed one.

**Diagnosis.** The predicted flux interpolates H/aP with the skewness
correction — call it I — without which the scheme is first order on skewed
meshes (the comment in `rhie_chow` records that). The old-flux residual
subtracts a plain linear interpolation L of the old velocity instead:
R = F − L[u]·S. With β = D_f·a0, the transient share of aP (D_f the
interpolated V/aP, a0 = 3/(2dt)), a steady state satisfies

    R (1 − β) = (I − L)[u]·S  +  I[(V/aP) ∇p]·S − D_f × (compact gradient terms)

The first term is the skewness correction of the velocity itself: second
order, zero on an orthogonal mesh, and divided by 1 − β = aP_s/aP, which in
a cell is about (2/3)·Co and so falls in proportion to dt. In the cylinder's
far wake, where cells are a diameter across and Co ≈ 0.025 at dt = 0.025,
it is multiplied by about sixty. That is where ADR-031's spurious velocity
lived. The second group leaks too, less: the interpolation of the product
(V/aP)∇p is not the product of interpolations, and interp(V/aP) does not
satisfy 1/D_f − a0 = (something dt-free), so 1 − β does not cancel even on
an orthogonal mesh — the 8e-4.

**The candidate**, three changes, each needed for the steady equations to
lose dt exactly:

1. The residual uses the same interpolation as the prediction:
   R = F − I[u]·S, the old velocity interpolated with its own skewness
   correction.
2. The predicted flux interpolates q = H/aP − (V/aP)∇p — the velocity the
   last pressure implies — with the skewness correction, and adds the
   pressure back with the face coefficient: F* = I[q]·S + D_f L[∇p]·S +
   D_f a0 R. Both pressure terms are interpolated cell gradients, blind to a
   checkerboard, so this is not the compact old-pressure term ADR-026 found
   turning the checkerboard into an eigenvalue of −1.
3. D_f = V_f / aP_f, volume and aP interpolated separately, so that
   1/D_f − a0 = aP_s,f / V_f exactly.

Then a steady state has F = I[u]·S + (V_f / aP_s,f)(L[∇p]·Δ − a_f (p_N − p_P)),
with Δ the orthogonal part of the face vector: the Rhie-Chow damping with the
spatial part of aP only, and no dt anywhere. The same treatment applies at
an outlet face, with the cell value in place of an interpolation.

**Predictions.** The gate's default spreads at the iteration tolerance. The
Ethier–Steinman orders unchanged in character (second order on both
families). On the cylinder at dt = 0.025, no far-wake mode: the fastest cell
within 1.5 over the whole statistics window (the physical peak beside the
cylinder is 1.33–1.36 on every run so far), and St and drag within 0.0003
and 0.003 of their dt = 0.05 values, as the naive form was (+0.0003,
+0.0015, ADR-031). The cylinder numbers themselves move, since the damping
coefficient changes; by how much is not predicted.

**Decision rule.** The candidate is adopted if all of these hold:

1. the Python gate passes, every spread ≤ 1e-6;
2. every other Python gate passes;
3. the C++ implementation passes a C++ port of the gate at the same bound,
   and the Python–C++ cross-checks agree as tightly as before;
4. the full suite passes;
5. on the cylinder at dt = 0.025, both predictions above hold.

If (1) fails, the derivation is wrong somewhere; the gate stays as written
and the candidate is not adopted. If (1)–(4) hold and (5) does not, the
steady-state defect is fixed but the far-wake mode has a second cause: the
candidate is adopted, since the gate is the specification, and (5) is
recorded as open.

**The wake gate tightens either way.** The cylinder case will record the
fastest cell at every step of the statistics window, not every 160th, and
the gate will require it to stay within 1.5 — ADR-031's recommendation. A
far-wake mode like the one at dt = 0.025 fails that; the physical flow, at
1.36, passes with room.

### Result

Everything above this heading was committed before the flux changed. On the
development line the Python gate was committed failing, then the Python
change, then the C++ port of the gate, failing, then the C++ change — four
commits in that order.

**(1) The Python gate passes**, every spread at the iteration tolerance, and
the discretisation errors do not move (L2 of u 1.87e-2, 2.00e-2, 1.95e-2,
2.08e-2 in both forms): dt leaves the answer and nothing else changes.

| problem | mesh | exact form, spread u / p | v1 form, spread u / p |
| --- | --- | --- | --- |
| constant p | orthogonal | 9.1e-12 / 9.9e-12 | 8.1e-4 / 9.9e-4 |
| constant p | skewed | 1.2e-11 / 9.5e-12 | 6.9e-1 / 1.5 |
| grad p | orthogonal | 7.5e-12 / 8.4e-12 | 4.8e-3 / 7.1e-3 |
| grad p | skewed | 1.2e-11 / 9.4e-12 | 6.6e-1 / 1.5 |

**The diagnosis, tested past the gate.** With dt = 0.002 added — a
thousandfold range — the exact form spreads 1.0e-9 to 1.2e-9 (the per-step
tolerance leaves more at the smallest step), while the v1 form's skewed-mesh
spread grows tenfold with the tenfold smaller step: 6.9 in u and 15 in p, the
steady velocity moving by seven times its own discretisation error. On the
orthogonal mesh, where the skewness term is absent, v1 grows only from 8.1e-4
to 1.3e-3. That is the 1/dt the derivation predicts, and where it predicts it.

**(2) Every other Python gate passes.** Ethier–Steinman orthogonal 1.990
(unchanged); smooth distortion 1.958, was 1.897; BDF2 2.428 (unchanged); the
warped family, reported only, 1.177, was 1.119.

**(3) The C++ port.** Run on the unchanged C++ flux, the C++ gate reproduced
the Python measurement of the v1 form digit for digit; with the change every
spread is 8e-11 to 1.1e-10. Ethier–Steinman: orthogonal 1.991 (unchanged),
smooth distortion 2.034, was 1.964, with the n = 32 error 5.6% lower. The
Navier-Stokes cross-check agrees to 5.1e-10 at the default non-orthogonal
tolerance, where it agreed to 6.6e-11 before — looser — but with the C++ loop
converged to 1e-12 the two implementations agree to 3e-12, so the gap is that
tolerance, not a difference in the formulation. The convection cross-check is
unchanged at 4.6e-13; Navier-Stokes on 2–4 ranks agrees with serial within
7.6e-15.

**(4) The full suite passes**: all 23 gates, the new C++ one among them.
The decoupling gate is where it was (fastest cell 1.34, drag 1.3008 against
1.2996); the Ghia cavity does not move in any printed digit.

**(5) The cylinder at dt = 0.025.** Both predictions hold, the drift is gone,
and so is the far-wake mode:

| flux | dt | St | Cd | lift amplitude | fastest cell, t > 120 |
| --- | --- | --- | --- | --- | --- |
| v1 (ADR-030) | 0.05 | 0.1698 | 1.4233 | 0.3671 | 1.36, beside the cylinder |
| v1 (ADR-030) | 0.025 | 0.1707 | 1.4141 | 0.3711 | **2.13, 17.5 D downstream** |
| naive (ADR-031) | 0.05 → 0.025 | +0.0003 | +0.0015 | +0.0010 | 1.36 |
| exact | 0.05 | 0.1689 | 1.4317 | 0.3681 | 1.361, beside the cylinder |
| exact | 0.025 | 0.1690 | 1.4316 | 0.3682 | 1.361, beside the cylinder |

From dt = 0.05 to 0.025 the exact form moves St by +0.0001, drag by −0.0001
and lift by +0.0001, against the prediction's bounds of 0.0003 and 0.003 —
and against BDF2's own estimate, in ADR-030, of +0.0001 in St for that
halving. The fastest cell is judged at every step of the window now, and
never leaves the cylinder's side.

**Verdict under the rule: adopted.** All five conditions hold.

**What it changed besides.** At the gate's dt the cylinder moves: St −0.0009
(0.1698 to 0.1689), drag +0.0084, lift +0.0010. The Strouhal number is now
2.8% above Williamson on the benchmark domain, and 0.3–1.2% above it once
ADR-032's inlet and ADR-029's sides are taken off (0.1689 − 0.0026 − up to
0.0015 = 0.1648–0.1663), if those corrections, measured with the v1 flux,
carry over. The grid study of ADR-033 used the v1 flux; if the
shift is the same on every level, its drag limit moves from 1.390 to about
1.398 and the unconfined drag to about 1.365, just above the band. It has
not been repeated.

**Why it survived.** ADR-010 named this trap in the first week of v1 and
asked for this gate; it was written as a report, found the wrong direction
(the form meant to remove the dt dependence was the one that showed it),
and was left ungated as inconclusive. The test was right. What it measured
was not decoupling but an interpolation mismatch that no order study can
see, because both interpolations are second order, and that the benchmark's
own time step kept small until a halving of dt made it sixty times larger in
the far wake.

## ADR-038 — v2 begins with heat transfer and buoyancy; its gates, stated before the code
**Decided.** v2 (turbulence and heat transfer, roadmap: k-ω SST, wall
functions, energy equation, Boussinesq buoyancy, conjugate heat transfer) is
split, and heat transfer goes first:

| part | scope | gates |
| --- | --- | --- |
| v2a | energy equation, Boussinesq buoyancy, slip walls | below |
| v2b | k-ω SST (Menter 2003), low-Reynolds wall treatment | manufactured solution for the model equations; flat plate Cf and log law; backward-facing step reattachment — each stated before its code |
| v2c | wall functions, conjugate heat transfer | stated when v2b passes |

*Why this order:* v2a reuses the verified convection–diffusion operator and
builds the one thing every later part needs — a transported scalar coupled
into the pressure-velocity loop, which k and ω will be too — against
benchmarks with sharp reference values. v2b is larger and its first gate
further away. The user chose this order.

### v2a design

- **Temperature** obeys ∂T/∂t + ∇·(uT) = ∇·(κ∇T) + S, discretised like a
  momentum component: BDF2, upwind in the matrix plus the deferred
  correction to the skew-corrected face value, diffusion with the
  non-orthogonal correction. Boundary faces: fixed temperature, or fixed
  heat flux (zero for an adiabatic wall). It is solved inside every outer
  iteration, after the pressure correctors, with the corrected face flux, so
  that at a converged outer loop the coupling carries no lag.
- **Buoyancy** is the body force f = −β(T − T_ref(x)) g, per unit mass,
  added to the momentum source. T_ref(x) is a reference stratification,
  constant by default or linear along g. Its buoyancy is a gradient, so it
  is absorbed into the pressure analytically, and a fluid resting in exactly
  that stratification is an exact discrete fixed point on any mesh.
  Without a matching reference, the discrete hydrostatic balance is not
  exact on distorted meshes. The complete remedy — the body force in the
  face flux and the cell velocity reconstructed from face fluxes, as
  OpenFOAM's buoyantBoussinesqPimpleFoam does with p_rgh and phig — would
  change the velocity correction every v1 gate stands on, and is deferred
  until a case needs it.
- **Slip walls**: a velocity boundary type with zero normal velocity and
  zero tangential stress, so that a symmetry plane can bound a roll.

### v2a gates

Python first, then C++, each written and shown to fail or skip before the
code it judges.

1. **Energy equation in an exact flow.** Ethier–Steinman flow with a
   manufactured temperature T = e^{−t} sin πx sin πy sin πz and the source
   that makes it exact, β = 0. Spatial order on 8³/16³/32³ (6³/12³/24³ in
   Python) within the Navier–Stokes gate's bands: [1.85, 2.15] orthogonal,
   [1.6, 2.3] and rising on the smooth distortion. BDF2 order in time for T
   at least 1.8.
2. **Boussinesq, steady, manufactured.** The steady solenoidal velocity of
   ADR-037's gate, p = cos πx cos πy cos πz, T = 1 + ½ sin πx sin πy sin πz,
   buoyancy on (β g = (0, 0, −1) per unit temperature), sources that make all
   three exact. Order of u and T in the same bands. The same steady state at
   dt = 0.2 and 2.0 to 1e-6 of the discretisation error: the buoyancy coupling
   must not bring dt back in.
3. **A stratified fluid at rest stays at rest.** The Rayleigh–Bénard
   conduction profile, T linear between a hot bottom and a cold top,
   adiabatic sides, the reference stratification equal to it, on a
   Cartesian and a distorted mesh: after 50 steps max|u| ≤ 1e-12 and T
   linear to 1e-12. With a constant reference instead, the spurious velocity
   is reported, not gated.
4. **Rayleigh–Bénard onset.** Rigid, isothermal plates one unit apart; slip,
   adiabatic side walls π/k_c apart, which hold exactly one roll of the
   infinite layer's critical mode; Pr = 1. Linear growth rates, from a
   perturbation of 1e-6, at Ra = 1600, 1700 and 1800, on meshes of 16, 24
   and 32 cells across the layer; each mesh's critical Ra is the zero of the
   growth rate interpolated in Ra. Reference: Ra_c = 1707.762, k_c = 3.117
   (Chandrasekhar; Scholarpedia's Rayleigh–Bénard article). Pass: the finest
   mesh within 1%, the observed order in [1.5, 2.6], the Richardson
   extrapolation within 0.3%.
5. **De Vahl Davis cavity.** The differentially heated square cavity, air
   (Pr = 0.71), Ra = 10³, 10⁴, 10⁵, 10⁶, run to a steady state on uniform
   meshes of 32², 64² and 128². The hot wall's mean Nusselt number,
   Richardson-extrapolated, within 0.5% of 1.1178, 2.2448, 4.5216 and 8.8252
   (Wang et al., Hortmann et al., Le Quéré, as tabulated in a lattice
   Boltzmann validation study that cites them; de Vahl Davis's own wall
   values are 1.117, 2.238, 4.509, 8.817), and on 128² within 1%. The maximum
   horizontal and vertical velocities on the mid-lines within 1% of de Vahl
   Davis's 3.649 / 3.697, 16.178 / 19.617, 34.73 / 68.59, 64.63 / 219.36
   (in units of κ/L).

**Not in v2a:** conjugate heat transfer, radiation, temperature-dependent
properties, the p_rgh reconstruction. Each needs a case to justify it.

### Revisions before the result

Two changes, made after the first Python run and before any C++ code, and
recorded here because they change what the gate would have said.

- *Gate 2's steady marching step.* The order runs used dt = 2.0. At 24³ that
  is a Courant number near 50, and the solver diverged there — plain
  Navier-Stokes without the energy equation included, measured separately —
  so the order runs use dt = 0.2. The dt-independence check still compares
  0.2 with 2.0, on the 8³ mesh where both converge. No criterion changed.
- *"Rising" becomes "approaching 2".* The rising-order condition was copied
  from the Navier-Stokes gate, whose pre-asymptotic orders approach two from
  below. Gate 2's approach it from above — 2.16 then 2.03 for u, 2.09 then
  2.04 for T on the smooth distortion — and fail it. For every v2a gate the
  condition is now that the last order is no further from 2 than the one
  before it, within 0.02. It catches a falling order as before, and a rising
  one past 2 as well; the Navier-Stokes gate's own orders (1.762 then 2.034)
  meet it. It is a change made after seeing the numbers whose verdict it
  changes, and it is recorded as that.

### The C++ gates, as written before the C++ code

Committed failing against a stubbed API in which every new entry point
throws. What each fixes beyond the plan above, before any run:

- *Gates 1–3* (`tests/mms/heat_transfer.cpp`) are the Python gate on
  8³/16³/32³. Two harness settings differ. Gate 2's order runs march at
  dt = 0.1: on 32³ a step of 0.2 would be a Courant number a third above
  the one Python's 24³ run converged at. And "steady" means a change per
  step below 1e-12, not 1e-13, because iterative linear solvers have a noise
  floor that direct factorisations do not — steady_dt's choice. Gate 3 also
  reports the spurious velocity under a constant reference, as the plan
  said and the Python gate left out.
- *Slip walls* (`tests/mms/taylor_green.cpp`): ADR-039's revised criteria on
  8³/16³/32³.
- *Two gates the plan did not list.* A cross-check
  (`tests/mms/crosscheck_v2.py`): gate 1's L2(T) and the Taylor–Green errors
  on 6³ and 12³ within 1e-4 relative of Python's, the NS cross-check's bound;
  gate 2's steady L2(u) and L2(T) on 6³ within 1e-6. And an MPI gate
  (`tests/mms/mms_parallel_heat.cpp`): gate 2's flow on a distorted mesh with
  slip and fixed-heat-flux faces, velocity and temperature each within 1e-10
  of the serial run on 2, 3 and 4 ranks.
- *Rayleigh–Bénard* (`tests/benchmark/rayleigh_benard.cpp`): the box π/k_c by
  1, one cell thick in y with slip, adiabatic faces. The growth rate is the
  least-squares slope of ln‖u‖ over 1.5 ≤ t ≤ 3.5 (dt = 0.01, four outer
  iterations); its two halves must agree within 1e-3 or the run fails. The
  critical Ra is the zero of the quadratic through the three points, and the
  observed order from the unequal ratios of 16/24/32 is solved for
  numerically. The neutral point depends on neither dt nor the outer count:
  a neutral mode is a steady solution of the linearised discrete equations,
  and the solver's steady states depend on neither (ADR-037).
- *De Vahl Davis* (`tests/benchmark/heated_cavity.cpp`): the velocity maxima
  are judged on 128² — the plan did not say which mesh. The Richardson
  extrapolation needs an observed order in [0.5, 4], as in ADR-033, or the
  check fails. Steady means the change per unit time of T and of u/u_ref
  below 1e-5, and dt is a Courant number of about one on the reference
  velocity. The Nusselt number is the solver's own wall heat flux.

### Revision to the C++ gate 2 harness, before its result

The first C++ run of gate 2 stalled on the 16³ distortion: the change per
step fell to 1.9e-9 and stayed there for hundreds of steps. It is the
non-orthogonal loop's tolerance, measured by varying it — the plateau sits at
about 190 times the tolerance (1e-11 → 1.9e-9, 1e-13 → 1.9e-11, 1e-14 →
1.9e-12). The warm-started loop ends a step's sweeps as soon as one moves the
correction by less than the tolerance, so the state wobbles at that level
from step to step. Python's loop starts cold and converges to 1e-14 on every
call, which C++ cannot afford: at 1e-14 a step on 32³ costs 55 s.

So the loop converges to 1e-12. The order runs call a state steady below 1e-8,
far below the errors they compare (1.4e-4 and up). The dt-spread check
compares two states to 1e-6 of a 2e-2 error, so it keeps 1e-12 on its 8³ mesh,
as steady_dt does. No criterion changed. The wobble is itself a finding about
the solver: on a distorted mesh a steady run cannot be called steady much
below 200 times nonOrthTol.

### Result

**All eight v2a gates pass.** Orders are the last of two: 8³→16³→32³ in C++,
6³→12³→24³ in Python.

| gate | Python | C++ |
| --- | --- | --- |
| 1. T in the exact flow, orthogonal / smooth distortion | 1.997 / 1.964 | 1.999 / 1.977 |
| 1. BDF2 in time | 2.153 | 2.153 |
| 2. Boussinesq u, orthogonal / distortion | 2.044 / 2.031 | 2.021 / 2.013 |
| 2. Boussinesq T, orthogonal / distortion | 2.005 / 2.038 | 2.003 / 2.024 |
| 2. dt = 0.2 against 2.0, spread of u / T (bound 1e-6) | 1.5e-13 / 9.5e-12 | 2.7e-12 / 1.0e-10 |
| 3. at rest, max\|u\| Cartesian / distorted (bound 1e-12) | 2.4e-14 / 1.0e-14 | 7.2e-15 / 1.1e-14 |
| 3. at rest, max\|T − (1 − z)\| | 3.4e-15 / 1.8e-15 | 1.3e-15 / 8.9e-16 |

The temporal errors are the same in both to seven digits (3.172e-5,
9.295e-5, 2.090e-5) and are not monotone: the coarsest step is the most
accurate, so the order is the last pair's, as the gate defines it.

- **Cross-check:** the sixteen rows agree to 9.5e-11 relative at worst,
  against bounds of 1e-4 (transient) and 1e-6 (steady).
- **MPI:** two, three and four ranks give the serial velocity to 8.7e-16 and
  temperature to 2.3e-15. The gate has teeth: leaving the temperature's
  ghosts one solve stale moves T by 1.0e-3 and u by 8.4e-6 on three ranks.
- **Rayleigh–Bénard onset:** critical Ra 1678.630, 1694.884 and 1700.563 on
  16, 24 and 32 cells across the layer. The observed order is 2.005 and the
  Richardson extrapolation 1707.842, +0.005% from 1707.762. The finest mesh
  is −0.42% off (bound 1%), the extrapolation 0.005% (bound 0.3%).
- **De Vahl Davis:** every condition passes at every Rayleigh number.

  | Ra | Nu 32² / 64² / 128² | observed order | Richardson Nu (reference) | u_max, v_max on 128² vs de Vahl Davis |
  | --- | --- | --- | --- | --- |
  | 10³ | 1.11909 / 1.11811 / 1.11787 | 2.01 | 1.11779 (1.1178), −0.001% | +0.01%, +0.01% |
  | 10⁴ | 2.26442 / 2.24970 / 2.24603 | 2.00 | 2.24482 (2.2448), +0.001% | +0.02%, +0.05% |
  | 10⁵ | 4.67173 / 4.55917 / 4.53101 | 2.00 | 4.52161 (4.5216), +0.000% | +0.06%, +0.10% |
  | 10⁶ | 9.72632 / 9.06324 / 8.88498 | 1.90 | 8.81945 (8.8252), −0.065% | +0.49%, +0.84% |

  The 128² Nusselt numbers are within 0.68% (bound 1%), the extrapolations
  within 0.065% (bound 0.5%). The closest call is Ra = 10⁶'s vertical
  maximum, 221.21 against de Vahl Davis's 219.36. The benchmark took 71
  minutes on two cores with BoomerAMG, 27 of them for Ra = 10⁶ on 128².

  Two harness changes were made before its first full run, neither to a
  criterion: the pressure solve uses BoomerAMG where PETSc is built, as the
  cylinder does (0.41 s a step on 128² against 1.47), and each mesh starts
  from the previous mesh's steady state, interpolated (below).
- **The whole suite:** `run_gates.py v0 v1 v2` passes all 31 gates in one
  run, none skipped: 3 h 28 min on two cores, 64 minutes of it v0 and v1.
  The v1 benchmarks did not move — cylinder St 0.1689, drag 1.4317, lift
  0.3681, fastest cell 1.361 — and the cavity repeated its separate run to
  every printed digit.

### Findings the gates did not ask for

- *Transient accuracy needs a converged outer loop.* The onset gate's four
  outer iterations give growth rates far below linear theory. At Ra = 1800
  on 32 cells the gate measured 0.173, against 0.694 from a Chebyshev
  solution of the linear problem — about 0.75 once that mesh's own offset of
  Ra_c is allowed for — and the shortfall grows with refinement. Converging the loop (100
  iterations, 1e-13) brings the same run to 0.743. The zeros do not move at
  all: 1678.630 and 1700.563 with four iterations or forty. At dt·ν/h² ≈ 10
  the pressure correction, which sees only the momentum diagonal, leaves the
  PIMPLE loop contracting slowly, so a small fixed count advances the slow
  modes by a fraction of dt. Steady answers are unaffected. Unsteady ones at
  large diffusion numbers are not, and no gate yet measures a transient
  coupling against an exact rate. *Correction (ADR-040):* 100 iterations did
  not converge the loop. 32 cells needs about 648 a step to reach 1e-10, so
  0.743 is the rate at 100 iterations, not the converged one.
- *The same slowness makes steady states on fine meshes expensive.* The
  cavity at Ra = 1e3 needed 4.9 time units to settle on 64², where 32² needed
  2.7, and 128² from rest was on course for hours. Starting each mesh from the
  last one's steady state halves the march (1173 steps to 613 on 64²) and
  leaves the answer unchanged; capping the diffusion number instead restores
  the physical settling time (0.66) at the same cost.
- *The non-orthogonal loop's wobble*, about 190 times nonOrthTol on 16³ (the
  harness revision above).
- *A constant reference leaves the resting fluid far from rest.* With
  T_ref = 0.5 in place of the conduction profile, gate 3's resting fluid
  reaches max|u| = 0.47 (Cartesian) and 0.48 (distorted) after 50 steps at
  Ra = 1700: the boundary-pressure extrapolation cannot carry the quadratic
  hydrostatic pressure. That is what the reference stratification is for,
  and what the deferred p_rgh treatment would fix in general.

## ADR-039 — Slip walls, gated by the Taylor–Green vortex. Stated before the code
**Decided.** ADR-038 adds a slip velocity boundary — zero normal velocity,
zero tangential stress — so that a symmetry plane can bound a roll, and
gives it no gate of its own. This is that gate, written before the code.

**The flow.** The two-dimensional Taylor–Green vortex,

    u = ( sin πx cos πy, −cos πx sin πy, 0 ) e^{−2π²νt},
    p = ¼ ( cos 2πx + cos 2πy ) e^{−4π²νt},

solves the Navier-Stokes equations exactly, and in the unit cube it meets
the slip condition on every wall: the normal component vanishes on each face
and the tangential components have zero normal derivative there. The roadmap
listed the Taylor–Green vortex as a v1 benchmark; Ethier–Steinman took its
place, and here it tests the one thing Ethier–Steinman cannot, a boundary
that is not a prescribed velocity.

**The gate.** Slip walls on all six faces, zero boundary flux, the exact
initial state, ν = 0.05, the Navier-Stokes spatial study's settings (dt =
2e-4, two steps): observed order on 8³/16³/32³ in C++ and 6³/12³/24³ in
Python, within [1.85, 2.15] on the orthogonal meshes and [1.6, 2.3], rising,
on the smooth distortion — whose walls stay planar and axis-aligned while
the cells beside them are not. Python first, then C++, each failing before
its implementation.

### Revision before the result

The first Python run failed the orthogonal half: the velocity error fell at
order 1.05 then 1.46. The same test with the exact velocity prescribed on
the walls instead of slip gives 1.03 then 1.38, and its errors are larger at
every mesh (1.517e-4, 7.415e-5, 2.849e-5 against slip's 1.504e-4, 7.275e-5,
2.645e-5). So the orthogonal half measured the test problem, not the slip
wall. On a uniform mesh the Taylor–Green interior error is exceptionally
small — 1.5e-4 at 6³, against 1.3e-2 on the distortion — and an error in the
wall-adjacent layer, which converges more slowly, dominates it whichever
boundary is used. Longer runs (20 steps of 2e-3) do not settle it either:
slip gives orders 1.71 then 2.94, the exact wall velocity 1.67 then 2.65.

The gate is revised to what it was meant to test, before any C++ code:

1. on the smooth distortion, the observed order in [1.6, 2.3] and
   approaching 2 (ADR-038's revised condition);
2. on both families, at every mesh, the slip wall's error no more than 1.1
   times the error with the exact wall velocity prescribed.

A slip wall with an error of its own would fail the second condition at the
finer meshes. The orthogonal family's convergence on this problem, with
either boundary, is reported and left as an open finding.

### Result

**Passes in both.** On the smooth distortion the slip wall's error is the
exact wall's to three digits at every mesh (ratios 1.000), and its order is
1.780 then 2.045 in C++ (8³/16³/32³) and 1.461 then 1.962 in Python
(6³/12³/24³). On the orthogonal family slip is the more accurate boundary —
ratios 0.990, 0.967, 0.877 in C++ and 0.992, 0.981, 0.928 in Python — and
both boundaries converge as ADR-039's revision found: slip 1.283 then 1.539
in C++, the exact wall 1.250 then 1.398. Python and C++ agree to 9.5e-11 on
all eight Taylor–Green errors the cross-check compares.

The slip wall then carried two benchmarks: the side walls of the
Rayleigh–Bénard roll and the faces of the one-cell slabs (ADR-038). The
orthogonal family's slow convergence stays open; it belongs to the test
problem, not to the boundary.

## ADR-040 — A transient-coupling gate: the Rayleigh–Bénard growth rate against linear theory. Stated before its runs, answered after
**Decided.** ADR-038's onset gate judges only the zero of the growth rate,
which does not depend on how the solver marches. Its runs showed that the
rates themselves do: four fixed outer iterations gave 0.173 at Ra = 1800 on
32 cells where linear theory gives 0.694. No gate measured a transient
coupling against an exact rate, so none would notice a scheme whose steady
states are right and whose dynamics are not. This is that gate.

**Reference.** The leading growth rate of the rigid-plate problem at
k = 3.117, Pr = 1, from a Chebyshev collocation of the linearised equations
(`tests/benchmark/rb_linear.py`) in the D² form of Dongarra, Straughan &
Walker (1996): σ* = 0.693973025 at Ra = 1800, the same to 2e-9 on 24 to 64
points. The direct fourth-order form, tried first, drifts by round-off to
3e-5 at 128 points, which is why it was not kept. The script checks itself
twice. Its neutral Rayleigh number is 1707.7619 against Chandrasekhar's
1707.762. Its slope dσ/dε at onset is 13.000 against 1/τ₀ = 12.999 from the
amplitude equation's τ₀ = (Pr + 0.5117)/(19.65 Pr) for rigid plates
(Bodenschatz et al., arXiv patt-sol/9305001).

**The gate** (`rb_growth`, C++). The onset gate's box, boundaries and
perturbation, at Ra = 1800. Every step's outer loop is iterated to
convergence: outerTol 1e-10 with a cap of 400 iterations, and a step that
reaches the cap fails the gate. The pressure solves go to 1e-12, below
anything the outer test can see.

1. *Space:* the growth rate on 16, 24 and 32 cells across the layer at
   dt = 0.01. The observed order, from the unequal ratios, must be in
   [1.5, 2.6]. The Richardson extrapolation must be within 1% of σ*.
2. *Time:* on 16 cells, the growth rate at dt = 0.04, 0.02 and 0.01 against
   a dt = 0.00125 run. The order of the last pair must be in [1.8, 2.6] —
   BDF2, as the Ethier–Steinman temporal gate asks.
3. The fitted rate's two halves agree within 1e-3, as in the onset gate.

The four-iteration rates are printed beside the converged ones and not
gated: they are the finding this gate exists to keep in view.

**Why 1%.** Close to onset a growth rate is mostly a distance from the
critical point, σ ≈ (dσ/dRa)(Ra − Ra_c). An error of δ in the extrapolated
Ra_c costs δ/(Ra − Ra_c) of σ, and the onset gate extrapolated Ra_c to 0.08,
that is 0.09% of σ at Ra = 1800. The slope carries its own extrapolation
error on top. 0.5% would leave too little room for it, and 2% would pass a
lagging scheme that happened to extrapolate close.

**Data seen before this was written,** during ADR-038's investigation: with
the outer loop converged, 0.9196 on 16 cells (40 iterations, 1e-11) and
0.7429 on 32 cells (100 iterations, 1e-13). No run on 24 cells, no
temporal study, and none at the gate's settings. The bands above come from
the onset gate and from the argument just given, not from those two
numbers — but they were written knowing them, and that is recorded here.

### First run, and a revision before the result

The first run (ae03f73, in the cell form, the only one then) finished two of
the three meshes and was stopped during the third:

    N = 16   0.920868008   +32.7%   halves agree to 8e-9   outer ≤ 191   four fixed: 0.528849
    N = 24   0.793826686   +14.4%   halves agree to 1e-9   outer ≤ 384   four fixed: 0.280965

24 cells had used 384 of the 400 iterations the cap allows. A probe then
marched 20 steps on each mesh with a cap of 2000 (`VIBEFLOW_PROBE_N`): a step
needs at most 190, 384 and 648 outer iterations to reach 1e-10 on 16, 24 and
32 cells, where the diffusion number dt·ν/h² is 2.6, 5.8 and 10.2. So the gate
as written would have failed on 32 cells at its own effort limit, whatever
the accuracy.

**Revision.** The cap is raised to 2000, three times what 32 cells needs.
Nothing else changes: dt = 0.01, outerTol 1e-10, the pressure tolerance, the
bands, and the rule that a step which reaches the cap fails the gate. The cap
of 400 was a guess at the cost, made without measuring it; the criterion it
guards — every step's loop converged — stays as it was.

Known when the revision was made: the two rates above. Assuming order 2,
they alone extrapolate to 0.692194, −0.26% from σ*.

**A correction to ADR-038, and to this entry's data.** The runs listed above
as converged were not. 40 iterations on 16 cells and 100 on 32 stopped at
their caps, short of the 190 and 648 a step needs, so 0.9196 and 0.7429 are
capped rates; the first gate run's 0.920868 is the converged one on 16
cells. ADR-038's finding stands — the rate is right only with the loop
converged — but its 0.743 is the rate at 100 iterations, not the converged
rate. A note there points here.

### Result

**Passes**, in the cell form (the tree of a97008b, one thread, 46 minutes):

    N = 16   0.920868009   +32.7%   outer ≤ 190   four fixed: 0.528849
    N = 24   0.793826684   +14.4%   outer ≤ 384   four fixed: 0.280965
    N = 32   0.749703122    +8.0%   outer ≤ 648   four fixed: 0.172594

The observed order is 2.021 and the Richardson extrapolation 0.693762,
−0.030% from σ*. In time, on 16 cells against dt = 0.00125 (0.920842570),
the errors at dt = 0.04, 0.02 and 0.01 are 4.05e-4, 1.02e-4 and 2.54e-5:
orders 1.986 then 2.008. Every fit's halves agree to 8e-9 or better, and no
step came near the cap: 648 iterations at most, 499 at dt = 0.04, 45 at
dt = 0.00125. The first run's two rates repeat to 2e-9; it ran on two
threads, which sum the reductions in another order.

So the coupled scheme's dynamics converge to linear theory at second order
in space and in time, once each step's outer loop converges. With four fixed
iterations they do not: those rates fall with refinement — 0.53, 0.28, 0.17
— away from σ*, as the diffusion number grows. A converged step at a
diffusion number of 10 costs 648 outer iterations, and that cost is this
loop's, not the gate's: the pressure correction sees only the momentum
diagonal. Faster transients at large diffusion numbers need a better outer
iteration, not fewer of them.

## ADR-041 — Balanced buoyancy: the force in the face flux, its hydrostatic part in a pressure of its own. Stated before the code, answered after
**Decided.** ADR-038's buoyancy is a cell force. A fluid at rest in a
stratification that the reference T_ref does not match starts to move: gate
3's box with T_ref = 0.5 reaches 0.47 κ/L within 50 steps at Ra = 1700 on 8³.
The pressure that should hold it is quadratic, and it is carried by the
least-squares gradient and the boundary extrapolation, neither exact for it.
ADR-038 deferred the remedy that OpenFOAM's buoyantBoussinesqPimpleFoam
uses: put the force in the face flux (phig) and reconstruct the cell
velocity from face quantities, so that a hydrostatic state balances face by
face. Done literally, that routes the solver's own pressure gradient through
the reconstruction too — first order on a perturbed mesh, a change to the
velocity correction every v1 gate stands on, and a bypass of ADR-037's
Rhie–Chow form. This ADR takes the same idea with one split, so that the v1
path stays exactly as it is.

**The form** (`buoyancy = balanced`), each outer iteration, from the latest T:

1. The face force B_f = f(T_f)·S_f, with T_f the skew-corrected face
   temperature, or on a boundary face the value the energy equation's
   gradient uses there.
2. A hydrostatic pressure p_h solves the discrete Poisson problem that makes
   the face residual r_f = B_f − [a_f (p_h,N − p_h,P) + k_f·∇p_h,f]
   divergence-free in every cell. r = 0 on every boundary face (Neumann), and
   the non-orthogonal part is iterated as the pressure's is. In the continuum
   this is the Leray split of the force: f − ∇p_h is divergence-free and has
   no normal component on the walls.
3. The cell force g_P = M_P⁻¹ Σ_f S_f r_f / |S_f|, with M_P = Σ_f S_f S_fᵀ/|S_f|
   (OpenFOAM's reconstruct), replaces the cell buoyancy in the momentum
   source.
4. In the predicted face flux the compact residual replaces the interpolated
   cell force: F* += D_f (r_f − L[g]·S_f), the buoyancy's counterpart of the
   pressure's D_f (L[∇p]·S − a_f Δp). ADR-037's old-flux term keeps it free of
   dt at a steady state, by the same algebra.

The solved pressure is then the dynamic part. p_h, plus T_ref's analytic
hydrostatic part, completes it.

**Why it is exact at rest.** Take T depending on height alone, on a mesh
whose cells stand in horizontal layers: Cartesian, and the smooth family,
which distorts the horizontal plane and extrudes along z. The force has no
horizontal component, so B vanishes on side faces. The horizontal faces of
stacked cells are orthogonal, so their differences of p_h absorb B exactly,
and every column gives the same differences. The least-squares gradient of
a layered field then has no horizontal component, so the non-orthogonal
correction on side faces vanishes too. Hence r = 0 on every face, the cell
force is exactly zero, and u = 0 is a fixed point for any T(z), matched
reference or not. On a randomly perturbed mesh the layers are gone and it is
not exact.

**Measured before this was written.** Reconstructing a smooth field with no
normal component on the walls from exact face values: second order on the
smooth family (orders 1.99, 2.00 orthogonal; 1.97, 1.99 distorted; 6³–24³)
and first order on the randomly perturbed one (1.02, 0.90). The order gates
run on the smooth family.

**Gates.** Python first, then C++. Each is committed failing (the form does
not exist) before its code:

1. *Rest, constant reference.* Gate 3's box with T_ref = 0.5 constant and
   T = 1 − z. After 50 steps, max|u| ≤ 1e-12 and max|T − (1 − z)| ≤ 1e-12, on
   the Cartesian and the distorted 8³ mesh.
2. *Rest, curved stratification.* A uniformly heated layer, T = 1 − z² with
   source 2κ, the same boundaries and T_ref = 0.5. max|u| ≤ 1e-12, and T
   horizontally uniform to 1e-12. The discrete conduction profile is not
   exactly 1 − z², so its shape is not judged, only its evenness.
3. *Accuracy unchanged.* ADR-038's gates 1–3, both benchmarks and ADR-040's
   transient gate, rerun in the balanced form against their own criteria.
4. *Cross-check and MPI in the balanced form.* Gate 2's steady rows, Python
   against C++, within 1e-6; two to four ranks within 1e-10 of the serial run.
5. *v1 untouched.* With buoyancy off the operations are the same; the full
   suite runs.

**Decision rule.** If every gate passes, the balanced form becomes the
default for buoyant runs and the cell form stays selectable as the recorded
baseline. If gate 3 fails on the distortion, the balanced form stays an
option and this entry says why. Reported, not gated: the rest state on the
randomly perturbed fixture mesh, in both forms.

**Cost.** One more Poisson solve per outer iteration, with a constant matrix
and a warm start.

### Results

**Every gate passes, and the balanced form becomes the default.**

1. *Rest, constant reference* (the harnesses' gate 4): max|u| 1.0e-14 and
   4.7e-15 in C++ (Cartesian, distorted), 2.2e-14 and 7.8e-15 in Python,
   where the cell form reaches 0.47 and 0.48. T stays at 1 − z to 2e-15.
2. *Rest, curved stratification* (gate 5): max|u| 1.3e-14 and 1.2e-14 in
   C++, 2.0e-14 and 1.2e-14 in Python; T even within each layer to 4e-15.
3. *Accuracy unchanged.* The temperature gate has no buoyancy and repeats to
   every digit. The steady Boussinesq orders are the cell form's to three
   digits — C++ u 2.021 / 2.013, T 2.003 / 2.024; Python u 2.044 / 2.030,
   T 2.005 / 2.038 — while the errors themselves move by 0.03–0.12%, the two
   forms differing at O(h²); dt = 0.2 against 2.0 still agree to 2.8e-12
   (u) and 1.0e-10 (T). Gate 3 holds at 1.1e-14 / 3.9e-15. Both benchmarks
   and the transient gate pass:

   | | cell form | balanced form |
   | --- | --- | --- |
   | Rayleigh–Bénard onset, 16 / 24 / 32 cells | 1678.630 / 1694.884 / 1700.563 | 1696.327 / 1702.737 / 1704.959 |
   | observed order, extrapolated Ra_c | 2.005, 1707.842 (+0.005%) | 2.026, 1707.768 (+0.000%) |
   | growth rate at Ra 1800, 16 / 24 / 32 cells | +32.7% / +14.4% / +8.0% | +12.4% / +5.5% / +3.0% |
   | observed order, extrapolation | 2.021, −0.030% | 2.026, −0.007% |
   | temporal orders | 1.986, 2.008 | 1.989, 2.009 |
   | de Vahl Davis cavity, extrapolated Nu, Ra 10³ / 10⁴ / 10⁵ / 10⁶ | 1.11779 / 2.24482 / 4.52161 / 8.81945 | 1.11779 / 2.24481 / 4.52149 / 8.81858 |
   | cavity, 128², largest departure of a velocity maximum | +0.84% | +0.84% |

   On the Rayleigh–Bénard problems the balanced form is the more accurate on
   every mesh — on the finest onset mesh it is 0.16% low, the cell form
   0.42%. On the cavity the two agree to 1e-4 in Nu, and the balanced
   form's largest departure of an extrapolated Nu is −0.075% (Ra = 10⁶,
   observed order 1.87), against 0.5% allowed.
4. *Cross-check and MPI in the balanced form.* Python against C++, both
   balanced: 16 rows, worst 9.5e-11, the steady Boussinesq rows 5.2e-11
   (bound 1e-6). Two to four ranks within 4.8e-15 (u) and 2.5e-15 (T) of the
   serial run (bound 1e-10). The serial answer is the balanced one:
   5.95961341332706e-03 where the cell form gave 5.96211681754360e-03.
5. *v1 untouched.* The full suite, `run_gates.py v0 v1 v2 v2b`, run with
   the balanced form the default: all 37 gates it ran pass, in 5 h 48 min
   on one core; the cavity above and ADR-042's flat plate ran on their own.
   It ran on the tree of 648c7ad with this default switched; what was
   committed after it changes no arithmetic outside the flat plate's
   harness (ADR-042's two diagnostic switches are off by default).

Reported, not gated: on the randomly perturbed fixture mesh, where the
layers are gone, the resting fluid with T_ref = 0.5 reaches 0.037 in the
balanced form against 0.50 in the cell form, in both codes.

**Decision.** By the rule above the balanced form is now the default for
buoyant runs — `EnergyModel.form = BuoyancyForm::Balanced` in C++,
`buoyancy_form="balanced"` in Python — and the cell form stays selectable
(`VIBEFLOW_BUOYANCY=cell` in every harness) as the recorded baseline.

**Cost, measured.** Gate 2 of the C++ heat-transfer gates on 16³ plus its
dt check, one thread: 218 s in the cell form, 263 s balanced (+21%). The
cavity at Ra = 10⁴ on 32² and 64², one thread, under the same load: 84 s
against 144 s (+71%). The p_h solve is Jacobi-preconditioned CG to 1e-15
of its right-hand side, whatever backend the pressure uses, and beside the
cavity's BoomerAMG pressure it is the dearer of the two solves; the whole
cavity gate took 243 minutes balanced, on one shared core, where ADR-038's
cell-form run took 71 on two. Giving p_h the pressure's backend is left as
a follow-up; it changes no answer.

## ADR-042 — v2b: Menter's k-ω SST model, integrated to the wall; its gates, stated before the code, answered after
**Decided.** v2b adds the k-ω SST model with low-Reynolds wall treatment:
the equations are integrated to the wall, which the mesh resolves to y⁺ ≈ 1,
and ω takes Menter's wall value. Wall functions are v2c's. This entry fixes
the model, the numerics, and the gates up to the flat plate. The
backward-facing step gets an entry of its own, written after the flat plate
passes and before any step is run: its grids, inflow and upper wall need
decisions this entry cannot yet make well.

### The model

Two variants, selectable, one code path:

| | SST-2003 (default) | SST-1994 |
| --- | --- | --- |
| eddy viscosity | ν_t = a₁k / max(a₁ω, S F₂) | ν_t = a₁k / max(a₁ω, Ω F₂) |
| production in the k equation | P̃ = min(P, 10 β*ωk) | P̃ = min(P, 20 β*ωk) |
| production in the ω equation | γ P̃ / ν_t | γ P / ν_t |
| floor of CD_kω | 1e-10 | 1e-20 |
| γ₁, γ₂ | 5/9, 0.44 | β_i/β* − σ_ωi κ²/√β* |

as the NASA Turbulence Modeling Resource (TMR) gives SST-2003 (Menter,
Kuntz & Langtry 2003) and SST (Menter 1994). Common to both:

    ∂k/∂t + ∇·(uk) = P̃ − β*ωk + ∇·[(ν + σ_k ν_t)∇k]
    ∂ω/∂t + ∇·(uω) = γ P̃(or P)/ν_t − βω² + ∇·[(ν + σ_ω ν_t)∇ω]
                     + 2(1 − F₁) σ_ω2 ∇k·∇ω / ω

with P = ν_t S², S = √(2 S_ij S_ij), Ω = √(2 W_ij W_ij); each of σ_k, σ_ω,
β, γ blended as φ = F₁φ₁ + (1 − F₁)φ₂; F₁ = tanh(arg₁⁴),
arg₁ = min[max(√k/(β*ωd), 500ν/(d²ω)), 4σ_ω2 k/(CD_kω d²)],
CD_kω = max(2σ_ω2 ∇k·∇ω/ω, floor); F₂ = tanh(arg₂²),
arg₂ = max(2√k/(β*ωd), 500ν/(d²ω)); σ_k1 0.85, σ_ω1 0.5, β₁ 0.075,
σ_k2 1.0, σ_ω2 0.856, β₂ 0.0828, β* 0.09, κ 0.41, a₁ 0.31. The flow is
incompressible, so P = τ_ij ∂u_i/∂x_j is exactly ν_t S², and the isotropic
part of the Reynolds stress, (2/3)k, is carried by the pressure: the forms
TMR marks "m" are here the exact ones. The TMR benchmarks were run in the
1994 variant, which is why it is kept.

SST-V, TMR's flat-plate variant, produces with Ω² instead of S². In a thin
shear layer the two agree to order (δ/L)², and TMR reports the two
variants' flat-plate results as nearly identical. It is not implemented,
and the flat-plate comparison below is made knowing that.

### The numerics

- **k and ω** are transported as the temperature is (ADR-038): BDF2, upwind
  in the matrix plus the deferred correction to the skew-corrected face
  value, diffusion with the non-orthogonal correction, the face diffusivity
  ν + σν_t interpolated linearly. Both are solved in every outer iteration
  after the pressure correctors (and the energy equation), then ν_t is
  updated, so a converged outer loop carries no lag. Destruction is
  implicit (β*ω and βω on the diagonal), production explicit, the cross
  diffusion explicit where positive and on the diagonal where negative.
  After each solve, k and ω are bounded below by positive floors far below
  anything the gates' solutions reach; the step report counts the cells
  bounded, and in the manufactured gates that count must be zero.
- **Momentum** gets a variable viscosity: ν + ν_t,f on every face, in the
  matrix and in the non-orthogonal correction, and the part of
  ∇·[ν_t(∇u + ∇uᵀ)] that a constant viscosity does not have, ∇·(ν_t ∇uᵀ),
  explicit from the latest velocity gradient. With the model off every
  operation is v2a's.
- **Wall distance** d: the exact distance to the nearest wall face, each face
  split into triangles, by brute force over all wall faces, which every
  rank gathers. Once per mesh.
- **Boundaries.** A wall is a no-slip face flagged as one: k = 0 and
  ω = 10 · 6ν/(β₁ d₁²) there, d₁ the wall distance of the adjacent cell
  centre — Menter's value, TMR's "distance to the next point away from the
  wall" read for a cell-centred scheme. An inlet prescribes k and ω; an
  outlet, a slip face and a symmetry plane take zero gradient.

### Gates

Python first, then C++, each committed failing before its code.

**1. Wall distance.** On all three box families with walls at y = 0 and
y = 1 (their boundaries stay planar), d = min(y, 1 − y) at every cell centre
to 1e-13. On the flat-plate grids below (C++), d = y above the plate and
√(x² + y²) ahead of it, to 1e-13. The same d on two to four ranks.

**2. Manufactured solutions for the model equations.** The unit cube, one
wall at y = 0, so d = y exactly; exact Dirichlet values of every field on
the other five faces; sources from sympy applied to the same formulas the
solver evaluates; both variants. Two solutions, because the SST limiter
cannot switch on and off inside the domain without a kink in ν_t that no
scheme converges through at second order (Eça et al., IST report D72-34):

- *MS-A — blending on, limiter off.* u = U₀ (sin πx (cos πy − cos πz),
  sin πy (cos πz − cos πx), sin πz (cos πx − cos πy)) with U₀ = 0.1,
  p = 0.01 cos πx cos πy cos πz, k = 0.1 (0.5 + y)(1 + 0.3 sin πx cos πz),
  ω = 5 (1 + 0.05/(y + 0.2)²)(1 + 0.2 cos πx sin πz), ν = 5e-5. Sampled on
  41³ points: F₁ spans [0.23, 1]; wherever F₁ is not saturated (arg₁ < 2) the
  first branch of arg₁ is the one active, at least 17 times the second and
  8 times the third; a₁ω ≥ 1.23 S F₂ and 1.52 Ω F₂; P ≤ 0.07 of its limit.
- *MS-B — limiter on.* u = (y + y², 0, 0) plus 0.05 times the same
  solenoidal field, so that S and Ω stay away from zero;
  k = 0.1 (1 − 0.5y)(1 + 0.3 sin πx cos πz),
  ω = (1 + 2y)(1 + 0.2 cos πx sin πz), ν = 5e-5. F₁ spans [0.25, 1] and F₂
  [0.96, 1], each on its first branch wherever not saturated;
  S F₂ ≥ 2.61 a₁ω and Ω F₂ ≥ 2.56 a₁ω; P ≤ 0.45 of its limit.

Where arg₁ ≥ 2, F₁ = 1 to 1e-14, so a switch there does not show; the same
for F₂ at arg₂ ≥ 4. The harness checks the properties on the exact fields
at the cell centres of each mesh before it runs — the first branches active
wherever unsaturated, by at least 5 times the others; the limiter off (or
on) by at least 1.1; P below 0.9 of its limit — and prints the margins.
`prototype/sst_ms.py` holds the formulas once and gives Python its sources
and C++ a generated header, so the two sides add the same terms.

  2a. *Frozen velocity*, Eça's first exercise: u and the face flux are
      exact, k and ω are solved. MS-A and MS-B, both variants.
  2b. *Coupled*: u, p, k and ω solved together, the momentum source
      including the variable-viscosity stress. MS-A, both variants.

  Orders of k and ω (and u in 2b) in the v1 and v2a bands: [1.85, 2.15] on
  the orthogonal meshes, [1.6, 2.3] and approaching 2 on the smooth
  distortion; 8³/16³/32³ in C++, 6³/12³/24³ in Python. Steady as ADR-038's
  gate 2 is: marched until the largest change per step of u, k and ω,
  each relative to its largest value, is below 1e-8 in C++ (the floor the
  non-orthogonal pressure loop leaves on the distortion) and 1e-12 in
  Python.

  *Not reached by these solutions:* the production limiter, arg₁'s third
  branch and the floor of CD_kω, and ω's wall value. The flat plate is where
  they act.

**3. Cross-check and MPI.** Gate 2's rows on the meshes both sides run,
Python against C++, within 1e-6, both converged to 1e-12 for it. Gate 2b on
two to four ranks within 1e-10 of the serial run.

**4. Zero-pressure-gradient flat plate**, TMR's 2DZP verification case, in
C++. Its grid family extruded one cell in z with slip faces: 69×49, 137×97
and 273×193 points, and 545×385 if the cost allows. Re = 5e6 per unit
length, U∞ = 1; an inlet at x = −1/3 with u = (1, 0, 0), k∞ = 2.25e-7 and
ω∞ = 125 (TMR's values, ν_t∞/ν = 0.009); symmetry on y = 0 ahead of the
plate and the wall on 0 ≤ x ≤ 2; a pressure outlet p = 0 at x = 2 and on the
top, y = 1, so that the displacement flow leaves (a slip top would speed the
free stream up by the displacement thickness over the height, about 0.25%,
and Cf with it). SST-1994, marched to a steady state: Cf at x = 0.97008 and
the peak of ν_t there changing by less than 1e-6 relative over the last
tenth of the march, and every field's change per step below 1e-6 of its
size. Against TMR's SST-V results:

  4a. Cf at x = 0.97008, interpolated along the wall faces: monotone over
      the three finest grids run, observed order in [0.8, 3.0], and the
      Richardson extrapolation within 1% of 0.0026964 — the mean of TMR's
      own extrapolations, 0.00269681 (CFL3D) and 0.00269607 (FUN3D), from
      137×97 to 545×385 at orders 1.21 and 1.39.
  4b. u⁺ against y⁺ at x = 0.97008 on the finest grid run, within 2% of
      TMR's CFL3D profile (545×385) for 1 ≤ y⁺ ≤ 500.
  4c. The log law u⁺ = ln(y⁺)/0.41 + 5.0 within 3% for 60 ≤ y⁺ ≤ 250.
  4d. The peak of ν_t/ν across the layer at x = 0.97008 within 2% of 221.7
      (CFL3D 221.4, FUN3D 221.9, on 545×385).

  SST-2003 is run on the finest grid too, reported and not gated.

**Why these bands.** The references are compressible, at M = 0.2. An
incompressible solution of the same model should lie higher in Cf by 0.2 to
0.3% on a van Driest II estimate (adiabatic wall, recovery factor 0.89) — an
estimate, not a measurement. TMR's two codes agree to 0.03% extrapolated, so
1% leaves this solver about 0.7% for its own extrapolation error. u⁺ at
fixed y⁺ carries about half of a Cf error through u_τ, and the reference
profile its own finest-grid error; 2% is twice 4a's band. TMR's SST profile
itself departs from the log law by −1.5% to +1.3% over 60 ≤ y⁺ ≤ 250, so 3%
passes it with room and fails a profile whose log layer is wrong. The ν_t
peak: ten times the two codes' spread, for this solver's discretisation
error on grids up to 273×193.

**5. v1 and v2a untouched.** With the model off the operations are the
same; the full suite runs.

**Data seen before this was written:** TMR's published SST files — the Cf
convergence tables, the u⁺ profiles at x = 0.97008 and 1.90334, the ν_t
profile at 0.97 and the Cf distributions — from which the numbers above
come; its grid files; and an exploration of candidate manufactured fields
by their exact values alone, no solver run, from which MS-A and MS-B were
chosen. No run of this solver with a turbulence model exists.

**Cost.** Two scalar solves per outer iteration and, on the flat plate, a
steady march on up to 52k cells, or 209k with 545×385.

### Revision before the result: the manufactured solutions

The first runs of gate 2a — Python, frozen velocity, SST-2003, orthogonal
meshes, the implementation as it now stands — found both solutions unfit
for these meshes:

- *MS-A:* k converged at orders 2.00 and 2.10, ω at 1.04 and 1.36 (errors
  9.9e-2, 4.8e-2, 1.9e-2), nearly all of it in the layer of cells on y = 0,
  where ω = 5(1 + 0.05/(y + 0.2)²) curves by ω'' ≈ 940. There the one-sided
  boundary flux leaves an O(1) truncation error in the first cell. Diffusion
  suppresses such an error to O(h²) only once a cell is smaller than the
  reaction length √(Γ/2βω), about 0.04 here — finer than 24³.
- *MS-B:* on 6³ k and ω drifted to the floor, bounded in thousands of
  cell-steps; on 12³ and 24³ ω converged at order 4.0, far from asymptotic.
  With the limiter on, the k equation's production beats its destruction —
  P/ε = a₁S/(β*ωF₂) ≥ a₁²/β* > 1 always, here about 3 — and only the
  manufactured sink holds the balance, an equilibrium that lasts while
  diffusion outweighs the net growth. On 6³ it did not.

One change to the solver came out of this: an imposed source enters by
Patankar's rule, its negative part on the diagonal as −s/φ, so that a sink
cannot drive φ negative; the steady state is the same. It slowed MS-B's
drift on 6³ and did not stop it — the solution was the cause.

Both solutions are revised, before any verdict, keeping their purpose:

- *MS-A:* k = 0.1 (0.5 + y)(1 + 0.3 sin πx sin πz),
  ω = 5 (1.5 − 0.5y)(1 + 0.2 sin πx sin πz): no steep wall term, and — sin
  in x and z, linear in y — zero normal curvature on every face, so the
  boundary flux carries no O(1) error. Velocity, pressure and ν unchanged.
  F₁ spans [0.42, 1]; the first branch of arg₁ is active wherever F₁ is
  unsaturated, 26 times the second and 9.7 times the third;
  a₁ω ≥ 1.25 S F₂ and 2.11 Ω F₂; P ≤ 0.07 of its limit.
- *MS-B:* u = 0.3 (0.5 + y, 0, 0) plus 0.015 times the solenoidal field — a
  linear shear, whose least-squares gradient is exact, so S carries no
  boundary error of its own — k = 0.5 (1 − 0.3y)(1 + 0.3 sin πx sin πz),
  ω = 0.24 (1 + y)(1 + 0.2 sin πx sin πz), p = 0, ν = 5e-5. S F₂ ≥ 1.69 a₁ω
  and Ω F₂ ≥ 1.66 a₁ω; P ≤ 0.50 of its limit; ν_t ≈ 0.1 makes the diffusion
  rate several times the net production. F₁ = F₂ = 1 throughout: MS-B now
  tests the limiter and MS-A the blending, and F₂ joins the list of what
  these solutions do not reach, for the flat plate.

Data seen in choosing them, all SST-2003, orthogonal, frozen velocity: the
revised MS-A at orders 2.13, 2.03 (k) and 2.11, 2.04 (ω); the revised MS-B
at 1.91, 1.97 and 1.93, 1.98. Five other candidates for MS-B were tried and
dropped: two kept the original flow with smoother k and ω and still failed
on 6³; two converged well above second order on 12³/24³ (ω at 2.67 and
2.13 on the last pair, after 3.30 and 2.49), one of them also losing its
branch margins on the finer meshes; and the same flow slowed to U₀ = 0.3
with its quadratic shear brought k to 1.74 then 1.89, from below, near the
band's edge — the linear shear removes the boundary error of S that held
it there. Not yet run: the distortion, SST-1994, and gate 2b. The criteria
are unchanged.

### Revision before the flat plate's result: k and ω advection, ω's destruction, and the march

A first flat-plate run on TMR's coarsest grid, 35×25 — not one of the
gate's grids — went wrong in three ways, each fixed before any gate grid
was run:

- *Central differencing of ω across the leading edge.* ω jumps by five
  orders of magnitude between the free stream (125) and the first cell on
  the plate (about 1e7), and at the leading edge that jump lies between two
  neighbouring cells. The deferred correction then carries the downstream
  value into the upstream cell's outflow, which drove that cell's ω
  negative on the second step; 1.4 million cell-steps were bounded and Cf
  settled at 3.1e-4, a tenth of the turbulent value. TMR's two codes use
  first-order upwinding for the turbulence advection ("Both codes used
  first-order upwinding for the advective terms of the turbulence model").
  So does this solver now, by default: `TurbulenceModel.secondOrderAdvection
  = false`, `advection="upwind"` in Python. The deferred correction stays
  as the option the manufactured gates run — they verify the model's
  terms, whose order a first-order advection error would hide — and the
  cross-check and MPI gates run it too.
- *ω's destruction linearised by Picard.* βω² as βω_old·ω makes its
  balance with the production a period-two map, ω_new·ω_old = P/β, which
  neither converges nor damps; on 69×49 the first cell on the plate
  swung between 6e5 and 2.6e7 and then grew without bound. It is now
  linearised by Newton, 2βω_old·ω − βω_old², the same fixed point.
- *The impulsive start.* From a uniform stream the first cell above the
  wall sees S ≈ 1e5 at once, and on 35×25 a full step there drove k and ω
  through zero on the first steps and kept them there. The march now ramps the
  step from dt/64, doubling every 100 steps (`PisoSolver::setTimeStep`,
  which restarts BDF at first order); the steady state does not depend on
  the step (ADR-037). The first few steps still bound k in the wall cells,
  where it falls by more than the factor of four BDF2 can follow; the
  count stops rising after them.

Data seen: on 35×25, with upwinding and the ramp (still Picard), the plate
reached a steady state at t = 40 (4,500 steps) with Cf(0.97008) =
0.0025569, CD = 0.0027222 and a ν_t/ν peak of 208.6, where TMR's CFL3D gives
0.0025518 and 0.0027062 on the same grid; 112 cells bounded, all in the
first steps. On 69×49, one of the gate's grids, the runs that showed the
Picard oscillation were stopped at steps 300, 30 and 60, blown up; nothing
was taken from them but that. With Newton, 69×49's first 60 steps kept ω
bounded and smooth. The criteria are unchanged, and gate 2's runs are
repeated with the Newton linearisation, which changes their iteration but
not their fixed point.

### Revision before the flat plate's result: momentum advection

The 69×49 run then reached its steady state — Cf(0.97008) = 0.0026427,
CD = 0.0028133, a ν_t/ν peak of 214.5, after 5,800 steps (t = 53) — and the
slowness of the march led to what was wrong with it. Ahead of the plate,
within about 2e-3 of the symmetry plane, the velocity alternates from one
column of cells to the next. On 35×25 at its steady state, the first row of
cells has u = 0.82, 1.13, 0.90, 1.08, 0.93, 1.10 in the six columns between
the inlet and the leading edge, while the pressure runs smoothly through
them; the shear this makes holds k at up to 1.9e-3 there, four orders of
magnitude above the free stream's, and ν_t/ν at up to 67, where the free
stream has 0.009. It is central differencing's odd–even mode, excited at
the leading edge, where u falls from 1 to 0.06 between two columns.
Momentum's face value is linear interpolation plus the skewness
correction, which does not see that mode, and nothing else here damps it:
these cells are about 0.1 long and 8e-6 thick, a cell Péclet number in x
above 1e4, so there is no streamwise diffusion to speak of. With
first-order upwinding (the deferred correction switched off, as a
diagnostic), the mode is gone: u is within 0.4% of 1 ahead of the plate,
k and ν_t are at their free-stream values, and 35×25 reaches its steady
state in 1,200 steps instead of 4,500 (Cf 0.0025382, first order).

So momentum gets a second choice of face value,
`PisoControls.convection = ConvectionScheme::LinearUpwind`: the upwind
cell's value extrapolated to the face centre by its own least-squares
gradient, u_U + ∇u_U · (x_f − x_U), still carried as the deferred
correction to upwind. It is exact for a linear field, so second order, and
it treats the odd–even mode as upwinding does (on a uniform grid it is
Fromm's scheme). CFL3D's and FUN3D's schemes are upwind-biased as well. The
default stays `Linear`, so every other case and gate is unchanged; the flat
plate runs `LinearUpwind`, the first case with a free stream meeting a wall
edge-on across cells this long. k and ω keep first-order upwinding.

Before the flat plate runs again, the new face value is verified as the old
one was, against the same criteria:

- Ethier–Steinman, the v1 C++ gate, with `LinearUpwind`: spatial order in
  [1.85, 2.15] on the orthogonal meshes and [1.6, 2.3] on the smooth
  distortion.
- The Navier–Stokes cross-check with `convection="linearUpwind"` in Python
  too: C++ against Python within the same 1e-4.

Ethier–Steinman then ran first, and passed (orders 1.985, 1.991; on the
distortion 1.805, 2.034), but it cannot tell the two face values apart: two
steps from the exact field, its error is mostly the projection's, and the
linear and linear-upwind errors differ by 1.5e-4 relative. So a third
check is added, before it runs: `convection_order`, the steady
manufactured flow of `steady_dt` at ν = 0.02 — cell Péclet numbers of 17,
8 and 4 on 6³, 12³ and 24³, so that convection's truncation error is a
large part of the whole — marched from the exact field to a steady state
(the change of u or p over a step below 1e-8) at a Courant number of 3.2;
both face values, both mesh families; the order of u between the two
finest meshes in the v1 bands, rising on the distortion.

Gate 4's criteria are unchanged. The 69×49 result above is the linear
scheme's and does not count for the gate: 69×49 is run again with the rest.

*Revised after its first run: the added check.* At ν = 0.02 it failed by
its own criteria for both face values, the long-verified linear one too.
L2(u) on 6³, 12³, 24³, and the orders:

| | orthogonal | smooth distortion |
| --- | --- | --- |
| linear | 8.755e-2, 1.782e-2, 4.037e-3 — 2.296, 2.143 | 9.113e-2, 1.887e-2, 4.308e-3 — 2.272, 2.131 |
| linear upwind | 4.966e-2, 8.189e-3, 1.602e-3 — 2.600, 2.353 | 5.293e-2, 9.487e-3, 1.960e-3 — 2.480, 2.275 |

The linear-upwind orthogonal order is above its band, and on the
distortion both orders fall towards 2 where the harness asked them to
rise — v1's "rising", where ADR-042's own bands, and ADR-038's, say
"approaching 2". None of it looks like a defect: every error falls
faster than second order and the orders come down towards 2 from above,
as they do when the meshes are short of the asymptotic range, and a fit
e = a h² + b h³ to the two finest linear-upwind errors (a = 0.67,
b = 6.2, against 2.08 and 5.8 for the linear value) reproduces the
coarsest within 6%: the linear-upwind leading term is a third of the
linear one's here, so the next term holds its order up longer. The
linear-upwind error is 1.7 to 2.5 times smaller on every mesh.

So the check is revised, with the data above seen: it runs at ν = 0.1,
`steady_dt`'s viscosity and the v2a steady gate's (cell Péclet numbers
3.3, 1.7 and 0.8), where the v1 and v2a gates show the linear face value
asymptotic on these meshes; the distortion's condition is ADR-038's,
approaching 2; and a condition is added that the check sees the face value
at all — on every mesh the two face values' errors at least 5% apart,
which Ethier–Steinman's are not. ν = 0.02 stays in the program
(`--nu 0.02`), reported.

**545×385 is run.** The gate left it to the cost. Measured on the runs so
far — 69×49 steady in 1,100 steps (72 s), 137×97 in 1,300 (559 s), 273×193
at about 3 s a step on one shared core — 545×385 is a few hours on two
threads, so it is run, and gate 4 is judged on 137×97, 273×193 and
545×385, as TMR's own extrapolations are; 69×49 is reported. Decided with
273×193 at step 300 of its march, before its result.

Then, with 273×193 steady in Cf and the ν_t peak but its march still
settling at the leading edge, and before any 545×385 step: the march there
needs more steps than 137×97's (1,300), and a step costs about 4 s alone
on 273×193, so 545×385 is a matter of many hours a run. So its step is
0.01, as on every other grid, not the 0.005 the gate script first set as a
precaution; and SST-2003, reported and not gated, is run on 273×193, a
grid short of the finest, instead of a second 545×385 march.

And then, with 545×385 at step 250 and SST-2003 at step 1050, both stopped
and restarted: at the full step, a step's cost was mostly the momentum, k
and ω solves — native BiCGStab with Jacobi, 79 iterations a solve on
137×97, where the cells' aspect ratio reaches 1e4 — and 545×385 was going
at 12 s a step. Those systems are now solved by PETSc's BiCGStab with
ILU(0) (`VIBEFLOW_MOMENTUM=bicgstab+ilu`) on both runs. On 137×97, 800
steps with either solver, and with BoomerAMG, give Cf, CD and the ν_t
peak equal to ten digits; the linear solves take 8 s instead of 189, the
whole march 112 s instead of 289. 69×49, 137×97 and 273×193 in SST-1994
stand as run, with the native solver.

*545×385 from 273×193's steady state.* Restarted so, 545×385 blew up at its
260th step, at dt = 6.25e-4 in the ramp, as the layer near the leading
edge went turbulent: ω in the wall cells at x ≈ 0.04 ran away from its
wall value (2e8) to 5e10 within ten steps and to 1e46 by step 300, k
having dropped to its floor where ν_t had peaked; the native-solver run,
stopped between steps 250 and 300, was on the same path (the same Cf to
seven digits at step 250, and the same 59% change of u at x = 0.034 near
the wall). The
coarser grids went through that transient; 545×385, whose wall cells are
half as thick, did not. So it starts instead from 273×193's steady state,
each cell taking the state of the coarse cell that contains it
(`VIBEFLOW_FP_INIT`, with `VIBEFLOW_FP_SAVE` writing the state), and
marches from there with the same ramp to the same steady criterion: grid
sequencing, as the heated cavity does. Checked on the grids below it:
137×97 started from 69×49's state reaches Cf, CD and the ν_t peak within
1.2e-9, 1.8e-8 and 4e-8 of its uniform start, in 1,000 steps instead of
1,300. 273×193 is marched again with the ILU solver to provide the state;
its result is reported beside the native run's. Gate 4's criteria are
unchanged.

*Four PISO correctors.* Started so, 545×385 blew up again, by step 250,
and it was not the start. Between two dumps a few steps apart the growing
part is a streamwise odd–even mode of the pressure and the velocity, the
same in every row across the boundary layer (0 < y < 3e-4) and wrapped in
an envelope some fifty cells long at 0.05 < x < 0.11; the pressure's
share reaches 0.2 of the dynamic head before k and ω follow it. It grows
at every step tried from 3e-4 up, and at dt = 0.01 from the sequenced
start within 50 steps. Two outer iterations a step make it grow sooner,
not later. Tried at dt = 0.01, 50 steps each: without the old-flux term,
with the pressure solved to 1e-14, with first-order momentum advection,
and with k, ω and ν_t frozen, it still blows up — the turbulence model is
not part of it; with four PISO correctors instead of two, it does not
(Cf 0.0027175, ν_t/ν peak 222.0 at step 50). So the flat plate now runs
four correctors (`VIBEFLOW_CORRECTORS`), and 545×385 is marched with them
from 273×193's state, with the ramp. The steady state does not depend on
the count: 137×97 with four reaches Cf, CD and the ν_t peak equal to ten
digits to its run with two, in the same 1,300 steps. What in the
two-corrector splitting feeds the mode, on this grid and not on the next
coarser one, is not found; it goes to the known limits. The solver's two
new diagnostic switches, `TurbulenceModel.frozen` and `transposeStress`,
stay in the tree for that.

### Results

**Every gate passes.** The flat plate converges towards TMR's extrapolated
skin friction from above, within 0.9% of it, and its profile in wall units
is TMR's to 0.1%.

1. *Wall distance.* Exact to round-off: 0 at every cell in C++ and at most
   1.1e-16 on a boundary face in Python, on all three box families; on the
   four flat-plate grids, at most 4.3e-19. The MPI gate below carries the
   gathered wall faces too: every blending function reads d.
2. *Manufactured solutions*, with the numerics as revised above (Newton for
   ω's destruction; the deferred correction the gates run for k and ω).
   Every one of the 28 order checks passes on each side. The orders between
   the two finest meshes:

   | | C++, 8³–32³ | Python, 6³–24³ |
   | --- | --- | --- |
   | k | 1.966 – 2.019 | 1.950 – 2.034 |
   | ω | 1.954 – 2.027 | 1.935 – 2.045 |
   | u (2b) | 1.971 – 1.978 | 1.954 – 1.965 |

   No cell was bounded in any run; the distortion's orders approach 2
   throughout. The two variants' errors differ by 0.01–0.2% on MS-A, where
   the limiter is off and they nearly coincide, and by 40–120% on MS-B,
   where it is on: each variant's own terms are exercised.
3. *Cross-check and MPI.* Python against C++, 44 rows: worst 4.0e-11
   (bound 1e-6). Two to four ranks within 2.8e-14 of the serial run (bound
   1e-10).

   *The momentum face value* (the revision above), at ν = 0.1: orders
   2.200, 2.044 (linear) and 2.089, 1.929 (linear upwind) on the
   orthogonal family, 2.160, 2.031 and 2.046, 1.937 on the distortion,
   approaching 2; the linear-upwind error 0.55 to 0.65 of the linear one
   on every mesh. Ethier–Steinman with linear upwind: 1.985, 1.991 and
   1.805, 2.034, as with the linear value. The NS cross-check agrees to
   5.4e-10 with linear upwind on both sides (5.1e-10 linear).
4. *Flat plate*, SST-1994, every run steady (the change of u,
   k and ω per 50 steps below 1e-6 of their size):

   | grid | Cf(0.97008) | against CFL3D, same grid | CD | ν_t/ν peak | steps |
   | --- | --- | --- | --- | --- | --- |
   | 69×49 (reported) | 0.0026440 | +0.68% | 0.0027995 | 212.8 | 1,100 |
   | 137×97 | 0.0026855 | +0.78% | 0.0028456 | 218.3 | 1,300 |
   | 273×193 | 0.0027051 | +0.83% | 0.0028679 | 221.7 | 1,950 |
   | 545×385 | 0.0027137 | +0.85% | 0.0028775 | 222.6 | 950, from 273×193 |

   - 4a: monotone, observed order 1.203, Richardson extrapolation
     0.0027203, +0.885% from 0.0026964 (within 1%). TMR's own orders are
     1.21 and 1.39.
   - 4b: u⁺ within 0.10% of TMR's CFL3D profile for 1 ≤ y⁺ ≤ 500, 180
     points (within 2%).
   - 4c: the log law within 1.48% for 60 ≤ y⁺ ≤ 250, 47 cells (within 3%).
   - 4d: the ν_t/ν peak 222.56, +0.39% from 221.7 (within 2%).
   - Gate 1 on these grids: |d − d_exact| at most 4.3e-19.
   - Reported: SST-2003 on 273×193, Cf 0.0026931 (−0.45% from SST-1994),
     CD 0.0028552, ν_t/ν peak 219.4. 273×193 marched again with the ILU
     solver gives the native run's Cf, CD and peak to ten digits.

   The margin in 4a is thin, and where it goes is visible. On every grid
   Cf stands 0.68–0.85% above CFL3D's on the same grid, approaching the
   0.87% between the two codes' extrapolations: the two converge alike
   (orders 1.20 and 1.21), so the offset is not discretisation error. Nor
   is it the layer: in wall units the profile is TMR's to 0.1% out to
   y⁺ = 500. It is outside it. Here the velocity at x = 0.97 is 1.0019 U∞
   just outside the layer and falls to 1.0006 at y = 0.65, the
   displacement flow speeding up under the top's p = 0, and the outer u⁺
   is 27.20, where TMR's is 27.12 and flat to 0.02% over the same heights.
   A faster edge flow than TMR's inflow and far-field conditions give, and
   the 0.2–0.3% of incompressibility, could account for most of the
   offset; neither is measured here, and the gate's 1% was set with only
   the second in view.
5. *v1 and v2a untouched.* The same full suite (ADR-041's gate 5): every
   v0, v1 and v2a gate passes with the model in the tree, and v2b's own
   with them, 37 gates; the flat plate above and the heated cavity ran on
   their own.

**Cost.** Each flat-plate grid on one core shared with another run:
69×49 72 s, 137×97 559 s, 273×193 2.6 h (the native solver), 545×385
56 minutes with ILU and four correctors from 273×193's state; the
momentum, k and ω solves were most of a step until the ILU solver, the
assembly most of it after. The manufactured gates: 44 minutes in C++,
38 in Python.

## ADR-043 — Two PISO correctors and 545×385's odd–even mode: the cause, sought with a model. Stated before the model and the runs, answered after

**Context.** ADR-042 marches the flat plate with four PISO correctors,
because with two, 545×385 grew a streamwise odd–even mode of the pressure
and the velocity: the same in every row across the boundary layer
(0 < y < 3e-4, 118 rows), wrapped in an envelope some fifty cells long at
0.05 < x < 0.11. What is known of it:

- it grew at every step tried from 3.1e-4 up, and at 1e-2 from the
  sequenced start within 50 steps; two or three outer iterations a step
  made it grow sooner, not later;
- at 1e-2 it still blew up within 50 steps without the old-flux term,
  with the pressure solved to 1e-14, with first-order momentum advection,
  with k, ω and ν_t frozen, and with the wall pressure extrapolated by 0, 3
  or 10 sweeps (this last not recorded in ADR-042: the three runs gave one
  history);
- four correctors stopped it, and the count does not move the steady state
  (137×97, ten digits);
- 273×193 with two correctors marched to its steady state through the same
  ramp to 1e-2.

The last fact is the one to explain. 545×385 is 273×193 with every
spacing halved, so the two have the same aspect ratios (Δx/Δy₁ = 2,400 at
x = 0.08) and differ by two in the cell Reynolds numbers and in the
x-stretching per cell (0.32% against 0.64% there). Neither of the numbers
a step usually answers to orders them: at dt = 3.1e-4, where 545×385 grew,
its wall cells' ν dt/Δy₁² is 250 and the free stream's u dt/Δx at
x = 0.08 is 0.26; 273×193 at 1e-2, where nothing grew, has 2,000 and 4.2.

**Decision.** Seek the cause in three steps, each judged by a rule written
here.

*1. Measure, in C++.* From 545×385's steady state, marched with four
correctors to ADR-042's criteria (`VIBEFLOW_FP_SAVE`), restart at a fixed
step (`VIBEFLOW_FP_RAMP=1`) with n correctors and one outer iteration, and
record at every step the odd–even part of the change the step made,

    A(φ) = max over interior cells of |δφ(i−1) − 2 δφ(i) + δφ(i+1)| / 4,

δφ the change of φ over the step and i the cell's column, for φ = p and u.
A pure odd–even change gives its amplitude, a smooth one next to nothing.
The growth factor per step, λ, is the geometric mean of A(p)'s ratio over a
run's last 20 steps; a run ends after 200 steps, or when A(p) passes 1e-2.
The runs:

- 545×385: n = 2 and 3 at dt = 1e-3, 3e-3 and 1e-2; n = 4 at 1e-2;
- 273×193 and 137×97, each from its own steady state: n = 1 and 2 at the
  same three steps;
- at n = 2 and 1e-2 on 545×385, four ablations, each a switch off by
  default: without the ν_t(∇u)ᵀ term, BDF1 throughout, ADR-031's V1
  old-flux form, and the momentum solved to 1e-15 instead of 1e-13;
- two hybrid grids from TMR's 545×385 points — 545×193 (545's x-lines,
  273's y-lines) and 273×385 (the reverse) — each marched to its steady
  state with four correctors from 545×385's, then n = 2 at 1e-2: which of
  the two refinements carries the mode.

*2. Model, in Python* (`tests/benchmark/piso_mode.py`). A linear model of
the C++ step for the odd–even mode at one station x₀, the envelope's
middle (0.08; 0.05 and 0.11 reported):

- the C++ steady state's column at x₀ — its rows and faces, its Δx, u(y)
  and ν_t(y) — made periodic in x over two cells, so that the mean and the
  odd–even mode are all it holds;
- about the parallel flow U(y), V = 0, p = 0, held steady to round-off by
  a body force equal to its discrete residual, and upwinded from below on
  the horizontal faces, as a vanishing V > 0 would be;
- otherwise the C++ step on a Cartesian mesh, term for term: BDF2; upwind
  convection and diffusion (ν + ν_t on the faces) implicit; the
  linear-upwind correction and ν_t(∇u)ᵀ explicit; least-squares gradients
  with 1/d² weights and the boundary values the C++ gives each field, the
  wall pressure extrapolated by three sweeps; the predictor solved exactly;
  n correctors, each with H/aP, the exact Rhie–Chow flux and its old-flux
  term, the pressure solved exactly, the flux and the velocity corrected;
  the step's last flux in the next step's matrix and correction, the top at
  p = 0; k, ω and ν_t frozen, as the frozen C++ run says they may be;
- the growth factor, the largest |eigenvalue| of the step's Jacobian over
  odd–even eigenvectors; central differences give that Jacobian exactly,
  the step being quadratic in the state once upwinding is fixed;
- any term of the step can be switched off or replaced; which were, and
  what each did, is recorded with the results.

*3. The rules.*

- The model *reproduces* the C++ if, at every run of step 1 — grid, n and
  dt, the hybrids included, the ablations not — its λ is on the same side
  of 1 as the measured one, and where both grow, ln λ is within a factor
  two of the measured one.
- The *cause is found* if the model reproduces and a named term, or pair
  of terms, passes three tests: (a) removed in the model, 545×385 with two
  correctors is stable at all three steps; (b) removed in C++, by a
  diagnostic switch off by default, the 545×385 run with two correctors at
  1e-2 stops growing (λ < 1 in the protocol above); (c) the model says why
  545×385 and not 273×193: the property of the grid, or of the flow it
  carries, that sets that term's strength, taken continuously from one
  grid's value to the other's, carries λ across 1.
- A *fix* is adopted only if, with it, 545×385 marches with two correctors
  from 273×193's state to ADR-042's steady criteria; 69×49 to 545×385 give
  the four-corrector runs' Cf, CD and ν_t/ν peak within 1e-8 relative; and
  the full suite passes. Otherwise the flat plate keeps its four
  correctors, and the cause, with the count the model says a grid needs,
  goes into the known limits.
- If the model does not reproduce, the cause is not found: step 1's
  measurements are recorded, with whatever the ablations and the hybrids
  narrow, and the limit stays.

*Revision after the first runs, stated before the rest.* Looked at by
then: on 545×385, n = 2 at 1e-2 and 3e-3, n = 3 and 4 at 1e-2; all twelve
runs on 273×193 and 137×97; and the model at x₀ = 0.08 for every grid, n
and dt of step 1. 545×385 with n = 3 at 3e-3 had run and had not been
looked at; nothing else had run.

By the first rule the model does not reproduce. At n = 3 and 1e-2 on
545×385 the C++ grows by 1.506 a step and the model at x₀ = 0.08 by
1.055: both grow, 7.6 apart in ln λ, where two is allowed. The results
report it so.

What the runs showed that the rule did not foresee is where the mode
grows. From the steady state it grows where the step amplifies most, not
where ADR-042 saw it: at n = 3 in the wall row at x ≈ 0.22, moving
upstream about half a cell a step; at n = 2 first near the outlet,
x ≈ 1.95, where the run reached its bound at step 8. The model's λ varies
along the plate as much — at n = 3 and 1e-2 it is 0.19 at x = 0.005, 1.05
at 0.08, 1.51 at 0.20 and 1.09 at 0.50 — and ADR-042's envelope at
0.05–0.11 was where the sequenced start's disturbance was, not where the
step amplifies most. Nor did the protocol foresee runs that reach the
bound before they have 21 steps, as every n = 2 run on 545×385 so far and
every n = 1 run did.

A second test, then, judged on the runs not yet looked at:

- the model's λ is the largest over its 23 stations from x = 0.005 to 1.99
  (`STATIONS` in `piso_mode.py`, each by Arnoldi iteration, falling back
  to 200 steps of power iteration where Arnoldi stalls), and it
  reproduces if, at each of those runs, it is on the same side of 1 as the
  measured λ and, where both grow, within a factor two of it in ln λ;
- a run that reaches its bound before its 23rd step has λ from its steps
  3 on, (A_N / A_3)^(1/(N − 3)) — the first two steps carry the restart's
  own disturbance — or its last ratio if it ends by step 4;
- the cause rules are unchanged, with "the model reproduces" read as
  passing this second test, and test (b)'s λ by the same window; the first
  test's failure is reported beside it.

The model's numbers for those runs, computed before any of them was
looked at:

| run | model: largest λ (station) |
| --- | --- |
| 545×385, n = 3, dt = 3e-3 | 1.081 (x = 0.15) |
| 545×385, n = 2, dt = 1e-3 | 3.237 (0.11) |
| 545×385, n = 3, dt = 1e-3 | 0.942 (1.2) |
| 545×193, n = 2, dt = 1e-2 | 1.711 (0.15) |
| 273×385, n = 2, dt = 1e-2 | to follow, before its run |

Reported, not judged: the ablations, where the model has the term —
BDF1 7.60 and no ν_t(∇u)ᵀ 7.23, against 7.23 as run (x = 0.15); the V1
form and the momentum tolerance are not in the model, which has one
old-flux form and exact solves. And test (b): with D_f's aP leaving out
the vertical diffusion, the model's largest λ at n = 2 and 1e-2 is 0.999,
between 0.985 and 0.999 at every station, with no sign flip.

*Second revision, after the runs the first one judged, stated before the
runs it judges.* The second test fails too. At n = 3 and 3e-3 on 545×385
the C++ grows by 1.033 a step over its last 20 steps, the model by 1.081:
2.4 apart in ln λ. At n = 2 and 1e-3, 1.421 from step 3 against 3.237:
3.3 apart. At n = 3 and 1e-3 both decay — 0.979 over the last 20 of the
109 steps the run made before the machine restarted, against 0.942. The
hybrids did not run.

Both failures are in the measured λ, not in the model. A run that grows
slowly ends with its packet leaving the band where the step amplifies: at
n = 3 and 3e-3, A(p) grew by 1.08 a step over steps 70–150 while its
maximum moved upstream from x = 0.19 to 0.14, and by 1.03 as it reached
0.12, where the model's λ is 0.99. A run that blows up fast spends its
first steps on the restart's own disturbance: at n = 2 and 1e-3, A(p) fell
for six steps and then grew by 3.0 a step for eight, which the window from
step 3 reads as 1.42. Where a run grows cleanly the two agree: 1.506
measured and 1.512 in the model at n = 3 and 1e-2; 1.08 and 1.081 at 3e-3;
3.0 and 3.23 at n = 2 and 1e-3. And one run already looked at says more
than the first rule allowed for: 273×193 with two correctors at 1e-2 is
unstable in a band too. Its A(p) grew by 1.08 a step over steps 30–100 as
its maximum ran upstream from x = 0.19 to 0.11 — the model gives 1.10 at
0.15 — and then fell back near the leading edge, where the model's λ is
below 1; later packets did the same. That is why 273×193 marched to its
steady state: its packets are amplified a thousandfold and die, where
545×385's grow sevenfold a step.

A third test, then, on runs made after it:

- measured: the steepest sustained growth — the largest
  (A_{k+10}/A_k)^(1/10) over k ≥ 3 with A_k ≥ 1e-10, above the solvers'
  noise; over three steps for a run that ends before step 13; the largest
  single-step ratio after step 2 for one that ends before step 6
  (`growth_phase` in `flat_plate_mode.py`);
- model: the largest λ over its 23 stations, as in the second test;
- a run grows when its number is above 1.01, on either side; the model
  reproduces if at every run below the two agree on whether it grows and,
  where both grow, agree within a factor two in ln λ;
- the runs, with the model's numbers computed before them:

  | run | model: largest λ (station) |
  | --- | --- |
  | 545×193, n = 2, dt = 1e-2 | 1.711 (x = 0.15) |
  | 273×385, n = 2, dt = 1e-2 | 4.584 (0.15), added before its run |
  | 545×385, n = 3, dt = 5e-3 | 1.294 (0.20) |
  | 545×385, n = 2, dt = 3e-4 | 1.315 (0.08) |
  | 545×385, n = 4, dt = 3e-2 | 0.581 (0.25) |
  | 273×193, n = 2, dt = 3e-2 | 1.334 (0.20) |

- the cause rules as written, with "the model reproduces" read as this
  test, and (b)'s run judged by this measure: no growth above 1.01.

Test (c) needs no C++ run, and its premise stands at the station it was
written for: at x₀ = 0.08, 273×193's λ with two correctors is −0.971 and
545×385's −6.84.

### Results

**By its rules ADR-043 does not find the cause.** The model fails all
three reproduction tests, each on the measured side: every measure of the
C++ growth the tests used catches, in some run, something other than the
mode's own growth. And removing the term the model names does not stop
the growth in C++: the scheme blows up at the leading edge instead. What
the model and the runs do show is recorded below. The flat plate keeps
four correctors.

1. *The measurements.* Every run, restarted from its grid's steady state:
   λ over the last 20 steps (or by the first revision's window), the
   third test's steepest sustained growth, and the model's λ at x₀ = 0.08
   and at its most unstable station.

   | run | steps | last 20 (window) | steepest | model x₀ = 0.08 | model, largest (x) |
   | --- | --- | --- | --- | --- | --- |
   | 545×385, n = 2, 1e-2 | 8 | 2.849 | 3.046 | 6.836 | 7.229 (0.15) |
   | 545×385, n = 2, 3e-3 | 12 | 1.896 | 4.711 | 5.306 | 5.454 (0.11) |
   | 545×385, n = 2, 1e-3 | 17 | 1.421 | 2.225 | 3.228 | 3.237 (0.11) |
   | 545×385, n = 2, 3e-4 | 73 | 1.284 | 1.286 | — | 1.315 (0.08) |
   | 545×385, n = 3, 1e-2 | 45 | 1.506 | 1.510 | 1.055 | 1.512 (0.20) |
   | 545×385, n = 3, 5e-3 | 74 | 1.289 | 1.289 | — | 1.294 (0.20) |
   | 545×385, n = 3, 3e-3 | 200 | 1.033 | 1.084 | 0.820 | 1.081 (0.15) |
   | 545×385, n = 3, 1e-3 | 109 | 0.979 | 0.989 | 0.644 | 0.942 (1.2) |
   | 545×385, n = 4, 1e-2 | 200 | 0.989 | 0.988 | 0.237 | 0.616 (1.99) |
   | 545×385, n = 4, 3e-2 | 200 | 0.990 | 1.087 | — | 0.581 (0.25) |
   | 273×193, n = 2, 3e-2 | 69 | 1.294 | 1.316 | — | 1.334 (0.20) |
   | 273×193, n = 2, 1e-2 | 200 | 0.995 | 1.090 | 0.971 | 1.098 (0.15) |
   | 273×193, n = 2, 3e-3 | 200 | 0.986 | 1.011 | 0.647 | 0.915 (1.4) |
   | 273×193, n = 2, 1e-3 | 200 | 0.988 | 0.992 | 0.783 | 0.970 (1.2) |
   | 273×193, n = 1, all three | 3–4 | 34–153 | | 73–172 | |
   | 137×97, n = 2, 1e-2 | 200 | 0.969 | 1.094 | 0.420 | 0.866 (1.98) |
   | 137×97, n = 2, 3e-3 / 1e-3 | 200 | 0.975 / 0.972 | 0.992 / 0.984 | 0.707 / 0.879 | 0.956 / 0.985 |
   | 137×97, n = 1, all three | 4–5 | 21–68 | | 19–51 | |
   | 545×193, n = 2, 1e-2 | 36 | 1.619 | 1.655 | — | 1.711 (0.15) |
   | 273×385, n = 2, 1e-2 | 14 | 1.722 | 1.809 | — | 4.584 (0.15) |

   545×385 at n = 3 and 1e-3 ran 109 of its 200 steps before the machine
   restarted; its queue's other runs were made again after it. The four
   ablations, n = 2 at 1e-2 on 545×385, all still blow up, within three to
   eight steps: without ν_t(∇u)ᵀ (3.71), with BDF1 (3.65), with the V1
   old-flux form (2.92), with the momentum solved to 1e-15 (2.85, the
   run without it to four digits). The model has the first two: 7.23 and
   7.60, against 7.23 as run.

2. *The three tests.* The first, at x₀ = 0.08: at n = 3 and 1e-2 the
   C++ grows by 1.506 and the model by 1.055, 7.6 apart in ln λ. The
   second, the model's largest λ against the last 20 steps: 1.033 against
   1.081 at n = 3 and 3e-3 (2.4 apart), 1.421 against 3.237 at n = 2 and
   1e-3 (3.3 apart). The third, the steepest sustained growth, on runs
   made after it: five of its seven agree within 1.02–1.09 in ln λ
   (545×193 1.655 and 1.711; 545×385 at n = 3 and 5e-3 1.289 and 1.294,
   at n = 2 and 3e-4 1.286 and 1.315; 273×193 at n = 2 and 3e-2 1.316 and
   1.334), and two do not. 545×385 with four correctors at 3e-2 decays
   from its restart to A(p) ≈ 1e-10 and then flickers at the outlet,
   x = 1.98, between 5e-11 and 3e-10, which the measure reads as growth,
   1.087, where the model gives 0.58. 273×385 grows at the outlet by 1.68
   a step for ten steps — the model's λ there is 1.68 — and reaches the
   bound at step 14, the step its fastest mode, 4.58 at x = 0.15 in the
   model, first shows: the measure reads 1.81.

   The failures have one thing in common: a single number per run cannot
   separate two modes growing at different rates in different places, nor
   a mode from the restart's own disturbance or from the solvers' noise.
   Where one mode grows alone the two agree to 0.2–7%.

3. *The term.* (a) In the model, D_f's aP without the vertical diffusion
   takes 545×385's largest λ at n = 2 and 1e-2 from 7.23 to 0.999. (b) In
   C++ (`VIBEFLOW_RC_AXIS_OFF=1`) the same run blows up in four steps, ten
   times a step, in the first wall cell behind the leading edge — a place
   the model, a parallel flow, does not have. (c) Its premise holds at the
   station it was written for, −0.971 on 273×193 and −6.84 on 545×385; the
   width of 273×193's column alone, taken to 545×385's, carries λ across
   −1 at 10% below 273×193's own (λ(2) −0.971, −1.066, −1.625 at Δx =
   2.41e-3, 2.17e-3, 1.20e-3).

4. *What the model shows*, reported and not judged. The mode is the
   pressure's odd–even mode in x, uniform across the inner layer to
   y ≈ 1e-4 and fading by the layer's middle; the velocity carries
   1e-3 of it. Its λ is negative: it changes sign every step. In one step
   the first corrector overshoots its pressure — by 41 times on 273×193
   and 101 times on 545×385 at x₀ — because D_f = V_f/aP_f counts the
   diffusion between rows, which a mode uniform across the rows does not
   feel, while the predictor's velocity answers the pressure through the
   whole operator. Each further corrector takes most of the overshoot
   out, 98% of it per corrector on 273×193 but only 77–93% on 545×385.
   Two correctors leave |λ| ≈ 1: 0.97 and 6.8. Three leave 0.28 and 1.05,
   four 0.28 and 0.24. Refining in y does most of it: at n = 2 and 1e-2,
   273×193's largest λ is 1.10, 545×193's (x refined) 1.71, 273×385's
   (y refined) 4.58, 545×385's 7.23.

   And 273×193 with two correctors is unstable too, in a band
   0.1 < x < 0.3 where its λ is up to 1.10. Its packets run upstream,
   growing some five-hundredfold, and die near the leading edge, where λ
   is below 1; that is how it marched to its steady state, and why
   545×385, at 7 a step, did not.

5. *A bug the hybrids found.* The flat plate's restart matched a cell to
   its coarse column by x within 1e-12; on 273×385 two columns' centres
   spread by 1.2e-12 of round-off, so a restart from its own state put
   690 cells in the wrong place (Cf after one step 5.19e-3 against the
   state's 2.71e-3). Every other grid's spread is below 6e-13, so their
   restarts, ADR-042's sequencing and the hybrids' own starts were exact.
   The match is 1e-9 now; 273×385's dump, its model number and its run
   came after the fix.

**Decision.** No fix is adopted — none was tried: the one change the
model points to breaks the scheme at the leading edge. The flat plate
keeps its four correctors: on 545×385 the smallest count whose largest λ
in the model is below 1, at 1e-2 (0.62) and at 3e-2 (0.58), and whose C++
runs decay at both. The known limits say what ADR-043 found, and that the
rules did not let it call that the cause. The diagnostic switches
(`bdf1`, `momentumSolveTol`, `rhieChowAxisOff`) stay, off.

**Cost.** Step 1's 30 C++ runs: about 4 hours on one core, shared; the
four-corrector marches to the base states (545×385, 137×97, the two
hybrids) 2.7 hours. The model: 1–6 s a station by Arnoldi, a minute where
it falls back to power iteration, 1–4 minutes a scan of 23 stations.

## ADR-044 — p_h on the pressure's backend. Stated before the code, answered after

**Context.** ADR-041 solves the hydrostatic pressure p_h with the native
Jacobi CG, whatever backend the pressure uses. Its Laplacian is constant —
the pressure's orthogonal coefficients, Neumann on every face — assembled
once, and solved to 1e-15 of its right-hand side's norm, absolute, inside
a non-orthogonal loop. Beside the heated cavity's BoomerAMG pressure it is
the dearer of the two solves: the cavity at Ra = 10⁴ on 32² and 64² took
144 s against the cell form's 84, the whole cavity gate 243 minutes on a
shared core. ADR-041 left giving p_h the pressure's backend as a follow-up
that changes no answer. This is it.

**Decision.**

- `PisoSolver::setHydrostaticSolver(std::unique_ptr<LinearSolver>)` gives
  p_h a solver of the caller's choosing, before or after `enableEnergy`;
  without it p_h keeps the native CG. The solve's tolerances do not change.
- So that the tolerance means for PETSc what it means for the native CG —
  the residual's 2-norm below 1e-15 of the right-hand side's — a PETSc
  solver for p_h measures the unpreconditioned residual (`KSPSetNormType`)
  and carries the constants as the operator's null space
  (`MatSetNullSpace`), as the native CG projects them out at every
  iteration. Both are options of `PetscSolver`, off unless asked for; the
  pressure's own solves do not change.
- The heated cavity gives p_h the pressure's configuration
  (`VIBEFLOW_PRESSURE`, BoomerAMG CG by default where PETSc is built): the
  same backend. Every other buoyant harness solves its pressure with the
  native CG, so its p_h has the pressure's backend already, and keeps it.
- The C++ heat-transfer gates take `VIBEFLOW_PH=<PETSc configuration>` for
  this ADR's check; the suite runs them as before.

**Gates**, stated before the code:

1. *The balanced gates with a PETSc p_h.* `heat_transfer`, gates 2 to 5 on
   8³ and 16³ with its dt check, with `VIBEFLOW_PH=cg+hypre` and without:
   every gate passes both ways, the resting fluids within their own
   1e-12, and gate 2's L2 errors agree within 1e-6 relative.
2. *The cavity.* The whole heated-cavity gate with p_h on BoomerAMG passes,
   and on all twelve meshes and Rayleigh numbers its step counts, and Nu,
   u_max and v_max to the digits printed, are those of ADR-041's run (its
   log kept). Where a count or a last digit differs, that mesh is run again
   with the native p_h, both printed to ten digits, and the two must agree
   within 1e-6 relative.
3. *Nothing else moves.* The full suite passes.

Adopted if the three pass and the cavity at Ra = 10⁴ on 32² and 64², one
thread, timed back to back with the native p_h, is not slower; the cost of
the whole gate is reported.
