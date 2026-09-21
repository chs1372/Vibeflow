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
