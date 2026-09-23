#include "linalg/NativeCG.hpp"
#include "linalg/LinearSystem.hpp"
#include "mesh/Mesh.hpp"
#include <chrono>
#include <cmath>

namespace nsflow {

NativeCG::NativeCG(const Mesh& mesh, Comm comm, bool singularNullSpace)
    : m_(mesh), comm_(comm), nullSpace_(singularNullSpace),
      r_("r", mesh.nTotal()), z_("z", mesh.nTotal()),
      p_("p", mesh.nTotal()), q_("q", mesh.nTotal()) {}

void NativeCG::projectOut(ScalarField& v) const {
  // Remove the constant component. Projecting the right-hand side once is not
  // enough: round-off re-injects a constant every iteration, CG cannot reduce
  // it, and the residual norm stalls above any tight tolerance. MEASURED on
  // the distorted Ethier-Steinman case, pressure iterations for the whole
  // run: 2,745,371 without this projection against 31,688 with PETSc's CG
  // using the same Jacobi preconditioner -- an 87x gap that looked like an
  // algorithmic cost and was a stalling solver.
  if (!nullSpace_) return;
  const Index nc = m_.nCells();
  Real mean = 0.0;
  Kokkos::parallel_reduce("nsMean", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index c, Real& a) { a += v(c); }, mean);
  mean = comm_.sum(mean) / static_cast<Real>(comm_.sum(nc));
  Kokkos::parallel_for("nsShift", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index c) { v(c) -= mean; });
  Kokkos::fence();
}

void NativeCG::apply(LinearSystem& sys, const ScalarField& x, ScalarField& y) const {
  // A face on a rank boundary reads x at a ghost cell, so the halo must be
  // current before the face loop. This is the one place a missing exchange
  // silently produces a plausible-looking wrong answer.
  if (const auto* h = m_.halo()) h->exchange(x);

  auto diag = sys.diag(); auto up = sys.upper(); auto lo = sys.lower();
  auto own = m_.owner(); auto nei = m_.neighbour();
  const Index nc = m_.nTotal(), nf = m_.nInternalFaces();

  Kokkos::parallel_for("spmvDiag", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index c) { y(c) = diag(c) * x(c); });
  Kokkos::fence();
  Kokkos::parallel_for("spmvFaces", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      Kokkos::atomic_add(&y(own(f)), up(f) * x(nei(f)));
      Kokkos::atomic_add(&y(nei(f)), lo(f) * x(own(f)));
    });
  Kokkos::fence();
}

namespace {
// Reductions run over OWNED cells only and are summed across ranks: counting a
// ghost would double-count the cell that owns it.
Real dot(const ScalarField& a, const ScalarField& b, Index nOwned, const Comm& comm) {
  Real s = 0.0;
  Kokkos::parallel_reduce("dot", Kokkos::RangePolicy<ExecSpace>(0, nOwned),
    KOKKOS_LAMBDA(const Index i, Real& acc) { acc += a(i) * b(i); }, s);
  return comm.sum(s);
}
}  // namespace

SolveReport NativeCG::solve(LinearSystem& sys, ScalarField& x,
                            Real relTol, Real absTol, int maxIter) {
  const auto t0 = std::chrono::steady_clock::now();
  const Index nc = m_.nCells();          // owned: reductions and updates
  const Index nt = m_.nTotal();          // total: vector writes
  auto b = sys.source(); auto diag = sys.diag();
  auto r = r_, z = z_, p = p_, q = q_;

  apply(sys, x, q);
  Kokkos::parallel_for("r0", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) { r(c) = b(c) - q(c); });
  Kokkos::fence();
  projectOut(r);

  const Real r0 = std::sqrt(dot(r, r, nc, comm_));
  SolveReport rep;
  rep.initialResidual = r0;
  if (r0 < absTol) { rep.converged = true; rep.finalResidual = r0; return rep; }

  Real rz_old = 0.0, rn = r0;
  for (int it = 1; it <= maxIter; ++it) {
    Kokkos::parallel_for("precon", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c) { z(c) = r(c) / diag(c); });
    Kokkos::fence();
    const Real rz = dot(r, z, nc, comm_);
    if (it == 1) {
      Kokkos::deep_copy(p, z);
    } else {
      const Real beta = rz / rz_old;
      Kokkos::parallel_for("pupd", Kokkos::RangePolicy<ExecSpace>(0, nt),
        KOKKOS_LAMBDA(const Index c) { p(c) = z(c) + beta * p(c); });
      Kokkos::fence();
    }
    rz_old = rz;

    apply(sys, p, q);
    const Real alpha = rz / dot(p, q, nc, comm_);
    Kokkos::parallel_for("xupd", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { x(c) += alpha * p(c); r(c) -= alpha * q(c); });
    Kokkos::fence();
    projectOut(r);

    rn = std::sqrt(dot(r, r, nc, comm_));
    rep.iterations = it;
    if (rn < absTol || rn < relTol * r0) { rep.converged = true; break; }
  }
  rep.finalResidual = rn;
  rep.wallSeconds = std::chrono::duration<Real>(
      std::chrono::steady_clock::now() - t0).count();
  record(rep);
  return rep;
}

}  // namespace nsflow
