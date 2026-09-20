#include "linalg/NativeCG.hpp"
#include "linalg/LinearSystem.hpp"
#include "mesh/Mesh.hpp"
#include <chrono>
#include <cmath>

namespace nsflow {

NativeCG::NativeCG(const Mesh& mesh)
    : m_(mesh),
      r_("r", mesh.nCells()), z_("z", mesh.nCells()),
      p_("p", mesh.nCells()), q_("q", mesh.nCells()) {}

void NativeCG::apply(LinearSystem& sys, const ScalarField& x, ScalarField& y) const {
  auto diag = sys.diag(); auto up = sys.upper(); auto lo = sys.lower();
  auto own = m_.owner(); auto nei = m_.neighbour();
  const Index nc = m_.nCells(), nf = m_.nInternalFaces();

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
Real dot(const ScalarField& a, const ScalarField& b, Index n) {
  Real s = 0.0;
  Kokkos::parallel_reduce("dot", Kokkos::RangePolicy<ExecSpace>(0, n),
    KOKKOS_LAMBDA(const Index i, Real& acc) { acc += a(i) * b(i); }, s);
  return s;
}
}  // namespace

SolveReport NativeCG::solve(LinearSystem& sys, ScalarField& x,
                            Real relTol, Real absTol, int maxIter) {
  const auto t0 = std::chrono::steady_clock::now();
  const Index nc = m_.nCells();
  auto b = sys.source(); auto diag = sys.diag();
  auto r = r_, z = z_, p = p_, q = q_;

  apply(sys, x, q);
  Kokkos::parallel_for("r0", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index c) { r(c) = b(c) - q(c); });
  Kokkos::fence();

  const Real r0 = std::sqrt(dot(r, r, nc));
  SolveReport rep;
  rep.initialResidual = r0;
  if (r0 < absTol) { rep.converged = true; rep.finalResidual = r0; return rep; }

  Real rz_old = 0.0, rn = r0;
  for (int it = 1; it <= maxIter; ++it) {
    Kokkos::parallel_for("precon", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c) { z(c) = r(c) / diag(c); });
    Kokkos::fence();
    const Real rz = dot(r, z, nc);
    if (it == 1) {
      Kokkos::deep_copy(p, z);
    } else {
      const Real beta = rz / rz_old;
      Kokkos::parallel_for("pupd", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c) { p(c) = z(c) + beta * p(c); });
      Kokkos::fence();
    }
    rz_old = rz;

    apply(sys, p, q);
    const Real alpha = rz / dot(p, q, nc);
    Kokkos::parallel_for("xupd", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c) { x(c) += alpha * p(c); r(c) -= alpha * q(c); });
    Kokkos::fence();

    rn = std::sqrt(dot(r, r, nc));
    rep.iterations = it;
    if (rn < absTol || rn < relTol * r0) { rep.converged = true; break; }
  }
  rep.finalResidual = rn;
  rep.wallSeconds = std::chrono::duration<Real>(
      std::chrono::steady_clock::now() - t0).count();
  return rep;
}

}  // namespace nsflow
