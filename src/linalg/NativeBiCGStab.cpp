#include "linalg/NativeBiCGStab.hpp"
#include "linalg/LinearSystem.hpp"
#include "mesh/Mesh.hpp"
#include <chrono>
#include <cmath>

namespace vibeflow {
namespace {
Real dot(const ScalarField& a, const ScalarField& b, Index n, const Comm& c) {
  Real s = 0.0;
  Kokkos::parallel_reduce("dot", Kokkos::RangePolicy<ExecSpace>(0, n),
    KOKKOS_LAMBDA(const Index i, Real& acc) { acc += a(i) * b(i); }, s);
  return c.sum(s);
}
}  // namespace

NativeBiCGStab::NativeBiCGStab(const Mesh& mesh, Comm comm)
    : m_(mesh), comm_(comm),
      r_("r", mesh.nTotal()), r0_("r0", mesh.nTotal()), p_("p", mesh.nTotal()),
      v_("v", mesh.nTotal()), s_("s", mesh.nTotal()), t_("t", mesh.nTotal()),
      ph_("ph", mesh.nTotal()), sh_("sh", mesh.nTotal()) {}

void NativeBiCGStab::apply(LinearSystem& sys, const ScalarField& x,
                           ScalarField& y) const {
  if (const auto* h = m_.halo()) h->exchange(x);
  auto diag = sys.diag(); auto up = sys.upper(); auto lo = sys.lower();
  auto own = m_.owner(); auto nei = m_.neighbour();
  Kokkos::parallel_for("spmvDiag", Kokkos::RangePolicy<ExecSpace>(0, m_.nTotal()),
    KOKKOS_LAMBDA(const Index c) { y(c) = diag(c) * x(c); });
  Kokkos::fence();
  Kokkos::parallel_for("spmvFace", Kokkos::RangePolicy<ExecSpace>(0, m_.nInternalFaces()),
    KOKKOS_LAMBDA(const Index f) {
      Kokkos::atomic_add(&y(own(f)), up(f) * x(nei(f)));
      Kokkos::atomic_add(&y(nei(f)), lo(f) * x(own(f)));
    });
  Kokkos::fence();
}

SolveReport NativeBiCGStab::solve(LinearSystem& sys, ScalarField& x,
                                  Real relTol, Real absTol, int maxIter) {
  const auto t0 = std::chrono::steady_clock::now();
  const Index nc = m_.nCells(), nt = m_.nTotal();
  auto b = sys.source(); auto diag = sys.diag();
  auto r = r_, r0 = r0_, p = p_, v = v_, s = s_, t = t_, ph = ph_, sh = sh_;

  apply(sys, x, v);
  Kokkos::parallel_for("r0", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) { r(c) = b(c) - v(c); r0(c) = r(c); p(c) = r(c); });
  Kokkos::fence();

  SolveReport rep;
  Real rn = std::sqrt(dot(r, r, nc, comm_));
  rep.initialResidual = rn;
  if (rn < absTol) { rep.converged = true; rep.finalResidual = rn; if (const auto* hx = m_.halo()) hx->exchange(x);
    record(rep); return rep; }
  const Real r0norm = rn;
  Real rho = dot(r0, r, nc, comm_);

  for (int it = 1; it <= maxIter; ++it) {
    Kokkos::parallel_for("precon1", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { ph(c) = p(c) / diag(c); });
    Kokkos::fence();
    apply(sys, ph, v);
    const Real alpha = rho / dot(r0, v, nc, comm_);

    Kokkos::parallel_for("sUpd", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { s(c) = r(c) - alpha * v(c); });
    Kokkos::fence();
    rn = std::sqrt(dot(s, s, nc, comm_));
    if (rn < absTol || rn < relTol * r0norm) {
      Kokkos::parallel_for("xEarly", Kokkos::RangePolicy<ExecSpace>(0, nt),
        KOKKOS_LAMBDA(const Index c) { x(c) += alpha * ph(c); });
      Kokkos::fence();
      rep.iterations = it; rep.converged = true; break;
    }

    Kokkos::parallel_for("precon2", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { sh(c) = s(c) / diag(c); });
    Kokkos::fence();
    apply(sys, sh, t);
    const Real omega = dot(t, s, nc, comm_) / dot(t, t, nc, comm_);

    Kokkos::parallel_for("xUpd", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) {
        x(c) += alpha * ph(c) + omega * sh(c);
        r(c) = s(c) - omega * t(c);
      });
    Kokkos::fence();

    const Real rho_new = dot(r0, r, nc, comm_);
    const Real beta = (rho_new / rho) * (alpha / omega);
    rho = rho_new;
    Kokkos::parallel_for("pUpd", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { p(c) = r(c) + beta * (p(c) - omega * v(c)); });
    Kokkos::fence();

    rn = std::sqrt(dot(r, r, nc, comm_));
    rep.iterations = it;
    if (rn < absTol || rn < relTol * r0norm) { rep.converged = true; break; }
  }
  rep.finalResidual = rn;
  // The last x update happened after the last matrix-vector product, so the
  // halo is one step stale. See the contract on LinearSolver::solve.
  if (const auto* hx = m_.halo()) hx->exchange(x);
  rep.wallSeconds = std::chrono::duration<Real>(
      std::chrono::steady_clock::now() - t0).count();
  record(rep);
  return rep;
}

}  // namespace vibeflow
