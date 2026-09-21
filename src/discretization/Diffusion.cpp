#include "discretization/Diffusion.hpp"
#include "linalg/LinearSystem.hpp"
#include "linalg/LinearSolver.hpp"
#include "mesh/Mesh.hpp"
#include <cmath>

namespace nsflow {

DiffusionOperator::DiffusionOperator(const Mesh& mesh, Real gamma, Comm comm)
    : m_(mesh), gamma_(gamma), comm_(comm),
      aInt_("aInt", mesh.nInternalFaces()),
      aBnd_("aBnd", mesh.nBoundaryFaces()),
      wOwner_("wOwner", mesh.nInternalFaces()),
      kInt_("kInt", mesh.nInternalFaces(), 3),
      kBnd_("kBnd", mesh.nBoundaryFaces(), 3),
      grad_(mesh) {
  const Index nf = mesh.nInternalFaces(), nb = mesh.nBoundaryFaces();
  auto own = mesh.owner(); auto nei = mesh.neighbour(); auto bc = mesh.boundaryCell();
  auto cc = mesh.cellCentre(); auto fa = mesh.faceArea(); auto fc = mesh.faceCentre();
  auto bar = mesh.boundaryArea(); auto bcen = mesh.boundaryCentre();
  auto a = aInt_, ab = aBnd_, w = wOwner_; auto k = kInt_, kb = kBnd_;

  Kokkos::parallel_for("diffInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      Real d[3], ds = 0.0, ss = 0.0, dof = 0.0, dnf = 0.0;
      for (int i = 0; i < 3; ++i) {
        d[i] = cc(nei(f), i) - cc(own(f), i);
        ds += d[i] * fa(f, i);
        ss += fa(f, i) * fa(f, i);
        const Real ro = fc(f, i) - cc(own(f), i);
        const Real rn = fc(f, i) - cc(nei(f), i);
        dof += ro * ro; dnf += rn * rn;
      }
      a(f) = ss / ds;
      for (int i = 0; i < 3; ++i) k(f, i) = fa(f, i) - a(f) * d[i];
      const Real lo = Kokkos::sqrt(dof), ln = Kokkos::sqrt(dnf);
      w(f) = ln / (lo + ln);
    });
  Kokkos::parallel_for("diffBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      Real d[3], ds = 0.0, ss = 0.0;
      for (int i = 0; i < 3; ++i) {
        d[i] = bcen(f, i) - cc(bc(f), i);
        ds += d[i] * bar(f, i);
        ss += bar(f, i) * bar(f, i);
      }
      ab(f) = ss / ds;
      for (int i = 0; i < 3; ++i) kb(f, i) = bar(f, i) - ab(f) * d[i];
    });
  Kokkos::fence();
}

void DiffusionOperator::assembleMatrix(LinearSystem& sys) const {
  sys.zero();
  const Index nf = m_.nInternalFaces(), nb = m_.nBoundaryFaces();
  auto own = m_.owner(); auto nei = m_.neighbour(); auto bc = m_.boundaryCell();
  auto diag = sys.diag(); auto up = sys.upper(); auto lo = sys.lower();
  auto a = aInt_, ab = aBnd_; const Real g = gamma_;

  Kokkos::parallel_for("asmInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      const Real c = g * a(f);
      Kokkos::atomic_add(&diag(own(f)), c);
      Kokkos::atomic_add(&diag(nei(f)), c);
      up(f) = -c; lo(f) = -c;
    });
  Kokkos::parallel_for("asmBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) { Kokkos::atomic_add(&diag(bc(f)), g * ab(f)); });
  Kokkos::fence();
}

void DiffusionOperator::assembleSource(LinearSystem& sys, const ScalarField& volSource,
                                       const ScalarField& phiB,
                                       const ScalarField& phiPrev) const {
  const Index nc = m_.nCells(), nt = m_.nTotal();
  const Index nf = m_.nInternalFaces(), nb = m_.nBoundaryFaces();
  auto own = m_.owner(); auto nei = m_.neighbour(); auto bc = m_.boundaryCell();
  auto vol = m_.cellVolume();
  auto b = sys.source();
  auto a = aInt_, ab = aBnd_, w = wOwner_; auto k = kInt_, kb = kBnd_;
  const Real g = gamma_;

  VectorField gr("grad", nt, 3);
  grad_(phiPrev, phiB, gr);

  Kokkos::parallel_for("srcVol", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) { b(c) = volSource(c) * vol(c); });
  Kokkos::fence();

  Kokkos::parallel_for("srcInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      Real corr = 0.0;
      for (int i = 0; i < 3; ++i) {
        const Real gf = w(f) * gr(own(f), i) + (1.0 - w(f)) * gr(nei(f), i);
        corr += k(f, i) * gf;
      }
      corr *= g;
      Kokkos::atomic_add(&b(own(f)), corr);
      Kokkos::atomic_add(&b(nei(f)), -corr);
    });
  Kokkos::parallel_for("srcBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      Real corr = 0.0;
      for (int i = 0; i < 3; ++i) corr += kb(f, i) * gr(bc(f), i);
      Kokkos::atomic_add(&b(bc(f)), g * (ab(f) * phiB(f) + corr));
    });
  Kokkos::fence();
}

int DiffusionOperator::solve(LinearSystem& sys, LinearSolver& solver,
                             const ScalarField& volSource, const ScalarField& phiB,
                             ScalarField& phi, int maxSweeps, Real tol) const {
  const Index nc = m_.nCells(), nt = m_.nTotal();
  assembleMatrix(sys);
  ScalarField prev("prev", nt);
  Kokkos::deep_copy(phi, 0.0);

  for (int sweep = 1; sweep <= maxSweeps; ++sweep) {
    Kokkos::deep_copy(prev, phi);
    assembleSource(sys, volSource, phiB, prev);
    // Solve to well below the discretization error so the sweep loop, not the
    // linear solve, sets the accuracy.
    solver.solve(sys, phi, 1e-14, 1e-16, 5000);

    Real delta = 0.0, scale = 0.0;
    auto p = phi, q = prev;
    Kokkos::parallel_reduce("sweepDelta", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& acc) {
        acc = Kokkos::max(acc, Kokkos::abs(p(c) - q(c)));
      }, Kokkos::Max<Real>(delta));
    Kokkos::parallel_reduce("sweepScale", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& acc) {
        acc = Kokkos::max(acc, Kokkos::abs(p(c)));
      }, Kokkos::Max<Real>(scale));
    // The sweep test must agree on every rank or they run different loop
    // counts and the parallel answer stops matching the serial one.
    delta = comm_.max(delta);
    scale = comm_.max(scale);
    if (delta < tol * std::max(1.0, scale)) return sweep;
  }
  return maxSweeps;
}

}  // namespace nsflow
