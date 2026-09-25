#include "discretization/Convection.hpp"
#include "linalg/LinearSystem.hpp"
#include "linalg/LinearSolver.hpp"
#include "mesh/Mesh.hpp"
#include <cmath>

namespace vibeflow {

ConvectionDiffusion::ConvectionDiffusion(const Mesh& mesh, Real gamma,
                                         const ScalarField& fi, const ScalarField& fb)
    : m_(mesh), gamma_(gamma), F_(fi), Fb_(fb),
      aInt_("aInt", mesh.nInternalFaces()),
      aBnd_("aBnd", mesh.nBoundaryFaces()),
      wOwner_("wOwner", mesh.nInternalFaces()),
      kInt_("kInt", mesh.nInternalFaces(), 3),
      kBnd_("kBnd", mesh.nBoundaryFaces(), 3),
      skew_("skew", mesh.nInternalFaces(), 3),
      grad_(mesh) {
  const Index nf = mesh.nInternalFaces(), nb = mesh.nBoundaryFaces();
  auto own = mesh.owner(); auto nei = mesh.neighbour(); auto bc = mesh.boundaryCell();
  auto cc = mesh.cellCentre(); auto fa = mesh.faceArea(); auto fc = mesh.faceCentre();
  auto bar = mesh.boundaryArea(); auto bcen = mesh.boundaryCentre();
  auto a = aInt_, ab = aBnd_, w = wOwner_;
  auto k = kInt_, kb = kBnd_, sk = skew_;

  Kokkos::parallel_for("cdInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      Real d[3], ds = 0.0, ss = 0.0, lo = 0.0, ln = 0.0;
      for (int i = 0; i < 3; ++i) {
        d[i] = cc(nei(f), i) - cc(own(f), i);
        ds += d[i] * fa(f, i);
        ss += fa(f, i) * fa(f, i);
        const Real ro = fc(f, i) - cc(own(f), i);
        const Real rn = fc(f, i) - cc(nei(f), i);
        lo += ro * ro; ln += rn * rn;
      }
      a(f) = ss / ds;
      for (int i = 0; i < 3; ++i) k(f, i) = fa(f, i) - a(f) * d[i];
      const Real do_ = Kokkos::sqrt(lo), dn = Kokkos::sqrt(ln);
      w(f) = dn / (do_ + dn);
      for (int i = 0; i < 3; ++i)
        sk(f, i) = fc(f, i) - (w(f) * cc(own(f), i) + (1.0 - w(f)) * cc(nei(f), i));
    });
  Kokkos::parallel_for("cdBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
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

void ConvectionDiffusion::assembleMatrix(LinearSystem& sys) const {
  sys.zero();
  const Index nf = m_.nInternalFaces(), nb = m_.nBoundaryFaces();
  auto own = m_.owner(); auto nei = m_.neighbour(); auto bc = m_.boundaryCell();
  auto diag = sys.diag(); auto up = sys.upper(); auto lo = sys.lower();
  auto a = aInt_, ab = aBnd_, F = F_;
  const Real g = gamma_;

  Kokkos::parallel_for("asmInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      const Real Fp = Kokkos::max(F(f), 0.0), Fn = Kokkos::max(-F(f), 0.0);
      Kokkos::atomic_add(&diag(own(f)), g * a(f) + Fp);
      Kokkos::atomic_add(&diag(nei(f)), g * a(f) + Fn);
      up(f) = -g * a(f) - Fn;      // owner row, neighbour column
      lo(f) = -g * a(f) - Fp;      // neighbour row, owner column
    });
  Kokkos::parallel_for("asmBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) { Kokkos::atomic_add(&diag(bc(f)), g * ab(f)); });
  Kokkos::fence();
}

void ConvectionDiffusion::assembleSource(LinearSystem& sys, const ScalarField& volSource,
                                         const ScalarField& phiB,
                                         const ScalarField& phiPrev) const {
  const Index nc = m_.nCells(), nt = m_.nTotal();
  const Index nf = m_.nInternalFaces(), nb = m_.nBoundaryFaces();
  auto own = m_.owner(); auto nei = m_.neighbour(); auto bc = m_.boundaryCell();
  auto vol = m_.cellVolume();
  auto b = sys.source();
  auto a = aInt_, ab = aBnd_, w = wOwner_, F = F_, Fb = Fb_;
  auto k = kInt_, kb = kBnd_, sk = skew_;
  const Real g = gamma_;
  (void)nc;

  VectorField gr("grad", nt, 3);
  grad_(phiPrev, phiB, gr);

  Kokkos::parallel_for("srcVol", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) { b(c) = volSource(c) * vol(c); });
  Kokkos::fence();

  Kokkos::parallel_for("srcInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      Real nonorth = 0.0, ho = 0.0;
      for (int i = 0; i < 3; ++i) {
        const Real gf = w(f) * gr(own(f), i) + (1.0 - w(f)) * gr(nei(f), i);
        nonorth += k(f, i) * gf;
        ho += gf * sk(f, i);
      }
      ho += w(f) * phiPrev(own(f)) + (1.0 - w(f)) * phiPrev(nei(f));
      const Real ud = F(f) > 0.0 ? phiPrev(own(f)) : phiPrev(nei(f));
      const Real dc = F(f) * (ho - ud);

      Kokkos::atomic_add(&b(own(f)), g * nonorth - dc);
      Kokkos::atomic_add(&b(nei(f)), -g * nonorth + dc);
    });
  Kokkos::parallel_for("srcBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      Real nonorth = 0.0;
      for (int i = 0; i < 3; ++i) nonorth += kb(f, i) * gr(bc(f), i);
      // Dirichlet: the face value is known, so its whole convective flux is a
      // source term (OpenFOAM's fixedValue behaviour).
      Kokkos::atomic_add(&b(bc(f)), g * (ab(f) * phiB(f) + nonorth) - Fb(f) * phiB(f));
    });
  Kokkos::fence();
}

int ConvectionDiffusion::solve(LinearSystem& sys, LinearSolver& solver,
                               const ScalarField& volSource, const ScalarField& phiB,
                               ScalarField& phi, bool& converged,
                               int maxSweeps, Real tol, Real relax) const {
  const Index nc = m_.nCells(), nt = m_.nTotal();
  assembleMatrix(sys);
  ScalarField prev("prev", nt);
  Kokkos::deep_copy(phi, 0.0);
  converged = false;

  for (int sweep = 1; sweep <= maxSweeps; ++sweep) {
    Kokkos::deep_copy(prev, phi);
    assembleSource(sys, volSource, phiB, prev);
    solver.solve(sys, phi, 1e-14, 1e-16, 5000);

    if (relax < 1.0) {
      auto p = phi; auto q = prev;
      Kokkos::parallel_for("relax", Kokkos::RangePolicy<ExecSpace>(0, nt),
        KOKKOS_LAMBDA(const Index c) { p(c) = q(c) + relax * (p(c) - q(c)); });
      Kokkos::fence();
    }

    Real delta = 0.0, scale = 0.0;
    auto p = phi; auto q = prev;
    Kokkos::parallel_reduce("delta", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& acc) {
        acc = Kokkos::max(acc, Kokkos::abs(p(c) - q(c)));
      }, Kokkos::Max<Real>(delta));
    Kokkos::parallel_reduce("scale", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& acc) {
        acc = Kokkos::max(acc, Kokkos::abs(p(c)));
      }, Kokkos::Max<Real>(scale));
    if (delta < tol * std::max(1.0, scale)) { converged = true; return sweep; }
  }
  return maxSweeps;
}

Real ConvectionDiffusion::maxPeclet() const {
  auto own = m_.owner(); auto nei = m_.neighbour();
  auto cc = m_.cellCentre(); auto fa = m_.faceArea(); auto F = F_;
  const Real g = gamma_;
  Real m = 0.0;
  Kokkos::parallel_reduce("pe", Kokkos::RangePolicy<ExecSpace>(0, m_.nInternalFaces()),
    KOKKOS_LAMBDA(const Index f, Real& acc) {
      Real dd = 0.0, ss = 0.0;
      for (int i = 0; i < 3; ++i) {
        const Real d = cc(nei(f), i) - cc(own(f), i);
        dd += d * d; ss += fa(f, i) * fa(f, i);
      }
      // Cell Peclet = |F| |d| / (gamma |S|); dividing by |S|^2 inflates it
      // by 1/|S| ~ h^-2 and reported 780 where the true value is 12.
      acc = Kokkos::max(acc, Kokkos::abs(F(f)) * Kokkos::sqrt(dd)
                                 / (g * Kokkos::sqrt(ss)));
    }, Kokkos::Max<Real>(m));
  return m;
}

}  // namespace vibeflow
