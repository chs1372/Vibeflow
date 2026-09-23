#include "physics/Piso.hpp"
#include <chrono>
#include "linalg/LinearSolver.hpp"
#include "linalg/LinearSystem.hpp"
#include "mesh/Mesh.hpp"
#include <algorithm>
#include <cmath>

namespace nsflow {
namespace {
struct Stopwatch {
  std::chrono::steady_clock::time_point t0{std::chrono::steady_clock::now()};
  Real* acc;
  explicit Stopwatch(Real* a) : acc(a) {}
  ~Stopwatch() {
    *acc += std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
  }
};
}  // namespace


PisoSolver::PisoSolver(const Mesh& mesh, Real nu, Real dt, PisoControls ctl, Comm comm)
    : m_(mesh), nu_(nu), dt_(dt), ctl_(ctl), comm_(comm),
      diff_(mesh, nu, comm), pdiff_(mesh, 1.0, comm), grad_(mesh),
      w_("w", mesh.nInternalFaces()),
      aP_("aP", mesh.nTotal()),
      Df_("Df", mesh.nInternalFaces()),
      Fstar_("Fstar", mesh.nInternalFaces()),
      skew_("skew", mesh.nInternalFaces(), 3),
      u_("u", mesh.nTotal(), 3), uOld_("uOld", mesh.nTotal(), 3),
      uOld2_("uOld2", mesh.nTotal(), 3), HbyA_("HbyA", mesh.nTotal(), 3),
      gp_("gp", mesh.nTotal(), 3),
      gH0_("gH0", mesh.nTotal(), 3), gH1_("gH1", mesh.nTotal(), 3),
      gH2_("gH2", mesh.nTotal(), 3),
      p_("p", mesh.nTotal()), F_("F", mesh.nInternalFaces()),
      FOld_("FOld", mesh.nInternalFaces()), Fb_("Fb", mesh.nBoundaryFaces()),
      bSrc_("bSrc", mesh.nTotal(), 3),
      diag_("diag", mesh.nTotal()),
      upper_("upper", mesh.nInternalFaces()), lower_("lower", mesh.nInternalFaces()),
      bcType_("bcType", mesh.nBoundaryFaces()),
      pType_("pType", mesh.nBoundaryFaces()),
      pValue_("pValue", mesh.nBoundaryFaces()),
      FbStar_("FbStar", mesh.nBoundaryFaces()),
      uBnd_("uBnd", mesh.nBoundaryFaces(), 3) {
  auto own = mesh.owner(); auto nei = mesh.neighbour();
  auto cc = mesh.cellCentre(); auto fc = mesh.faceCentre();
  auto w = w_; auto sk = skew_;
  Kokkos::parallel_for("weights", Kokkos::RangePolicy<ExecSpace>(0, mesh.nInternalFaces()),
    KOKKOS_LAMBDA(const Index f) {
      Real lo = 0.0, ln = 0.0;
      for (int i = 0; i < 3; ++i) {
        const Real ro = fc(f, i) - cc(own(f), i);
        const Real rn = fc(f, i) - cc(nei(f), i);
        lo += ro * ro; ln += rn * rn;
      }
      w(f) = Kokkos::sqrt(ln) / (Kokkos::sqrt(lo) + Kokkos::sqrt(ln));
      for (int i = 0; i < 3; ++i)
        sk(f, i) = fc(f, i) - (w(f) * cc(own(f), i) + (1.0 - w(f)) * cc(nei(f), i));
    });
  Kokkos::fence();
}

void PisoSolver::setPressureBoundary(const View1<int>& pType,
                                     const ScalarField& pValue) {
  Kokkos::deep_copy(pType_, pType);
  Kokkos::deep_copy(pValue_, pValue);
  Index open = 0;
  auto pt = pType_;
  Kokkos::parallel_reduce("openCount",
    Kokkos::RangePolicy<ExecSpace>(0, m_.nBoundaryFaces()),
    KOKKOS_LAMBDA(const Index f, Index& a) {
      a += (pt(f) == static_cast<int>(PressureBC::FixedValue)) ? 1 : 0;
    }, open);
  openDomain_ = comm_.sum(open) > 0;
}

void PisoSolver::sync(const ScalarField& f) const {
  if (const auto* hx = m_.halo()) hx->exchange(f);
}
void PisoSolver::sync(const VectorField& f) const {
  if (const auto* hx = m_.halo()) hx->exchange(f);
}

void PisoSolver::setState(const VectorField& u, const ScalarField& p,
                          const ScalarField& F) {
  Kokkos::deep_copy(u_, u); Kokkos::deep_copy(uOld_, u); Kokkos::deep_copy(uOld2_, u);
  Kokkos::deep_copy(p_, p);
  Kokkos::deep_copy(F_, F); Kokkos::deep_copy(FOld_, F);

  // On a FixedValue face the mass flux is a STATE VARIABLE -- the caller does
  // not prescribe it, the pressure equation solves for it -- so an initial
  // state that leaves it at zero is inconsistent with the velocity field the
  // caller just supplied. The momentum matrix is assembled before the first
  // pressure solve, so that first assembly sees a cell whose fluxes do not
  // sum to zero and picks up a spurious source worth the entire outlet mass
  // flow. Seed it the way a zero-gradient outlet is defined: the cell value
  // carried to the face.
  //
  // Requires setPressureBoundary to have been called first, which is the
  // natural order anyway; without it there is no outlet to seed.
  sync(u_); sync(uOld_); sync(uOld2_); sync(p_);

  if (!openDomain_) return;
  const Index nb = m_.nBoundaryFaces();
  auto bc = m_.boundaryCell(); auto bar = m_.boundaryArea();
  auto pt = pType_; auto fb = Fb_; auto uu = u_;
  Kokkos::parallel_for("fbSeed", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      if (pt(f) != static_cast<int>(PressureBC::FixedValue)) return;
      Real s = 0.0;
      for (int i = 0; i < 3; ++i) s += uu(bc(f), i) * bar(f, i);
      fb(f) = s;
    });
  Kokkos::fence();
}

void PisoSolver::bdf(Real& aP, Real& a1, Real& a2) const {
  if (step_ == 0) { aP = 1.0 / dt_; a1 = -1.0 / dt_; a2 = 0.0; return; }
  aP = 1.5 / dt_; a1 = -2.0 / dt_; a2 = 0.5 / dt_;
}

ScalarField PisoSolver::pressureBoundary(const ScalarField& p) const {
  // Extrapolate from the interior gradient, iterated. Using the cell value
  // (zero normal gradient) is wrong: the wall normal pressure gradient is
  // whatever the momentum equation requires there.
  //
  // The extrapolation deliberately uses the LINEAR least-squares gradient. A
  // quadratic fit is more accurate inside its stencil and less accurate
  // outside it, and using it here dropped the measured velocity order from
  // 1.44 to 1.29.
  //
  // Extrapolation is for FixedFlux faces only. Where the pressure is
  // PRESCRIBED the answer is already known, and extrapolating there makes the
  // solver impose two different outlet conditions at once: the pressure
  // equation drives the flux towards p_out through its boundary term while
  // the gradient operator and the force integral read an extrapolated value
  // that need not equal it. With a constant pressure field the two agree by
  // accident, which is why a uniform-flow check cannot see this.
  Stopwatch _sw(&t_.boundaryP);
  ++t_.boundaryPCalls;
  const Index nb = m_.nBoundaryFaces();
  ScalarField v("pb", nb);
  auto bc = m_.boundaryCell(); auto bcen = m_.boundaryCentre(); auto cc = m_.cellCentre();
  auto pt = pType_; auto pvals = pValue_;
  const bool open = openDomain_;
  Kokkos::parallel_for("pb0", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      v(f) = (open && pt(f) == static_cast<int>(PressureBC::FixedValue))
                 ? pvals(f) : p(bc(f));
    });
  Kokkos::fence();
  VectorField g("gpb", m_.nTotal(), 3);
  for (int it = 0; it < 3; ++it) {
    grad_(p, v, g);
    Kokkos::parallel_for("pbIt", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) {
        if (open && pt(f) == static_cast<int>(PressureBC::FixedValue)) return;
        Real acc = 0.0;
        for (int i = 0; i < 3; ++i) acc += g(bc(f), i) * (bcen(f, i) - cc(bc(f), i));
        v(f) = p(bc(f)) + acc;
      });
    Kokkos::fence();
  }
  return v;
}

void PisoSolver::gradP(const ScalarField& p, VectorField& g) const {
  const ScalarField pb = pressureBoundary(p);
  Stopwatch _sw(&t_.gradient);
  ++t_.gradCalls;
  grad_(p, pb, g);
}

void PisoSolver::assembleMomentum(const VectorField& uB, const VectorField& src) {
  Stopwatch _sw(&t_.assemble);
  const Index nc = m_.nCells(), nt = m_.nTotal();
  const Index nf = m_.nInternalFaces(), nb = m_.nBoundaryFaces();
  auto own = m_.owner(); auto nei = m_.neighbour(); auto bc = m_.boundaryCell();
  auto vol = m_.cellVolume();
  auto a = diff_.aInt(); auto ab = diff_.aBnd(); auto wo = diff_.wOwner();
  auto k = diff_.kInt(); auto kb = diff_.kBnd();
  auto diag = diag_; auto up = upper_; auto lo = lower_;
  auto w = w_; auto sk = skew_; auto F = F_; auto Fb = Fb_;
  auto b = bSrc_; auto u = u_; auto uo = uOld_; auto uo2 = uOld2_;
  const Real nu = nu_;
  Real aPt, a1, a2; bdf(aPt, a1, a2);
  (void)nc;

  Kokkos::deep_copy(diag, 0.0);
  Kokkos::parallel_for("mtrans", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) { diag(c) = aPt * vol(c); });
  Kokkos::fence();
  Kokkos::parallel_for("mint", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      const Real Fp = Kokkos::max(F(f), 0.0), Fn = Kokkos::max(-F(f), 0.0);
      Kokkos::atomic_add(&diag(own(f)), nu * a(f) + Fp);
      Kokkos::atomic_add(&diag(nei(f)), nu * a(f) + Fn);
      up(f) = -nu * a(f) - Fn;
      lo(f) = -nu * a(f) - Fp;
    });
  auto bt = bcType_;
  auto FbA = Fb_;
  Kokkos::parallel_for("mbnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      if (bt(f) == static_cast<int>(VelocityBC::Dirichlet)) {
        Kokkos::atomic_add(&diag(bc(f)), nu * ab(f));
      } else {
        // Zero gradient: the face value IS the cell value, so the convective
        // flux through it is implicit. Leaving it on the right-hand side makes
        // outflow an explicit source that feeds itself -- the cylinder case
        // reached a continuity residual of 1.7 and a lift amplitude of 15
        // within a hundred steps. Backflow (Fb < 0) would give a negative
        // diagonal contribution, so only the outflow part goes implicit and
        // any inflow is handled explicitly against the cell value.
        Kokkos::atomic_add(&diag(bc(f)), Kokkos::max(FbA(f), 0.0));
      }
    });
  Kokkos::fence();
  Kokkos::deep_copy(aP_, diag);
  // A ghost's diagonal only accumulated the faces THIS rank stores, so it is
  // a partial sum. Df_ interpolates aP to the face and Rhie-Chow reads it on
  // both sides, so the partial value would bias every rank-boundary flux.
  sync(aP_);

  // Gradients of each velocity component, for the skewness and non-orthogonal
  // corrections.
  VectorField g0("g0", nt, 3), g1("g1", nt, 3), g2("g2", nt, 3);
  VectorField* gs[3] = {&g0, &g1, &g2};
  for (int d = 0; d < 3; ++d) {
    ScalarField comp("comp", nt), compB("compB", nb);
    Kokkos::parallel_for("ex", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { comp(c) = u(c, d); });
    Kokkos::parallel_for("exb", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) {
        // A zero-gradient face's value is the CELL's, not the caller's uB,
        // which is meaningless there and is usually left at zero. Feeding
        // that zero to the gradient operator invents a velocity gradient of
        // order u/h along every slip plane and outlet. It is invisible on a
        // Cartesian mesh, where the corrections this gradient feeds are
        // identically zero, and it is not invisible on a distorted one.
        compB(f) = (bt(f) == static_cast<int>(VelocityBC::ZeroGradient))
                       ? u(bc(f), d) : uB(f, d);
      });
    Kokkos::fence();
    grad_(comp, compB, *gs[d]);
  }

  Kokkos::parallel_for("msrcVol", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) {
      for (int d = 0; d < 3; ++d)
        b(c, d) = src(c, d) * vol(c) - (a1 * uo(c, d) + a2 * uo2(c, d)) * vol(c);
    });
  Kokkos::fence();

  auto G0 = g0, G1 = g1, G2 = g2;
  Kokkos::parallel_for("msrcInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      for (int d = 0; d < 3; ++d) {
        Real gf[3];
        for (int i = 0; i < 3; ++i) {
          const Real go = d == 0 ? G0(own(f), i) : d == 1 ? G1(own(f), i) : G2(own(f), i);
          const Real gn = d == 0 ? G0(nei(f), i) : d == 1 ? G1(nei(f), i) : G2(nei(f), i);
          gf[i] = w(f) * go + (1.0 - w(f)) * gn;
        }
        Real nonorth = 0.0, ho = 0.0, gfo[3];
        for (int i = 0; i < 3; ++i) {
          const Real go = d == 0 ? G0(own(f), i) : d == 1 ? G1(own(f), i) : G2(own(f), i);
          const Real gn = d == 0 ? G0(nei(f), i) : d == 1 ? G1(nei(f), i) : G2(nei(f), i);
          gfo[i] = wo(f) * go + (1.0 - wo(f)) * gn;
          nonorth += k(f, i) * gfo[i];
          ho += gf[i] * sk(f, i);
        }
        ho += w(f) * u(own(f), d) + (1.0 - w(f)) * u(nei(f), d);
        const Real ud = F(f) > 0.0 ? u(own(f), d) : u(nei(f), d);
        const Real dc = F(f) * (ho - ud);
        Kokkos::atomic_add(&b(own(f), d), nu * nonorth - dc);
        Kokkos::atomic_add(&b(nei(f), d), -nu * nonorth + dc);
      }
    });
  Kokkos::parallel_for("msrcBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      const bool dirichlet = bt(f) == static_cast<int>(VelocityBC::Dirichlet);
      for (int d = 0; d < 3; ++d) {
        if (dirichlet) {
          Real nonorth = 0.0;
          for (int i = 0; i < 3; ++i) {
            const Real gi = d == 0 ? G0(bc(f), i) : d == 1 ? G1(bc(f), i) : G2(bc(f), i);
            nonorth += kb(f, i) * gi;
          }
          Kokkos::atomic_add(&b(bc(f), d),
                             nu * (ab(f) * uB(f, d) + nonorth) - Fb(f) * uB(f, d));
        } else {
          // No diffusive flux. The outflow part of the convective flux is in
          // the matrix; only backflow is left here.
          const Real back = Kokkos::min(Fb(f), 0.0);
          Kokkos::atomic_add(&b(bc(f), d), -back * u(bc(f), d));
        }
      }
    });
  Kokkos::fence();
}

void PisoSolver::computeHbyA() {
  Stopwatch _sw(&t_.hbya);
  // H = b - (A - diag) u, recomputed from the CURRENT u. Recomputing this is
  // the entire point of a second PISO corrector.
  const Index nt = m_.nTotal(), nf = m_.nInternalFaces();
  auto own = m_.owner(); auto nei = m_.neighbour();
  auto H = HbyA_; auto u = u_; auto b = bSrc_;
  auto up = upper_; auto lo = lower_; auto aP = aP_;

  Kokkos::parallel_for("Hinit", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) {
      for (int d = 0; d < 3; ++d) H(c, d) = b(c, d);
    });
  Kokkos::fence();
  Kokkos::parallel_for("Hface", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      for (int d = 0; d < 3; ++d) {
        Kokkos::atomic_add(&H(own(f), d), -up(f) * u(nei(f), d));
        Kokkos::atomic_add(&H(nei(f), d), -lo(f) * u(own(f), d));
      }
    });
  Kokkos::fence();
  Kokkos::parallel_for("Hscale", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) {
      for (int d = 0; d < 3; ++d) H(c, d) /= aP(c);
    });
  Kokkos::fence();
  // Hface only summed this rank's faces into a ghost, so ghost H is a partial
  // sum. rhieChow interpolates H/aP to every face, including rank boundaries.
  sync(HbyA_);

  // H/aP is interpolated to the face in rhieChow, and that face value SETS
  // the mass flux. Plain linear interpolation is first order once the face
  // centre sits off the line joining the two cell centres, so it needs the
  // same skewness correction every other face value gets -- which needs the
  // gradient of each component.
  const Index nb = m_.nBoundaryFaces();
  VectorField* gs[3] = {&gH0_, &gH1_, &gH2_};
  for (int d = 0; d < 3; ++d) {
    ScalarField comp("Hcomp", nt), compB("HcompB", nb);
    auto bc = m_.boundaryCell();
    Kokkos::parallel_for("Hex", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { comp(c) = H(c, d); });
    Kokkos::parallel_for("Hexb", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) { compB(f) = H(bc(f), d); });
    Kokkos::fence();
    grad_(comp, compB, *gs[d]);
  }
}

void PisoSolver::rhieChow() {
  Stopwatch _sw(&t_.rhieChow);
  const Index nf = m_.nInternalFaces();
  auto own = m_.owner(); auto nei = m_.neighbour();
  auto fa = m_.faceArea(); auto vol = m_.cellVolume();
  auto w = w_; auto aP = aP_; auto Df = Df_; auto Fstar = Fstar_; auto FOld = FOld_;
  auto H = HbyA_; auto uo = uOld_; auto gp = gp_;
  auto sk = skew_; auto GH0 = gH0_; auto GH1 = gH1_; auto GH2 = gH2_;
  auto ap = pdiff_.aInt();
  auto p = p_;
  Real aPt, a1, a2; bdf(aPt, a1, a2);
  const bool consistent = ctl_.consistentRhieChow;

  Kokkos::parallel_for("rhieChow", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      const Real D = w(f) * (vol(own(f)) / aP(own(f)))
                   + (1.0 - w(f)) * (vol(nei(f)) / aP(nei(f)));
      Df(f) = D;
      Real flux = 0.0, gpf = 0.0, ufOld = 0.0;
      for (int i = 0; i < 3; ++i) {
        Real Hf = w(f) * H(own(f), i) + (1.0 - w(f)) * H(nei(f), i);
        for (int j = 0; j < 3; ++j) {
          const Real go = i == 0 ? GH0(own(f), j) : i == 1 ? GH1(own(f), j) : GH2(own(f), j);
          const Real gn = i == 0 ? GH0(nei(f), j) : i == 1 ? GH1(nei(f), j) : GH2(nei(f), j);
          Hf += (w(f) * go + (1.0 - w(f)) * gn) * sk(f, j);
        }
        flux += Hf * fa(f, i);
        gpf += (w(f) * gp(own(f), i) + (1.0 - w(f)) * gp(nei(f), i)) * fa(f, i);
        ufOld += (w(f) * uo(own(f), i) + (1.0 - w(f)) * uo(nei(f), i)) * fa(f, i);
      }
      const Real snGrad = ap(f) * (p(nei(f)) - p(own(f)));
      flux += D * (gpf - snGrad);
      if (consistent) {
        // Old-flux term (Choi 1999): carrying the Rhie-Chow residual forward
        // with coefficient Df*aP_t cancels the transient part of aP, so the
        // pressure damping does not vanish as dt shrinks.
        flux += D * aPt * (FOld(f) - ufOld);
      }
      Fstar(f) = flux;
    });
  Kokkos::fence();

  // Predicted flux through an outlet: H/aP extrapolated to the face. The
  // pressure solve then corrects it, exactly as it corrects an internal face.
  if (openDomain_) {
    auto bc = m_.boundaryCell(); auto bar = m_.boundaryArea();
    auto fbs = FbStar_; auto pt = pType_; auto fb = Fb_;
    auto Hb = HbyA_;
    Kokkos::parallel_for("rcBnd", Kokkos::RangePolicy<ExecSpace>(0, m_.nBoundaryFaces()),
      KOKKOS_LAMBDA(const Index f) {
        if (pt(f) != static_cast<int>(PressureBC::FixedValue)) { fbs(f) = fb(f); return; }
        Real s = 0.0;
        for (int i = 0; i < 3; ++i) s += Hb(bc(f), i) * bar(f, i);
        fbs(f) = s;
      });
    Kokkos::fence();
  }
}

void PisoSolver::solvePressure(LinearSolver& solver) {
  Stopwatch _sw(&t_.pressureAssembly);
  const Index nc = m_.nCells(), nt = m_.nTotal(), nf = m_.nInternalFaces();
  const Index nb = m_.nBoundaryFaces();
  auto own = m_.owner(); auto nei = m_.neighbour(); auto bc = m_.boundaryCell();
  auto ap = pdiff_.aInt(); auto kInt = pdiff_.kInt();
  auto w = w_; auto Df = Df_; auto Fstar = Fstar_; auto Fb = Fb_; auto F = F_;
  auto p = p_;
  // FbStar_ is only populated on the open path; a closed domain has no
  // solved boundary flux and Fb_ is the whole story.
  auto Fbs = openDomain_ ? FbStar_ : Fb_;
  auto pvals = pValue_;
  const bool open = openDomain_;

  LinearSystem sys(m_);
  sys.zero();
  auto diag = sys.diag(); auto up = sys.upper(); auto lo = sys.lower();
  Kokkos::parallel_for("pmat", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      const Real a = ap(f) * Df(f);
      Kokkos::atomic_add(&diag(own(f)), a);
      Kokkos::atomic_add(&diag(nei(f)), a);
      up(f) = -a; lo(f) = -a;
    });
  auto apb = pdiff_.aBnd(); auto pt = pType_; auto aPv = aP_;
  auto volAll = m_.cellVolume();
  if (openDomain_) {
    Kokkos::parallel_for("pmatBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) {
        if (pt(f) != static_cast<int>(PressureBC::FixedValue)) return;
        const Real Dfb = volAll(bc(f)) / aPv(bc(f));
        Kokkos::atomic_add(&diag(bc(f)), apb(f) * Dfb);
      });
  }
  Kokkos::fence();
  // The coefficients are now fixed for every sweep below, so tell the backend
  // once. A backend that builds an AMG hierarchy would otherwise rebuild it
  // 20-40 times per corrector for nothing.
  solver.notifyMatrixChanged();

  ScalarField nonorth("nonorth", nf);
  VectorField g("gp", nt, 3);
  Kokkos::deep_copy(p, 0.0);
  auto src = sys.source();
  int sweeps = 0;

  for (int k = 0; k < ctl_.nonOrthCorrectors; ++k) {
    Kokkos::deep_copy(src, 0.0);
    Kokkos::parallel_for("pdivInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
      KOKKOS_LAMBDA(const Index f) {
        const Real q = Fstar(f) - nonorth(f);
        Kokkos::atomic_add(&src(own(f)), -q);
        Kokkos::atomic_add(&src(nei(f)), q);
      });
    // The right-hand side must contain the flux that the correction step
    // will actually correct. On a FixedValue face that is FbStar, not Fb:
    // Fb still holds last step's solved outlet flux, while the correction
    // updates FbStar. Subtracting one and adding the other leaves every
    // outlet cell with a divergence of exactly FbStar -- the whole outlet
    // mass flow, constant from the first step, which is what this looked
    // like in the cylinder wake. On a FixedFlux face the two arrays are
    // equal by construction (see rcBnd), so FbStar is right everywhere.
    //
    // The prescribed pressure enters here too. The matrix carries the
    // diagonal apb*Dfb; without the matching apb*Dfb*p_out on the source the
    // solver quietly imposes p_out = 0 whatever the caller asked for, and a
    // case that prescribes zero cannot tell the difference.
    Kokkos::parallel_for("pdivBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) {
        Kokkos::atomic_add(&src(bc(f)), -Fbs(f));
        // pType_ is an empty view on a closed domain, so it must not be
        // indexed there.
        if (open && pt(f) == static_cast<int>(PressureBC::FixedValue)) {
          const Real Dfb = volAll(bc(f)) / aPv(bc(f));
          Kokkos::atomic_add(&src(bc(f)), apb(f) * Dfb * pvals(f));
        }
      });
    // With Dirichlet velocity everywhere the pressure operator is pure
    // Neumann and singular, its null space the constants. CG stalls on a
    // right-hand side that has a component along that null space, so project
    // it out first. Pinning a cell instead would break the symmetry CG needs.
    // One FixedValue face removes the null space, and projecting then would
    // shift the answer away from the prescribed level.
    Real rmean = 0.0;
    if (!openDomain_) {
    Kokkos::parallel_reduce("rmean", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& a) { a += src(c); }, rmean);
    rmean = comm_.sum(rmean) / static_cast<Real>(comm_.sum(nc));
    Kokkos::parallel_for("rproj", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c) { src(c) -= rmean; });
    Kokkos::fence();
    }

    // A p = -div(F*): solving A p = +div doubles the divergence.
    solver.solve(sys, p, 1e-14, 1e-20, 5000);

    // Remove the constant null-space component instead of pinning a cell:
    // pinning makes the matrix non-symmetric and breaks CG. With an outlet the
    // level is set by the boundary and must not be shifted.
    if (!openDomain_) {
      Real mean = 0.0;
      Kokkos::parallel_reduce("pmean", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c, Real& a) { a += p(c); }, mean);
      mean = comm_.sum(mean) / static_cast<Real>(comm_.sum(nc));
      Kokkos::parallel_for("pshift", Kokkos::RangePolicy<ExecSpace>(0, nt),
        KOKKOS_LAMBDA(const Index c) { p(c) -= mean; });
      Kokkos::fence();
    }

    gradP(p, g);
    Real delta = 0.0, scale = 1e-300;
    auto no = nonorth;
    Kokkos::parallel_reduce("nonorthUpd", Kokkos::RangePolicy<ExecSpace>(0, nf),
      KOKKOS_LAMBDA(const Index f, Real& acc) {
        Real v = 0.0;
        for (int i = 0; i < 3; ++i)
          v += kInt(f, i) * (w(f) * g(own(f), i) + (1.0 - w(f)) * g(nei(f), i));
        v *= Df(f);
        // Under-relax the deferred correction. Undamped, the fixed-point
        // iteration stalled on the coarsest distorted mesh -- 40 sweeps and
        // a continuity residual of 1.6e-7, while the finer meshes converged
        // in 20 sweeps to 1e-14.
        v = no(f) + 0.7 * (v - no(f));
        acc = Kokkos::max(acc, Kokkos::abs(v - no(f)));
        no(f) = v;
      }, Kokkos::Max<Real>(delta));
    Kokkos::parallel_reduce("fscale", Kokkos::RangePolicy<ExecSpace>(0, nf),
      KOKKOS_LAMBDA(const Index f, Real& acc) {
        acc = Kokkos::max(acc, Kokkos::abs(Fstar(f)));
      }, Kokkos::Max<Real>(scale));
    sweeps = k + 1;
    // The threshold has to sit above the linear solver's own noise floor.
    // At 1e-14 the loop chased residual noise, used all 40 sweeps, and still
    // reported 8.9e-9 continuity error.
    if (comm_.max(delta) < ctl_.nonOrthTol * scale) break;
  }
  lastNonOrth_ = sweeps;

  Kokkos::parallel_for("fluxCorrect", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      F(f) = Fstar(f) - ap(f) * Df(f) * (p(nei(f)) - p(own(f))) - nonorth(f);
    });
  if (openDomain_) {
    auto fbs = FbStar_; auto pv = pValue_; auto fbOut = Fb_;
    Kokkos::parallel_for("fluxCorrectBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) {
        if (pt(f) != static_cast<int>(PressureBC::FixedValue)) return;
        const Real Dfb = volAll(bc(f)) / aPv(bc(f));
        fbOut(f) = fbs(f) - apb(f) * Dfb * (pv(f) - p(bc(f)));
      });
  }
  Kokkos::fence();
}

Real PisoSolver::continuityError(const ScalarField& F, const ScalarField& Fb) const {
  ScalarField div("div", m_.nTotal());
  auto own = m_.owner(); auto nei = m_.neighbour(); auto bc = m_.boundaryCell();
  Kokkos::parallel_for("cInt", Kokkos::RangePolicy<ExecSpace>(0, m_.nInternalFaces()),
    KOKKOS_LAMBDA(const Index f) {
      Kokkos::atomic_add(&div(own(f)), F(f));
      Kokkos::atomic_add(&div(nei(f)), -F(f));
    });
  Kokkos::parallel_for("cBnd", Kokkos::RangePolicy<ExecSpace>(0, m_.nBoundaryFaces()),
    KOKKOS_LAMBDA(const Index f) { Kokkos::atomic_add(&div(bc(f)), Fb(f)); });
  Kokkos::fence();
  Real m = 0.0;
  Kokkos::parallel_reduce("cMax", Kokkos::RangePolicy<ExecSpace>(0, m_.nCells()),
    KOKKOS_LAMBDA(const Index c, Real& a) { a = Kokkos::max(a, Kokkos::abs(div(c))); },
    Kokkos::Max<Real>(m));
  return comm_.max(m);
}

Vec3 PisoSolver::boundaryForce(const View1<int>& mask) const {
  const Index nb = m_.nBoundaryFaces(), nt = m_.nTotal();
  auto bc = m_.boundaryCell(); auto bar = m_.boundaryArea();
  auto ab = diff_.aBnd(); auto kb = diff_.kBnd();
  auto u = u_;
  const ScalarField pb = pressureBoundary(p_);
  const Real nu = nu_;

  // Velocity gradients for the non-orthogonal part of the wall stress.
  VectorField g0("g0", nt, 3), g1("g1", nt, 3), g2("g2", nt, 3);
  VectorField* gs[3] = {&g0, &g1, &g2};
  for (int d = 0; d < 3; ++d) {
    ScalarField comp("comp", nt), compB("compB", nb);
    Kokkos::parallel_for("fex", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { comp(c) = u(c, d); });
    // The same boundary values the momentum equation used. Hard-coding a
    // no-slip wall here was right for the cavity, where every boundary is
    // one, and wrong for any case with an inlet, an outlet or a slip plane:
    // it reconstructs a different velocity field from the one being solved,
    // and the force it reports is the force in that other field.
    auto bt = bcType_; auto ubv = uBnd_; auto bcl = m_.boundaryCell();
    Kokkos::parallel_for("fexb", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) {
        compB(f) = (bt(f) == static_cast<int>(VelocityBC::ZeroGradient))
                       ? u(bcl(f), d) : ubv(f, d);
      });
    Kokkos::fence();
    grad_(comp, compB, *gs[d]);
  }
  auto G0 = g0, G1 = g1, G2 = g2;

  Real fx = 0.0, fy = 0.0, fz = 0.0;
  auto accumulate = [&](int d, const VectorField& G) {
    Real acc = 0.0;
    Kokkos::parallel_reduce("force", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f, Real& a) {
        if (!mask(f)) return;
        // Pressure acts along the outward normal; the viscous term is the
        // diffusive flux the momentum equation applies through this face,
        // with the sign flipped to give the force ON the body.
        Real visc = ab(f) * (0.0 - u(bc(f), d));
        for (int i = 0; i < 3; ++i) visc += kb(f, i) * G(bc(f), i);
        a += pb(f) * bar(f, d) - nu * visc;
      }, acc);
    return acc;
  };
  fx = accumulate(0, G0); fy = accumulate(1, G1); fz = accumulate(2, G2);
  return {comm_.sum(fx), comm_.sum(fy), comm_.sum(fz)};
}

StepReport PisoSolver::advance(const VectorField& uB, const ScalarField& fB,
                               const VectorField& src, LinearSolver& momentumSolver,
                               LinearSolver& pressureSolver) {
  const Index nc = m_.nCells(), nt = m_.nTotal();
  auto vol = m_.cellVolume();
  if (openDomain_) {
    // Take the caller's flux only where it is prescribed. On a FixedValue
    // face the flux is part of the solution, and copying the input over it
    // resets the outlet to zero at the start of every step -- the inflow then
    // has nowhere to go and the continuity residual sits at O(0.1).
    auto fbIn = fB; auto fbOut = Fb_; auto pt = pType_;
    Kokkos::parallel_for("fbIn", Kokkos::RangePolicy<ExecSpace>(0, m_.nBoundaryFaces()),
      KOKKOS_LAMBDA(const Index f) {
        if (pt(f) != static_cast<int>(PressureBC::FixedValue)) fbOut(f) = fbIn(f);
      });
    Kokkos::fence();
  } else {
    Kokkos::deep_copy(Fb_, fB);
  }
  // Time levels shift ONCE per step, not once per outer iteration.
  Kokkos::deep_copy(uOld2_, uOld_);
  Kokkos::deep_copy(uOld_, u_);
  Kokkos::deep_copy(FOld_, F_);

  Kokkos::deep_copy(uBnd_, uB);
  // Everything downstream reads u at ghost cells: the convection matrix, the
  // deferred correction, the velocity gradients.
  sync(u_); sync(p_);

  Stopwatch _sw(&t_.total);
  StepReport rep;
  VectorField uPrev("uPrev", nt, 3);

  for (int outer = 0; outer < ctl_.outer; ++outer) {
    Kokkos::deep_copy(uPrev, u_);
    assembleMomentum(uB, src);
    gradP(p_, gp_);

    // Momentum predictor, one component at a time.
    LinearSystem sys(m_);
    Kokkos::deep_copy(sys.diag(), diag_);
    Kokkos::deep_copy(sys.upper(), upper_);
    Kokkos::deep_copy(sys.lower(), lower_);
    // One matrix, three component solves.
    momentumSolver.notifyMatrixChanged();
    for (int d = 0; d < 3; ++d) {
      auto rhs = sys.source(); auto b = bSrc_; auto g = gp_;
      Kokkos::parallel_for("mrhs", Kokkos::RangePolicy<ExecSpace>(0, nt),
        KOKKOS_LAMBDA(const Index c) { rhs(c) = b(c, d) - g(c, d) * vol(c); });
      Kokkos::fence();
      ScalarField x("x", nt);
      auto u = u_;
      Kokkos::parallel_for("seed", Kokkos::RangePolicy<ExecSpace>(0, nt),
        KOKKOS_LAMBDA(const Index c) { x(c) = u(c, d); });
      Kokkos::fence();
      momentumSolver.solve(sys, x, 1e-13, 1e-18, 5000);
      Kokkos::parallel_for("store", Kokkos::RangePolicy<ExecSpace>(0, nt),
        KOKKOS_LAMBDA(const Index c) { u(c, d) = x(c); });
      Kokkos::fence();
    }

    for (int corr = 0; corr < ctl_.correctors; ++corr) {
      computeHbyA();
      rhieChow();
      solvePressure(pressureSolver);
      gradP(p_, gp_);
      auto u = u_; auto H = HbyA_; auto g = gp_; auto aP = aP_;
      Kokkos::parallel_for("uCorrect", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c) {
          for (int d = 0; d < 3; ++d) u(c, d) = H(c, d) - g(c, d) * vol(c) / aP(c);
        });
      Kokkos::fence();
      // Correct owned cells and exchange, rather than computing ghosts from
      // ghost inputs. Both give the same answer when every input is current,
      // and only one of them keeps saying so when an input stops being.
      sync(u_);
    }

    rep.outerUsed = outer + 1;
    Real delta = 0.0, scale = 1e-300;
    auto u = u_; auto q = uPrev;
    Kokkos::parallel_reduce("outerDelta", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& acc) {
        for (int d = 0; d < 3; ++d) acc = Kokkos::max(acc, Kokkos::abs(u(c, d) - q(c, d)));
      }, Kokkos::Max<Real>(delta));
    Kokkos::parallel_reduce("outerScale", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& acc) {
        for (int d = 0; d < 3; ++d) acc = Kokkos::max(acc, Kokkos::abs(u(c, d)));
      }, Kokkos::Max<Real>(scale));
    if (comm_.max(delta) < ctl_.outerTol * comm_.max(scale)) break;
  }

  ++step_;
  rep.nonOrthSweeps = lastNonOrth_;
  rep.continuityError = continuityError(F_, Fb_);
  return rep;
}

}  // namespace nsflow
