#include "physics/Piso.hpp"
#include <chrono>
#include "linalg/LinearSolver.hpp"
#include "linalg/LinearSystem.hpp"
#include "linalg/NativeCG.hpp"
#include "mesh/Mesh.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vibeflow {
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
      nonorth_("nonorth", mesh.nInternalFaces()),
      Fstar_("Fstar", mesh.nInternalFaces()),
      skew_("skew", mesh.nInternalFaces(), 3),
      u_("u", mesh.nTotal(), 3), uOld_("uOld", mesh.nTotal(), 3),
      uOld2_("uOld2", mesh.nTotal(), 3), HbyA_("HbyA", mesh.nTotal(), 3),
      gp_("gp", mesh.nTotal(), 3),
      gH0_("gH0", mesh.nTotal(), 3), gH1_("gH1", mesh.nTotal(), 3),
      gH2_("gH2", mesh.nTotal(), 3),
      q_("q", mesh.nTotal(), 3),
      rOld_("rOld", mesh.nInternalFaces()), rOldB_("rOldB", mesh.nBoundaryFaces()),
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
  pbWarm_ = ScalarField("pbWarm", mesh.nBoundaryFaces());
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

PisoSolver::~PisoSolver() = default;

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
  pbWarmValid_ = false;     // a new pressure field: extrapolate it from scratch
  gpValid_ = false;         // and its gradient is not the one in gp_

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
  // Warm start only for the solver's own pressure: a caller extrapolating
  // some other field must not start from p_'s boundary values.
  const bool warm = ctl_.pressureExtrapolation && ctl_.pressureExtrapWarmStart &&
                    pbWarmValid_ && p.data() == p_.data();
  auto last = pbWarm_;
  Kokkos::parallel_for("pb0", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      v(f) = (open && pt(f) == static_cast<int>(PressureBC::FixedValue))
                 ? pvals(f) : (warm ? last(f) : p(bc(f)));
    });
  Kokkos::fence();
  VectorField g("gpb", m_.nTotal(), 3);
  const int sweeps = !ctl_.pressureExtrapolation ? 0
                   : warm ? 1 : ctl_.pressureExtrapSweeps;
  t_.boundaryPSweeps += sweeps;
  for (int it = 0; it < sweeps; ++it) {
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
  if (ctl_.pressureExtrapolation && ctl_.pressureExtrapWarmStart &&
      p.data() == p_.data()) {
    Kokkos::deep_copy(pbWarm_, v);
    pbWarmValid_ = true;
  }
  return v;
}

std::vector<Real> PisoSolver::extrapolationHistory(int n) const {
  const Index nb = m_.nBoundaryFaces(), nc = m_.nCells();
  auto p = p_;
  auto bc = m_.boundaryCell(); auto bcen = m_.boundaryCentre(); auto cc = m_.cellCentre();
  auto pt = pType_; auto pvals = pValue_;
  const bool open = openDomain_;
  ScalarField v("pbHist", nb), vPrev("pbHistPrev", nb);
  Kokkos::parallel_for("pbHist0", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      v(f) = (open && pt(f) == static_cast<int>(PressureBC::FixedValue)) ? pvals(f) : p(bc(f));
    });
  Real pmax = 0.0;
  Kokkos::parallel_reduce("pbHistScale", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index c, Real& m) { m = Kokkos::fmax(m, Kokkos::fabs(p(c))); },
    Kokkos::Max<Real>(pmax));
  pmax = std::max(comm_.max(pmax), 1e-300);
  VectorField g("gpbHist", m_.nTotal(), 3);
  std::vector<Real> hist;
  for (int it = 0; it < n; ++it) {
    Kokkos::deep_copy(vPrev, v);
    grad_(p, v, g);
    Kokkos::parallel_for("pbHistIt", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) {
        if (open && pt(f) == static_cast<int>(PressureBC::FixedValue)) return;
        Real acc = 0.0;
        for (int i = 0; i < 3; ++i) acc += g(bc(f), i) * (bcen(f, i) - cc(bc(f), i));
        v(f) = p(bc(f)) + acc;
      });
    Real d = 0.0;
    Kokkos::parallel_reduce("pbHistDelta", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f, Real& m) { m = Kokkos::fmax(m, Kokkos::fabs(v(f) - vPrev(f))); },
      Kokkos::Max<Real>(d));
    hist.push_back(comm_.max(d) / pmax);
  }
  return hist;
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
  auto fa = m_.faceArea();
  const Real nu = nu_;
  const bool deferred = ctl_.deferredCorrection;
  const bool linearUpwind = ctl_.convection == ConvectionScheme::LinearUpwind;
  auto fcen = m_.faceCentre(); auto ccen = m_.cellCentre();
  const bool dNonOrth = ctl_.diffusionNonOrth;
  Real aPt, a1, a2; bdf(aPt, a1, a2);
  (void)nc;
  // The eddy viscosity (ADR-042): nu + nu_t on every face. Off, the face
  // viscosity is exactly nu, as it always was.
  const bool turb = turb_;
  auto nut = nut_; auto nutB = nutB_;

  Kokkos::deep_copy(diag, 0.0);
  Kokkos::parallel_for("mtrans", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) { diag(c) = aPt * vol(c); });
  Kokkos::fence();
  Kokkos::parallel_for("mint", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      const Real Fp = Kokkos::max(F(f), 0.0), Fn = Kokkos::max(-F(f), 0.0);
      const Real nuf = turb ? nu + (w(f) * nut(own(f)) + (1.0 - w(f)) * nut(nei(f))) : nu;
      Kokkos::atomic_add(&diag(own(f)), nuf * a(f) + Fp);
      Kokkos::atomic_add(&diag(nei(f)), nuf * a(f) + Fn);
      up(f) = -nuf * a(f) - Fn;
      lo(f) = -nuf * a(f) - Fp;
    });
  auto bt = bcType_;
  auto FbA = Fb_;
  Kokkos::parallel_for("mbnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      if (bt(f) != static_cast<int>(VelocityBC::ZeroGradient)) {
        // Dirichlet, and slip: a Dirichlet face whose value is the tangential
        // part of the cell velocity (ADR-039), so the same implicit diagonal.
        const Real nub = turb ? nu + nutB(f) : nu;
        Kokkos::atomic_add(&diag(bc(f)), nub * ab(f));
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
          if (dNonOrth) nonorth += k(f, i) * gfo[i];
          ho += gf[i] * sk(f, i);
        }
        ho += w(f) * u(own(f), d) + (1.0 - w(f)) * u(nei(f), d);
        const Index upc = F(f) > 0.0 ? own(f) : nei(f);
        const Real ud = u(upc, d);
        if (linearUpwind) {
          // The upwind cell's value, extrapolated to the face centre.
          ho = ud;
          for (int i = 0; i < 3; ++i) {
            const Real gu = d == 0 ? G0(upc, i) : d == 1 ? G1(upc, i) : G2(upc, i);
            ho += gu * (fcen(f, i) - ccen(upc, i));
          }
        }
        const Real dc = deferred ? F(f) * (ho - ud) : 0.0;
        Real nuf = nu, tflux = 0.0;
        if (turb) {
          // div(nu_t grad(u)^T), component d: nu_t,f sum_j (du_j/dx_d)_f S_j.
          const Real ntf = w(f) * nut(own(f)) + (1.0 - w(f)) * nut(nei(f));
          nuf = nu + ntf;
          for (int j = 0; j < 3; ++j) {
            const Real go = j == 0 ? G0(own(f), d) : j == 1 ? G1(own(f), d) : G2(own(f), d);
            const Real gn = j == 0 ? G0(nei(f), d) : j == 1 ? G1(nei(f), d) : G2(nei(f), d);
            tflux += (w(f) * go + (1.0 - w(f)) * gn) * fa(f, j);
          }
          tflux *= ntf;
        }
        Kokkos::atomic_add(&b(own(f), d), nuf * nonorth - dc + tflux);
        Kokkos::atomic_add(&b(nei(f), d), -nuf * nonorth + dc - tflux);
      }
    });
  auto bar = m_.boundaryArea();
  Kokkos::parallel_for("msrcBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      const bool zeroGradient = bt(f) == static_cast<int>(VelocityBC::ZeroGradient);
      const bool slip = bt(f) == static_cast<int>(VelocityBC::Slip);
      // A slip face's non-orthogonal correction acts on the normal component
      // only (ADR-039): n (k . grad(u.n)). Its tangential stress is zero.
      Real n[3] = {0.0, 0.0, 0.0}, kgn = 0.0;
      if (slip && dNonOrth) {
        const Real mag = Kokkos::sqrt(bar(f,0)*bar(f,0) + bar(f,1)*bar(f,1) + bar(f,2)*bar(f,2));
        for (int i = 0; i < 3; ++i) n[i] = bar(f, i) / mag;
        for (int k2 = 0; k2 < 3; ++k2)
          kgn += kb(f, k2) * (n[0] * G0(bc(f), k2) + n[1] * G1(bc(f), k2) + n[2] * G2(bc(f), k2));
      }
      for (int d = 0; d < 3; ++d) {
        if (!zeroGradient) {
          Real nonorth = 0.0;
          if (dNonOrth) {
            if (slip) {
              nonorth = n[d] * kgn;
            } else {
              for (int i = 0; i < 3; ++i) {
                const Real gi = d == 0 ? G0(bc(f), i) : d == 1 ? G1(bc(f), i) : G2(bc(f), i);
                nonorth += kb(f, i) * gi;
              }
            }
          }
          const Real nub = turb ? nu + nutB(f) : nu;
          Real tflux = 0.0;
          if (turb && !slip) {
            // The transpose part on a Dirichlet face, from the cell gradient;
            // a slip face carries no tangential stress, so none there.
            for (int j = 0; j < 3; ++j) {
              const Real gj = j == 0 ? G0(bc(f), d) : j == 1 ? G1(bc(f), d) : G2(bc(f), d);
              tflux += gj * bar(f, j);
            }
            tflux *= nutB(f);
          }
          Kokkos::atomic_add(&b(bc(f), d),
                             nub * (ab(f) * uB(f, d) + nonorth) - Fb(f) * uB(f, d) + tflux);
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
  //
  // In the exact form (ADR-037) the vector interpolated is q = H/aP -
  // (V/aP) grad p instead -- the velocity the last pressure implies -- and
  // rhieChow adds D_f L[grad p].S back, so the product (V/aP) grad p is never
  // interpolated as a product. gp_ is grad(p_) for the p_ now stored.
  VectorField Q = HbyA_;
  if (exactOldFlux()) {
    if (!gpValid_) { gradP(p_, gp_); gpValid_ = true; }
    auto q = q_; auto gp = gp_; auto vol = m_.cellVolume();
    Kokkos::parallel_for("qInterp", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) {
        for (int d = 0; d < 3; ++d) q(c, d) = H(c, d) - gp(c, d) * vol(c) / aP(c);
      });
    Kokkos::fence();
    sync(q_);
    Q = q_;
  }
  const Index nb = m_.nBoundaryFaces();
  VectorField* gs[3] = {&gH0_, &gH1_, &gH2_};
  for (int d = 0; d < 3; ++d) {
    ScalarField comp("Hcomp", nt), compB("HcompB", nb);
    auto bc = m_.boundaryCell();
    Kokkos::parallel_for("Hex", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { comp(c) = Q(c, d); });
    Kokkos::parallel_for("Hexb", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) { compB(f) = Q(bc(f), d); });
    Kokkos::fence();
    grad_(comp, compB, *gs[d]);
  }
}

void PisoSolver::computeOldResidual() {
  // R = F_old - I[u_old].S, with the same skew-corrected interpolation I the
  // predicted flux uses (ADR-037), once per step. The gradient of u_old takes
  // the cell values as its boundary values, as the gradient of q does, so I
  // is the same linear operator on both. gH* are scratch here: computeHbyA
  // overwrites them before rhieChow reads them.
  const Index nt = m_.nTotal(), nf = m_.nInternalFaces(), nb = m_.nBoundaryFaces();
  auto uo = uOld_; auto bc = m_.boundaryCell();
  VectorField* gs[3] = {&gH0_, &gH1_, &gH2_};
  for (int d = 0; d < 3; ++d) {
    ScalarField comp("uOcomp", nt), compB("uOcompB", nb);
    Kokkos::parallel_for("uOex", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { comp(c) = uo(c, d); });
    Kokkos::parallel_for("uOexb", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) { compB(f) = uo(bc(f), d); });
    Kokkos::fence();
    grad_(comp, compB, *gs[d]);
  }
  auto own = m_.owner(); auto nei = m_.neighbour(); auto fa = m_.faceArea();
  auto w = w_; auto sk = skew_; auto FOld = FOld_; auto r = rOld_;
  auto G0 = gH0_; auto G1 = gH1_; auto G2 = gH2_;
  Kokkos::parallel_for("rOld", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      Real uf = 0.0;
      for (int i = 0; i < 3; ++i) {
        Real v = w(f) * uo(own(f), i) + (1.0 - w(f)) * uo(nei(f), i);
        for (int j = 0; j < 3; ++j) {
          const Real go = i == 0 ? G0(own(f), j) : i == 1 ? G1(own(f), j) : G2(own(f), j);
          const Real gn = i == 0 ? G0(nei(f), j) : i == 1 ? G1(nei(f), j) : G2(nei(f), j);
          v += (w(f) * go + (1.0 - w(f)) * gn) * sk(f, j);
        }
        uf += v * fa(f, i);
      }
      r(f) = FOld(f) - uf;
    });
  // An outlet face: Fb_ still holds last step's solved flux here, and the
  // interpolation to a boundary face is the cell value.
  auto rb = rOldB_;
  Kokkos::deep_copy(rb, 0.0);
  if (openDomain_) {
    auto bar = m_.boundaryArea(); auto pt = pType_; auto fb = Fb_;
    Kokkos::parallel_for("rOldB", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) {
        if (pt(f) != static_cast<int>(PressureBC::FixedValue)) return;
        Real s = 0.0;
        for (int i = 0; i < 3; ++i) s += uo(bc(f), i) * bar(f, i);
        rb(f) = fb(f) - s;
      });
  }
  Kokkos::fence();
}

void PisoSolver::rhieChow() {
  Stopwatch _sw(&t_.rhieChow);
  const Index nf = m_.nInternalFaces();
  auto own = m_.owner(); auto nei = m_.neighbour();
  auto fa = m_.faceArea(); auto vol = m_.cellVolume();
  auto w = w_; auto aP = aP_; auto Df = Df_; auto Fstar = Fstar_; auto FOld = FOld_;
  const bool exact = exactOldFlux();
  auto H = exact ? q_ : HbyA_; auto uo = uOld_; auto gp = gp_; auto rOld = rOld_;
  auto sk = skew_; auto GH0 = gH0_; auto GH1 = gH1_; auto GH2 = gH2_;
  auto ap = pdiff_.aInt();
  auto p = p_;
  Real aPt, a1, a2; bdf(aPt, a1, a2);
  const bool consistent = ctl_.consistentRhieChow;
  const bool interpolatedForm = ctl_.rhieChowForm == RhieChowForm::Interpolated;
  const bool probing = probing_;
  auto comp = rcComp_;
  const bool balanced = energy_ && balanced_;
  auto gC = gCell_; auto rF = rFace_;

  Kokkos::parallel_for("rhieChow", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      // Exact form: V_f / aP_f, so that 1/D - aPt = aPs_f / V_f exactly.
      const Real D = exact
          ? (w(f) * vol(own(f)) + (1.0 - w(f)) * vol(nei(f)))
              / (w(f) * aP(own(f)) + (1.0 - w(f)) * aP(nei(f)))
          : w(f) * (vol(own(f)) / aP(own(f)))
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
      const Real HfS = flux;
      if (interpolatedForm) flux += D * (gpf - snGrad);
      // Exact form: the pressure comes back with the face coefficient. Like
      // the term it replaces inside I[H/aP], it is an interpolated cell
      // gradient, blind to a checkerboard -- not the compact old-pressure
      // term of the interpolated form (ADR-026).
      if (exact) flux += D * gpf;
      // Balanced buoyancy (ADR-041): the compact face residual in place of
      // the interpolated cell force, as the pressure's compact gradient
      // replaces its interpolated one.
      if (balanced) {
        Real gf = 0.0;
        for (int i = 0; i < 3; ++i)
          gf += (w(f) * gC(own(f), i) + (1.0 - w(f)) * gC(nei(f), i)) * fa(f, i);
        flux += D * (rF(f) - gf);
      }
      Real choi = 0.0;
      if (consistent) {
        // Old-flux term (Choi 1999): carrying the Rhie-Chow residual forward
        // with coefficient Df*aP_t cancels the transient part of aP, so the
        // pressure damping does not vanish as dt shrinks -- exactly only in
        // the exact form, whose residual uses the interpolation the flux
        // uses (ADR-037).
        choi = D * aPt * (exact ? rOld(f) : FOld(f) - ufOld);
        flux += choi;
      }
      Fstar(f) = flux;
      if (probing) {
        comp(f, 0) = HfS;
        comp(f, 1) = (interpolatedForm || exact) ? D * gpf : 0.0;
        comp(f, 2) = interpolatedForm ? D * snGrad : 0.0;
        comp(f, 3) = choi;
      }
    });
  Kokkos::fence();

  // Predicted flux through an outlet: H/aP extrapolated to the face. The
  // pressure solve then corrects it, exactly as it corrects an internal face.
  if (openDomain_) {
    auto bc = m_.boundaryCell(); auto bar = m_.boundaryArea();
    auto fbs = FbStar_; auto pt = pType_; auto fb = Fb_;
    auto Hb = HbyA_; auto rb = rOldB_;
    // At a boundary face q + (V/aP) grad p is H/aP exactly, so the exact
    // form's prediction is unchanged there; it adds the old-flux term with
    // the cell's own coefficient.
    const bool choiB = exact && consistent;
    Kokkos::parallel_for("rcBnd", Kokkos::RangePolicy<ExecSpace>(0, m_.nBoundaryFaces()),
      KOKKOS_LAMBDA(const Index f) {
        if (pt(f) != static_cast<int>(PressureBC::FixedValue)) { fbs(f) = fb(f); return; }
        Real s = 0.0;
        for (int i = 0; i < 3; ++i) s += Hb(bc(f), i) * bar(f, i);
        if (choiB) s += vol(bc(f)) / aP(bc(f)) * aPt * rb(f);
        fbs(f) = s;
      });
    Kokkos::fence();
  }
}

void PisoSolver::solvePressure(LinearSolver& solver) {
  Stopwatch _sw(&t_.pressureAssembly);
  gpValid_ = false;         // p_ is about to change
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

  // Warm start, both of them. The correction carries over from the last solve
  // (see the member), and the pressure keeps whatever the last solve left --
  // zeroing it threw away the best initial guess available and made the first
  // sweep of every call pay for a cold start.
  auto nonorth = nonorth_;
  VectorField g("gp", nt, 3);
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
    solver.solve(sys, p, ctl_.pressureSolveTol, 1e-20, 5000);

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
  // The last sweep's gradient is grad(p_) for the p_ just solved: nothing
  // below changes p_. Hand it over rather than have the caller recompute it.
  if (sweeps > 0) {
    Kokkos::deep_copy(gp_, g);
    gpValid_ = true;
  }

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

void PisoSolver::enableProbe(Index cell) {
  probing_ = true;
  probeCell_ = cell;
  rcComp_ = View2<Real>("rcComp", m_.nInternalFaces(), 4);
}

CellProbe PisoSolver::probe() const {
  CellProbe r;
  if (!probing_ || probeCell_ < 0) return r;
  const Index c = probeCell_;
  auto H = [](const auto& v) {
    return Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
  };
  auto own = H(m_.owner()); auto nei = H(m_.neighbour()); auto bc = H(m_.boundaryCell());
  auto fa = H(m_.faceArea()); auto ba = H(m_.boundaryArea());
  auto cc = H(m_.cellCentre()); auto vol = H(m_.cellVolume());
  auto u = H(u_); auto p = H(p_); auto aP = H(aP_); auto hb = H(HbyA_); auto gp = H(gp_);
  auto F = H(F_); auto Fs = H(Fstar_); auto no = H(nonorth_); auto Df = H(Df_);
  auto ap = H(pdiff_.aInt()); auto comp = H(rcComp_); auto Fb = H(Fb_);
  auto pb = H(pressureBoundary(p_));
  auto ubt = H(bcType_); auto pbt = H(pType_);

  r.cell = c;
  for (int i = 0; i < 3; ++i) {
    r.x[i] = cc(c, i); r.u[i] = u(c, i); r.HbyA[i] = hb(c, i); r.gp[i] = gp(c, i);
  }
  r.vol = vol(c); r.p = p(c); r.aP = aP(c); r.VbyAP = vol(c) / aP(c);
  for (int i = 0; i < 3; ++i) r.corr[i] = gp(c, i) * r.VbyAP;

  auto mag = [](Real a, Real b, Real d) { return std::sqrt(a * a + b * b + d * d); };
  for (Index f = 0; f < m_.nInternalFaces(); ++f) {
    if (own(f) != c && nei(f) != c) continue;
    const Real s = own(f) == c ? 1.0 : -1.0;
    FaceProbe q;
    q.face = f; q.boundary = false;
    q.other = own(f) == c ? nei(f) : own(f);
    q.F = s * F(f); q.Fstar = s * Fs(f);
    q.HfS = s * comp(f, 0); q.Dgpf = s * comp(f, 1);
    q.DsnOld = s * comp(f, 2); q.choi = s * comp(f, 3);
    q.DsnNew = s * ap(f) * Df(f) * (p(nei(f)) - p(own(f)));
    q.nonorth = s * no(f);
    q.area = mag(fa(f, 0), fa(f, 1), fa(f, 2));
    for (int i = 0; i < 3; ++i) q.uOther[i] = u(q.other, i);
    q.pOther = p(q.other);
    r.faces.push_back(q);
    r.divF += q.F;
  }
  for (Index f = 0; f < m_.nBoundaryFaces(); ++f) {
    if (bc(f) != c) continue;
    FaceProbe q;
    q.face = f; q.boundary = true;
    q.F = Fb(f); q.Fstar = Fb(f);
    q.area = mag(ba(f, 0), ba(f, 1), ba(f, 2));
    q.pBnd = pb(f);
    q.uBC = ubt(f); q.pBC = pbt(f);
    r.faces.push_back(q);
    r.divF += q.F;
  }
  return r;
}

Real PisoSolver::courant() const {
  ScalarField s("sumF", m_.nTotal());
  auto own = m_.owner(); auto nei = m_.neighbour(); auto bc = m_.boundaryCell();
  auto F = F_; auto Fb = Fb_; auto vol = m_.cellVolume();
  Kokkos::parallel_for("coInt", Kokkos::RangePolicy<ExecSpace>(0, m_.nInternalFaces()),
    KOKKOS_LAMBDA(const Index f) {
      Kokkos::atomic_add(&s(own(f)), Kokkos::abs(F(f)));
      Kokkos::atomic_add(&s(nei(f)), Kokkos::abs(F(f)));
    });
  Kokkos::parallel_for("coBnd", Kokkos::RangePolicy<ExecSpace>(0, m_.nBoundaryFaces()),
    KOKKOS_LAMBDA(const Index f) { Kokkos::atomic_add(&s(bc(f)), Kokkos::abs(Fb(f))); });
  Kokkos::fence();
  const Real dt = dt_;
  Real co = 0.0;
  Kokkos::parallel_reduce("coMax", Kokkos::RangePolicy<ExecSpace>(0, m_.nCells()),
    KOKKOS_LAMBDA(const Index c, Real& a) {
      a = Kokkos::max(a, 0.5 * dt * s(c) / vol(c));
    }, Kokkos::Max<Real>(co));
  return comm_.max(co);
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
  auto bt = bcType_; auto ubv = uBnd_;
  auto accumulate = [&](int d) {
    Real acc = 0.0;
    Kokkos::parallel_reduce("force", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f, Real& a) {
        if (!mask(f)) return;
        // Pressure acts along the outward normal; the viscous term is the
        // diffusive flux the momentum equation applies through this face,
        // with the sign flipped to give the force ON the body: against the
        // face's own velocity (zero on a fixed wall), none through a
        // zero-gradient face, the normal component alone through a slip face.
        Real visc = 0.0;
        if (bt(f) == static_cast<int>(VelocityBC::Dirichlet)) {
          visc = ab(f) * (ubv(f, d) - u(bc(f), d));
          for (int i = 0; i < 3; ++i)
            visc += kb(f, i) * (d == 0 ? G0(bc(f), i) : d == 1 ? G1(bc(f), i) : G2(bc(f), i));
        } else if (bt(f) == static_cast<int>(VelocityBC::Slip)) {
          const Real mag = Kokkos::sqrt(bar(f,0)*bar(f,0) + bar(f,1)*bar(f,1) + bar(f,2)*bar(f,2));
          const Real n0 = bar(f,0) / mag, n1 = bar(f,1) / mag, n2 = bar(f,2) / mag;
          Real kgn = 0.0;
          for (int k2 = 0; k2 < 3; ++k2)
            kgn += kb(f, k2) * (n0 * G0(bc(f), k2) + n1 * G1(bc(f), k2) + n2 * G2(bc(f), k2));
          visc = ab(f) * (ubv(f, d) - u(bc(f), d)) + (d == 0 ? n0 : d == 1 ? n1 : n2) * kgn;
        }
        a += pb(f) * bar(f, d) - nu * visc;
      }, acc);
    return acc;
  };
  fx = accumulate(0); fy = accumulate(1); fz = accumulate(2);
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
  if (energy_) {
    Kokkos::deep_copy(TOld2_, TOld_);
    Kokkos::deep_copy(TOld_, T_);
  }
  if (turb_) {
    Kokkos::deep_copy(kOld2_, kOld_); Kokkos::deep_copy(kOld_, k_);
    Kokkos::deep_copy(wOld2_, wOld_); Kokkos::deep_copy(wOld_, w_t_);
  }

  Kokkos::deep_copy(uBnd_, uB);
  // Everything downstream reads u at ghost cells: the convection matrix, the
  // deferred correction, the velocity gradients. T likewise, in the energy
  // equation's convection and gradient.
  sync(u_); sync(p_);
  if (energy_) sync(T_);
  if (exactOldFlux() && ctl_.consistentRhieChow) { sync(uOld_); computeOldResidual(); }

  Stopwatch _sw(&t_.total);
  StepReport rep;
  VectorField uPrev("uPrev", nt, 3);

  VectorField uSweep("uSweep", nt, 3);
  ScalarField TPrev;
  if (energy_) TPrev = ScalarField("TPrev", nt);
  ScalarField kPrev, wPrev;
  if (turb_) { kPrev = ScalarField("kPrev", nt); wPrev = ScalarField("wPrev", nt); }
  for (int outer = 0; outer < ctl_.outer; ++outer) {
    Kokkos::deep_copy(uPrev, u_);
    // The buoyancy of the latest temperature: from the last outer iteration's
    // energy solve, or the last step's on the first.
    if (energy_) {
      Kokkos::deep_copy(TPrev, T_);
      if (balanced_) balancedBuoyancy(src); else addBuoyancy(src);
    }
    const VectorField& srcUse = energy_ ? srcTotal_ : src;
    const int nSweeps = ctl_.convectionSweeps > 1 ? ctl_.convectionSweeps : 1;
    for (int cs = 0; cs < nSweeps; ++cs) {
    if (nSweeps > 1) Kokkos::deep_copy(uSweep, u_);
    // Slip faces take the tangential part of the latest cell velocity.
    if (hasSlip_) slipBoundaryVelocity(uB);
    // nu_t of the initial state, before the first momentum assembly needs it.
    if (turb_ && !nutValid_) updateEddyViscosity(hasSlip_ ? uBEff_ : uB);
    assembleMomentum(hasSlip_ ? uBEff_ : uB, srcUse);
    // p_ has not changed since the last pressure solve (or since the last
    // step), so neither has its gradient.
    if (!gpValid_) { gradP(p_, gp_); gpValid_ = true; }

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
    ++rep.convectionSweeps;
    if (nSweeps > 1 && ctl_.convectionSweepTol > 0.0) {
      // Stop once another sweep would barely move the predicted velocity.
      auto u = u_; auto us = uSweep;
      Real dmax = 0.0, umax = 1e-300;
      Kokkos::parallel_reduce("csDelta", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c, Real& acc) {
          for (int d = 0; d < 3; ++d) acc = Kokkos::max(acc, Kokkos::abs(u(c, d) - us(c, d)));
        }, Kokkos::Max<Real>(dmax));
      Kokkos::parallel_reduce("csScale", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c, Real& acc) {
          for (int d = 0; d < 3; ++d) acc = Kokkos::max(acc, Kokkos::abs(u(c, d)));
        }, Kokkos::Max<Real>(umax));
      if (comm_.max(dmax) < ctl_.convectionSweepTol * comm_.max(umax)) break;
    }
    }  // convection sweeps

    for (int corr = 0; corr < ctl_.correctors; ++corr) {
      computeHbyA();
      rhieChow();
      solvePressure(pressureSolver);
      if (!gpValid_) { gradP(p_, gp_); gpValid_ = true; }
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

    // Temperature with the corrected flux, inside the outer loop, so that a
    // converged outer loop carries no coupling lag (ADR-038).
    if (energy_) solveEnergy(momentumSolver);
    // k and omega likewise, then nu_t for the next outer iteration (ADR-042).
    if (turb_) {
      Kokkos::deep_copy(kPrev, k_);
      Kokkos::deep_copy(wPrev, w_t_);
      if (hasSlip_) slipBoundaryVelocity(uB);
      Real aPt, a1, a2; bdf(aPt, a1, a2);
      solveTurbulence(momentumSolver, hasSlip_ ? uBEff_ : uB, aPt, a1, a2);
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
    bool converged = comm_.max(delta) < ctl_.outerTol * comm_.max(scale);
    if (energy_) {
      // The temperature must have stopped moving too, relative to its size.
      Real dT = 0.0, sT = 1e-300;
      auto T = T_; auto tp = TPrev;
      Kokkos::parallel_reduce("outerDeltaT", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c, Real& acc) {
          acc = Kokkos::max(acc, Kokkos::abs(T(c) - tp(c)));
        }, Kokkos::Max<Real>(dT));
      Kokkos::parallel_reduce("outerScaleT", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c, Real& acc) { acc = Kokkos::max(acc, Kokkos::abs(T(c))); },
        Kokkos::Max<Real>(sT));
      converged = converged && comm_.max(dT) < ctl_.outerTol * comm_.max(sT);
    }
    if (turb_) {
      // So must k and omega, each relative to its size.
      for (int q2 = 0; q2 < 2; ++q2) {
        auto now = q2 == 0 ? k_ : w_t_; auto was = q2 == 0 ? kPrev : wPrev;
        Real dq = 0.0, sq = 1e-300;
        Kokkos::parallel_reduce("outerDeltaKW", Kokkos::RangePolicy<ExecSpace>(0, nc),
          KOKKOS_LAMBDA(const Index c, Real& acc) {
            acc = Kokkos::max(acc, Kokkos::abs(now(c) - was(c)));
          }, Kokkos::Max<Real>(dq));
        Kokkos::parallel_reduce("outerScaleKW", Kokkos::RangePolicy<ExecSpace>(0, nc),
          KOKKOS_LAMBDA(const Index c, Real& acc) { acc = Kokkos::max(acc, Kokkos::abs(now(c))); },
          Kokkos::Max<Real>(sq));
        converged = converged && comm_.max(dq) < ctl_.outerTol * comm_.max(sq);
      }
    }
    if (converged) break;
  }
  // The force integral rebuilds the gradient the momentum equation used, so
  // it needs the slip faces' values as the last assembly took them.
  if (hasSlip_) Kokkos::deep_copy(uBnd_, uBEff_);

  ++step_;
  rep.nonOrthSweeps = lastNonOrth_;
  rep.continuityError = continuityError(F_, Fb_);
  rep.courant = courant();
  {
    auto u = u_; auto cc = m_.cellCentre();
    using Reducer = Kokkos::MaxLoc<Real, Index>;
    Reducer::value_type best;
    Kokkos::parallel_reduce("uMax", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Reducer::value_type& a) {
        const Real m = Kokkos::sqrt(u(c,0)*u(c,0) + u(c,1)*u(c,1) + u(c,2)*u(c,2));
        if (m > a.val) { a.val = m; a.loc = c; }
      }, Reducer(best));
    Kokkos::fence();
    rep.uMax = best.val;
    if (best.loc >= 0 && best.loc < nc) {
      auto hcc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), cc);
      for (int i = 0; i < 3; ++i) rep.uMaxAt[i] = hcc(best.loc, i);
    }
  }
  return rep;
}

// --------------------------------------------------------------- v2a
void PisoSolver::setBoundaryTypes(const View1<int>& bcType) {
  bcType_ = bcType;
  Index slip = 0;
  auto bt = bcType;
  Kokkos::parallel_reduce("slipCount",
    Kokkos::RangePolicy<ExecSpace>(0, static_cast<Index>(bt.extent(0))),
    KOKKOS_LAMBDA(const Index f, Index& a) {
      a += bt(f) == static_cast<int>(VelocityBC::Slip) ? 1 : 0;
    }, slip);
  // Summed over ranks, so that every rank takes the same path through
  // advance whether or not it owns a slip face itself.
  hasSlip_ = comm_.sum(slip) > 0;
  if (hasSlip_ && uBEff_.extent(0) == 0)
    uBEff_ = VectorField("uBEff", m_.nBoundaryFaces(), 3);
}

void PisoSolver::slipBoundaryVelocity(const VectorField& uB) {
  const Index nb = m_.nBoundaryFaces();
  auto bt = bcType_; auto bc = m_.boundaryCell(); auto bar = m_.boundaryArea();
  auto u = u_; auto out = uBEff_;
  Kokkos::parallel_for("slipValues", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      if (bt(f) != static_cast<int>(VelocityBC::Slip)) {
        for (int d = 0; d < 3; ++d) out(f, d) = uB(f, d);
        return;
      }
      const Real mag = Kokkos::sqrt(bar(f,0)*bar(f,0) + bar(f,1)*bar(f,1) + bar(f,2)*bar(f,2));
      Real n[3], un = 0.0;
      for (int i = 0; i < 3; ++i) { n[i] = bar(f, i) / mag; un += u(bc(f), i) * n[i]; }
      for (int d = 0; d < 3; ++d) out(f, d) = u(bc(f), d) - un * n[d];
    });
  Kokkos::fence();
}

void PisoSolver::enableEnergy(const EnergyModel& model) {
  const Index nt = m_.nTotal(), nb = m_.nBoundaryFaces(), nf = m_.nInternalFaces();
  energy_ = true;
  em_ = model;
  T_ = ScalarField("T", nt);
  TOld_ = ScalarField("TOld", nt);
  TOld2_ = ScalarField("TOld2", nt);
  TSrc_ = ScalarField("TSrc", nt);
  tType_ = View1<int>("tType", nb);          // all FixedValue ...
  tValue_ = ScalarField("tValue", nb);       // ... at zero
  srcTotal_ = VectorField("srcTotal", nt, 3);

  balanced_ = model.form == BuoyancyForm::Balanced;
  if (!balanced_) return;
  pH_ = ScalarField("pH", nt);
  rFace_ = ScalarField("rFace", nf);
  kgH_ = ScalarField("kgH", nf);
  gCell_ = VectorField("gCell", nt, 3);
  reconMinv_ = View2<Real>("reconMinv", nt, 9);
  {
    // M_P = sum over the cell's faces of S S^T / |S|, inverted once, on the
    // host. Ghost cells see only the faces this rank stores, so their M is
    // partial: they are left at zero and their force comes by exchange.
    const Index nc = m_.nCells();
    auto H = [](const auto& v) {
      return Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
    };
    auto own = H(m_.owner()); auto nei = H(m_.neighbour()); auto fa = H(m_.faceArea());
    auto bc = H(m_.boundaryCell()); auto ba = H(m_.boundaryArea());
    std::vector<Real> M(static_cast<std::size_t>(nt) * 9, 0.0);
    auto add = [&](Index c, Real s0, Real s1, Real s2) {
      const Real s[3] = {s0, s1, s2};
      const Real mag = std::sqrt(s0*s0 + s1*s1 + s2*s2);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) M[c*9 + 3*i + j] += s[i] * s[j] / mag;
    };
    for (Index f = 0; f < nf; ++f) {
      add(own(f), fa(f,0), fa(f,1), fa(f,2));
      add(nei(f), fa(f,0), fa(f,1), fa(f,2));
    }
    for (Index f = 0; f < nb; ++f) add(bc(f), ba(f,0), ba(f,1), ba(f,2));
    auto hm = Kokkos::create_mirror_view(reconMinv_);
    for (Index c = 0; c < nt; ++c) {
      for (int i = 0; i < 9; ++i) hm(c, i) = 0.0;
      if (c >= nc) continue;
      const Real* a = &M[c*9];
      const Real det = a[0]*(a[4]*a[8] - a[5]*a[7]) - a[1]*(a[3]*a[8] - a[5]*a[6])
                     + a[2]*(a[3]*a[7] - a[4]*a[6]);
      const Real inv[9] = {
        (a[4]*a[8] - a[5]*a[7]) / det, (a[2]*a[7] - a[1]*a[8]) / det, (a[1]*a[5] - a[2]*a[4]) / det,
        (a[5]*a[6] - a[3]*a[8]) / det, (a[0]*a[8] - a[2]*a[6]) / det, (a[2]*a[3] - a[0]*a[5]) / det,
        (a[3]*a[7] - a[4]*a[6]) / det, (a[1]*a[6] - a[0]*a[7]) / det, (a[0]*a[4] - a[1]*a[3]) / det};
      for (int i = 0; i < 9; ++i) hm(c, i) = inv[i];
    }
    Kokkos::deep_copy(reconMinv_, hm);
  }
  // p_h's Laplacian: the pressure's orthogonal coefficients alone, Neumann on
  // every boundary face. Constant, so assembled once.
  phSys_ = std::make_unique<LinearSystem>(m_);
  phSys_->zero();
  {
    auto own = m_.owner(); auto nei = m_.neighbour(); auto a = pdiff_.aInt();
    auto diag = phSys_->diag(); auto up = phSys_->upper(); auto lo = phSys_->lower();
    Kokkos::parallel_for("phMat", Kokkos::RangePolicy<ExecSpace>(0, nf),
      KOKKOS_LAMBDA(const Index f) {
        Kokkos::atomic_add(&diag(own(f)), a(f));
        Kokkos::atomic_add(&diag(nei(f)), a(f));
        up(f) = -a(f); lo(f) = -a(f);
      });
    Kokkos::fence();
  }
  phSolver_ = std::make_unique<NativeCG>(m_, comm_, true);
  phSolver_->notifyMatrixChanged();
}

void PisoSolver::setTemperatureBoundary(const View1<int>& type, const ScalarField& value) {
  if (!energy_) throw std::runtime_error("setTemperatureBoundary before enableEnergy");
  Kokkos::deep_copy(tType_, type);
  Kokkos::deep_copy(tValue_, value);
}

void PisoSolver::setTemperatureSource(const ScalarField& source) {
  if (!energy_) throw std::runtime_error("setTemperatureSource before enableEnergy");
  Kokkos::deep_copy(TSrc_, source);
}

void PisoSolver::setTemperature(const ScalarField& T) {
  if (!energy_) throw std::runtime_error("setTemperature before enableEnergy");
  Kokkos::deep_copy(T_, T);
  Kokkos::deep_copy(TOld_, T);
  Kokkos::deep_copy(TOld2_, T);
  sync(T_);
}

void PisoSolver::addBuoyancy(const VectorField& src) {
  // f = -(T - T_ref(x)) betaG per unit mass. T_ref is evaluated at the cell
  // centre in the same arithmetic a caller uses to write the stratification
  // there, so a fluid resting in it feels exactly no force.
  auto s = srcTotal_; auto T = T_; auto cc = m_.cellCentre();
  const Real b0 = em_.betaG.x, b1 = em_.betaG.y, b2 = em_.betaG.z;
  const Real tr = em_.tRef, g0 = em_.tRefGrad.x, g1 = em_.tRefGrad.y, g2 = em_.tRefGrad.z;
  Kokkos::parallel_for("buoyancy", Kokkos::RangePolicy<ExecSpace>(0, m_.nTotal()),
    KOKKOS_LAMBDA(const Index c) {
      const Real dT = T(c) - (tr + g0 * cc(c,0) + g1 * cc(c,1) + g2 * cc(c,2));
      s(c,0) = src(c,0) - dT * b0;
      s(c,1) = src(c,1) - dT * b1;
      s(c,2) = src(c,2) - dT * b2;
    });
  Kokkos::fence();
}

void PisoSolver::balancedBuoyancy(const VectorField& src) {
  // ADR-041. The face force B_f = f(T_f).S_f; the hydrostatic pressure p_h
  // that leaves the face residual r_f = B_f - [a_f dp_h + k_f . grad p_h]
  // divergence-free in every cell, with r = 0 on every boundary face; the cell
  // force reconstructed from r alone. A resting fluid in any T(z) on a layered
  // mesh has r = 0 exactly, so it feels nothing.
  const Index nc = m_.nCells(), nt = m_.nTotal();
  const Index nf = m_.nInternalFaces(), nb = m_.nBoundaryFaces();
  auto own = m_.owner(); auto nei = m_.neighbour(); auto bc = m_.boundaryCell();
  auto fa = m_.faceArea(); auto fc = m_.faceCentre();
  auto ba = m_.boundaryArea(); auto bcen = m_.boundaryCentre();
  auto w = w_; auto sk = skew_;
  auto a = pdiff_.aInt(); auto k = pdiff_.kInt(); auto wo = pdiff_.wOwner();
  auto ab = pdiff_.aBnd(); auto kb = pdiff_.kBnd();
  auto T = T_; auto tt = tType_; auto tv = tValue_;
  const Real b0 = em_.betaG.x, b1 = em_.betaG.y, b2 = em_.betaG.z;
  const Real tr = em_.tRef, g0 = em_.tRefGrad.x, g1 = em_.tRefGrad.y, g2 = em_.tRefGrad.z;

  VectorField gT("gTbal", nt, 3);
  gradT(gT);
  ScalarField B("Bface", nf), Bb("Bbnd", nb);
  Kokkos::parallel_for("Bface", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      Real Tf = w(f) * T(own(f)) + (1.0 - w(f)) * T(nei(f));
      for (int i = 0; i < 3; ++i)
        Tf += (w(f) * gT(own(f), i) + (1.0 - w(f)) * gT(nei(f), i)) * sk(f, i);
      const Real dT = Tf - (tr + g0 * fc(f,0) + g1 * fc(f,1) + g2 * fc(f,2));
      B(f) = -dT * (b0 * fa(f,0) + b1 * fa(f,1) + b2 * fa(f,2));
    });
  Kokkos::parallel_for("Bbnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      const Real Tb = tt(f) == static_cast<int>(TemperatureBC::FixedValue) ? tv(f) : T(bc(f));
      const Real dT = Tb - (tr + g0 * bcen(f,0) + g1 * bcen(f,1) + g2 * bcen(f,2));
      Bb(f) = -dT * (b0 * ba(f,0) + b1 * ba(f,1) + b2 * ba(f,2));
    });
  Kokkos::fence();
  Real scale = 1e-300;
  Kokkos::parallel_reduce("Bscale", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f, Real& m) { m = Kokkos::fmax(m, Kokkos::fabs(B(f))); },
    Kokkos::Max<Real>(scale));
  scale = std::max(comm_.max(scale), 1e-300);

  // p_h, its non-orthogonal part deferred and kept across calls, as the
  // pressure's is. The matrix is constant; the tolerance is absolute, a
  // fraction of the right-hand side, so a warm start costs nothing when p_h
  // has not moved.
  auto ph = pH_; auto kg = kgH_;
  auto rhs = phSys_->source();
  VectorField g("gPh", nt, 3);
  ScalarField vb("phBnd", nb);
  auto gradPh = [&]() {
    // Boundary values that leave no residual on a boundary face:
    // a_b (p_b - p_P) + k_b . grad p_P = B_b, iterated with the gradient.
    Kokkos::parallel_for("phB0", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) { vb(f) = ph(bc(f)) + Bb(f) / ab(f); });
    Kokkos::fence();
    for (int s = 0; s < 3; ++s) {
      grad_(ph, vb, g);
      Kokkos::parallel_for("phB", Kokkos::RangePolicy<ExecSpace>(0, nb),
        KOKKOS_LAMBDA(const Index f) {
          Real kgb = 0.0;
          for (int i = 0; i < 3; ++i) kgb += kb(f, i) * g(bc(f), i);
          vb(f) = ph(bc(f)) + (Bb(f) - kgb) / ab(f);
        });
      Kokkos::fence();
    }
    grad_(ph, vb, g);
  };
  for (int it = 0; it < ctl_.nonOrthCorrectors; ++it) {
    Kokkos::deep_copy(rhs, 0.0);
    Kokkos::parallel_for("phRhs", Kokkos::RangePolicy<ExecSpace>(0, nf),
      KOKKOS_LAMBDA(const Index f) {
        const Real q = B(f) - kg(f);
        Kokkos::atomic_add(&rhs(own(f)), -q);
        Kokkos::atomic_add(&rhs(nei(f)), q);
      });
    Kokkos::fence();
    // Pure Neumann: the sum of the right-hand side vanishes analytically;
    // project out what round-off leaves, as solvePressure does.
    Real rmean = 0.0, rnorm = 0.0;
    Kokkos::parallel_reduce("phRmean", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& s) { s += rhs(c); }, rmean);
    rmean = comm_.sum(rmean) / static_cast<Real>(comm_.sum(nc));
    Kokkos::parallel_for("phRproj", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c) { rhs(c) -= rmean; });
    Kokkos::parallel_reduce("phRnorm", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& s) { s += rhs(c) * rhs(c); }, rnorm);
    rnorm = std::sqrt(comm_.sum(rnorm));
    phSolver_->solve(*phSys_, ph, 0.0, 1e-15 * rnorm + 1e-300, 10000);
    Real mean = 0.0;
    Kokkos::parallel_reduce("phMean", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& s) { s += ph(c); }, mean);
    mean = comm_.sum(mean) / static_cast<Real>(comm_.sum(nc));
    Kokkos::parallel_for("phShift", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { ph(c) -= mean; });
    Kokkos::fence();
    gradPh();
    Real delta = 0.0;
    Kokkos::parallel_reduce("phNonorth", Kokkos::RangePolicy<ExecSpace>(0, nf),
      KOKKOS_LAMBDA(const Index f, Real& m) {
        Real v = 0.0;
        for (int i = 0; i < 3; ++i)
          v += k(f, i) * (wo(f) * g(own(f), i) + (1.0 - wo(f)) * g(nei(f), i));
        v = kg(f) + 0.7 * (v - kg(f));      // under-relaxed, as the pressure's
        m = Kokkos::fmax(m, Kokkos::fabs(v - kg(f)));
        kg(f) = v;
      }, Kokkos::Max<Real>(delta));
    if (comm_.max(delta) < ctl_.nonOrthTol * scale) break;
  }

  // The residual, and the cell force reconstructed from it.
  auto r = rFace_; auto gc = gCell_; auto Minv = reconMinv_;
  VectorField acc("reconRhs", nt, 3);
  Kokkos::parallel_for("rFace", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      r(f) = B(f) - a(f) * (ph(nei(f)) - ph(own(f))) - kg(f);
      const Real mag = Kokkos::sqrt(fa(f,0)*fa(f,0) + fa(f,1)*fa(f,1) + fa(f,2)*fa(f,2));
      for (int i = 0; i < 3; ++i) {
        Kokkos::atomic_add(&acc(own(f), i), fa(f, i) * r(f) / mag);
        Kokkos::atomic_add(&acc(nei(f), i), fa(f, i) * r(f) / mag);
      }
    });
  Kokkos::fence();
  Kokkos::parallel_for("recon", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) {
      for (int i = 0; i < 3; ++i) {
        Real v = 0.0;
        for (int j = 0; j < 3; ++j) v += Minv(c, 3*i + j) * acc(c, j);
        gc(c, i) = v;
      }
    });
  Kokkos::fence();
  sync(gCell_);                 // ghosts: their reconstruction was partial
  auto st = srcTotal_;
  Kokkos::parallel_for("srcBal", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) {
      for (int i = 0; i < 3; ++i) st(c, i) = src(c, i) + gc(c, i);
    });
  Kokkos::fence();
}

void PisoSolver::gradT(VectorField& g) const {
  // The prescribed temperature on a FixedValue face; the cell's own on a
  // FixedFlux face -- exact for the adiabatic walls the gates use, first
  // order in the boundary value where a non-zero flux is prescribed.
  const Index nb = m_.nBoundaryFaces();
  ScalarField tb("tb", nb);
  auto bc = m_.boundaryCell(); auto tt = tType_; auto tv = tValue_; auto T = T_;
  Kokkos::parallel_for("tb", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      tb(f) = tt(f) == static_cast<int>(TemperatureBC::FixedValue) ? tv(f) : T(bc(f));
    });
  Kokkos::fence();
  grad_(T_, tb, g);
}

void PisoSolver::solveEnergy(LinearSolver& solver) {
  // One implicit solve for T with the current face flux, assembled as a
  // velocity component is (assembleMomentum): BDF2, upwind in the matrix
  // plus the deferred correction to the skew-corrected face value, diffusion
  // with the non-orthogonal correction on the right-hand side.
  const Index nt = m_.nTotal(), nf = m_.nInternalFaces(), nb = m_.nBoundaryFaces();
  auto own = m_.owner(); auto nei = m_.neighbour(); auto bc = m_.boundaryCell();
  auto vol = m_.cellVolume(); auto bar = m_.boundaryArea();
  auto a = diff_.aInt(); auto ab = diff_.aBnd(); auto wo = diff_.wOwner();
  auto k = diff_.kInt(); auto kb = diff_.kBnd();
  auto w = w_; auto sk = skew_; auto F = F_; auto Fb = Fb_;
  auto T = T_; auto To = TOld_; auto To2 = TOld2_; auto S = TSrc_;
  auto tt = tType_; auto tv = tValue_;
  const Real kap = em_.kappa;
  const bool deferred = ctl_.deferredCorrection;
  const bool dNonOrth = ctl_.diffusionNonOrth;
  Real aPt, a1, a2; bdf(aPt, a1, a2);

  VectorField g("gT", nt, 3);
  gradT(g);

  LinearSystem sys(m_);
  sys.zero();
  auto diag = sys.diag(); auto up = sys.upper(); auto lo = sys.lower(); auto b = sys.source();
  Kokkos::parallel_for("eVol", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) {
      diag(c) = aPt * vol(c);
      b(c) = S(c) * vol(c) - (a1 * To(c) + a2 * To2(c)) * vol(c);
    });
  Kokkos::fence();
  Kokkos::parallel_for("eInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      const Real Fp = Kokkos::max(F(f), 0.0), Fn = Kokkos::max(-F(f), 0.0);
      Kokkos::atomic_add(&diag(own(f)), kap * a(f) + Fp);
      Kokkos::atomic_add(&diag(nei(f)), kap * a(f) + Fn);
      up(f) = -kap * a(f) - Fn;
      lo(f) = -kap * a(f) - Fp;
      Real nonorth = 0.0, ho = 0.0;
      for (int i = 0; i < 3; ++i) {
        const Real gf = w(f) * g(own(f), i) + (1.0 - w(f)) * g(nei(f), i);
        const Real gfo = wo(f) * g(own(f), i) + (1.0 - wo(f)) * g(nei(f), i);
        if (dNonOrth) nonorth += k(f, i) * gfo;
        ho += gf * sk(f, i);
      }
      ho += w(f) * T(own(f)) + (1.0 - w(f)) * T(nei(f));
      const Real ud = F(f) > 0.0 ? T(own(f)) : T(nei(f));
      const Real dc = deferred ? F(f) * (ho - ud) : 0.0;
      Kokkos::atomic_add(&b(own(f)), kap * nonorth - dc);
      Kokkos::atomic_add(&b(nei(f)), -kap * nonorth + dc);
    });
  Kokkos::parallel_for("eBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      if (tt(f) == static_cast<int>(TemperatureBC::FixedValue)) {
        Real nonorth = 0.0;
        if (dNonOrth)
          for (int i = 0; i < 3; ++i) nonorth += kb(f, i) * g(bc(f), i);
        Kokkos::atomic_add(&diag(bc(f)), kap * ab(f));
        // Whatever crosses a fixed-temperature face carries its temperature.
        Kokkos::atomic_add(&b(bc(f)), kap * (ab(f) * tv(f) + nonorth) - Fb(f) * tv(f));
      } else {
        const Real mag = Kokkos::sqrt(bar(f,0)*bar(f,0) + bar(f,1)*bar(f,1) + bar(f,2)*bar(f,2));
        // Outflow carries the cell's own temperature, implicitly; inflow
        // carries it too, explicitly, since no inflow temperature is given.
        Kokkos::atomic_add(&diag(bc(f)), Kokkos::max(Fb(f), 0.0));
        Kokkos::atomic_add(&b(bc(f)), tv(f) * mag - Kokkos::min(Fb(f), 0.0) * T(bc(f)));
      }
    });
  Kokkos::fence();

  solver.notifyMatrixChanged();
  ScalarField x("Tnew", nt);
  Kokkos::deep_copy(x, T_);
  solver.solve(sys, x, 1e-13, 1e-18, 5000);
  Kokkos::deep_copy(T_, x);
  sync(T_);
}

ScalarField PisoSolver::boundaryHeatFlux() const {
  const Index nt = m_.nTotal(), nb = m_.nBoundaryFaces();
  ScalarField q("boundaryHeatFlux", nb);
  if (!energy_) return q;
  VectorField g("gTq", nt, 3);
  gradT(g);
  auto bc = m_.boundaryCell(); auto bar = m_.boundaryArea();
  auto ab = diff_.aBnd(); auto kb = diff_.kBnd();
  auto T = T_; auto tt = tType_; auto tv = tValue_;
  const Real kap = em_.kappa;
  const bool dNonOrth = ctl_.diffusionNonOrth;
  Kokkos::parallel_for("heatFlux", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      if (tt(f) == static_cast<int>(TemperatureBC::FixedValue)) {
        Real nonorth = 0.0;
        if (dNonOrth)
          for (int i = 0; i < 3; ++i) nonorth += kb(f, i) * g(bc(f), i);
        q(f) = kap * (ab(f) * (tv(f) - T(bc(f))) + nonorth);
      } else {
        q(f) = tv(f) * Kokkos::sqrt(bar(f,0)*bar(f,0) + bar(f,1)*bar(f,1) + bar(f,2)*bar(f,2));
      }
    });
  Kokkos::fence();
  return q;
}

}  // namespace vibeflow
