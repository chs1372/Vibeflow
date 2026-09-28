// The k-omega SST model (ADR-042) and the wall distance.
//
// The C++ port of prototype/sst.py and prototype/walldist.py. k and omega are
// transported as the temperature is (solveEnergy): BDF2, upwind in the
// matrix plus the deferred correction to the skew-corrected face value,
// diffusion with the non-orthogonal correction -- here with a face
// diffusivity nu + sigma nu_t interpolated from the cells. Destruction is
// implicit, production explicit, the cross diffusion explicit where positive
// and on the diagonal where negative, and an imposed source by Patankar's
// rule (its negative part on the diagonal as -s/phi).

#include "physics/Piso.hpp"
#include "physics/WallDistance.hpp"
#include "linalg/LinearSolver.hpp"
#include "linalg/LinearSystem.hpp"
#include "mesh/Mesh.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vibeflow {

namespace {

constexpr Real SIGMA_K1 = 0.85, SIGMA_W1 = 0.5, BETA1 = 0.075;
constexpr Real SIGMA_K2 = 1.0, SIGMA_W2 = 0.856, BETA2 = 0.0828;
constexpr Real BETA_STAR = 0.09, KAPPA = 0.41, A1 = 0.31;

struct SstConstants {
  Real g1, g2, cdFloor, plim;
  bool limitW;       // omega production from the limited P (2003)
  bool vorticity;    // eddy-viscosity limiter on Omega (1994) or S (2003)
};

SstConstants constantsOf(SstVariant v) {
  if (v == SstVariant::Menter2003) return {5.0 / 9.0, 0.44, 1e-10, 10.0, true, false};
  const Real rb = std::sqrt(BETA_STAR);
  return {BETA1 / BETA_STAR - SIGMA_W1 * KAPPA * KAPPA / rb,
          BETA2 / BETA_STAR - SIGMA_W2 * KAPPA * KAPPA / rb, 1e-20, 20.0, false, true};
}

// F1, F2 and nu_t from k, omega, grad k . grad omega, the wall distance and
// the rate the limiter uses. On the wall (d = 0) F1 = F2 = 1.
KOKKOS_INLINE_FUNCTION void blending(Real k, Real w, Real kw, Real d, Real V, Real nu,
                                     Real cdFloor, Real& F1, Real& F2, Real& nut) {
  if (d > 0.0) {
    const Real rk = Kokkos::sqrt(Kokkos::fmax(k, 0.0));
    const Real brA = rk / (BETA_STAR * w * d);
    const Real brB = 500.0 * nu / (d * d * w);
    const Real CD = Kokkos::fmax(2.0 * SIGMA_W2 * kw / w, cdFloor);
    const Real brC = 4.0 * SIGMA_W2 * k / (CD * d * d);
    const Real arg1 = Kokkos::fmin(Kokkos::fmax(brA, brB), brC);
    const Real arg2 = Kokkos::fmax(2.0 * rk / (BETA_STAR * w * d), brB);
    const Real a1sq = arg1 * arg1;
    F1 = Kokkos::tanh(a1sq * a1sq);
    F2 = Kokkos::tanh(arg2 * arg2);
  } else {
    F1 = 1.0;
    F2 = 1.0;
  }
  nut = A1 * k / Kokkos::fmax(A1 * w, V * F2);
}

// Closest-point distance from p to the triangle (a, b, c), Ericson 5.1.5.
KOKKOS_INLINE_FUNCTION Real triangleDistance(const Real p[3], const Real a[3], const Real b[3],
                                             const Real c[3]) {
  Real ab[3], ac[3], ap[3], bp[3], cp[3];
  for (int i = 0; i < 3; ++i) {
    ab[i] = b[i] - a[i]; ac[i] = c[i] - a[i]; ap[i] = p[i] - a[i];
    bp[i] = p[i] - b[i]; cp[i] = p[i] - c[i];
  }
  auto dot = [](const Real* u, const Real* v) { return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]; };
  const Real d1 = dot(ab, ap), d2 = dot(ac, ap);
  const Real d3 = dot(ab, bp), d4 = dot(ac, bp);
  const Real d5 = dot(ab, cp), d6 = dot(ac, cp);
  const Real vc = d1 * d4 - d3 * d2;
  const Real vb = d5 * d2 - d1 * d6;
  const Real va = d3 * d6 - d5 * d4;
  auto safe = [](Real n, Real d) { return d != 0.0 ? n / d : 0.0; };
  Real v, w;
  if (d1 <= 0.0 && d2 <= 0.0)                          { v = 0.0; w = 0.0; }
  else if (d3 >= 0.0 && d4 <= d3)                      { v = 1.0; w = 0.0; }
  else if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)        { v = safe(d1, d1 - d3); w = 0.0; }
  else if (d6 >= 0.0 && d5 <= d6)                      { v = 0.0; w = 1.0; }
  else if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)        { v = 0.0; w = safe(d2, d2 - d6); }
  else if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    w = safe(d4 - d3, (d4 - d3) + (d5 - d6));
    v = 1.0 - w;
  } else {
    const Real den = va + vb + vc;
    v = safe(vb, den); w = safe(vc, den);
  }
  Real s = 0.0;
  for (int i = 0; i < 3; ++i) {
    const Real q = a[i] + v * ab[i] + w * ac[i];
    s += (p[i] - q) * (p[i] - q);
  }
  return Kokkos::sqrt(s);
}

}  // namespace

// ------------------------------------------------------------ wall distance
ScalarField wallDistance(const Mesh& mesh, const View1<int>& wallMask, const VectorField& points,
                         Index n, const Comm& comm) {
  const Index nb = mesh.nBoundaryFaces();
  const VectorField corners = mesh.boundaryCorners();
  if (nb > 0 && corners.extent(0) != static_cast<std::size_t>(4 * nb))
    throw std::runtime_error("wallDistance: this mesh does not keep its boundary corners");
  auto hc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), corners);
  auto hm = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), wallMask);
  // This rank's wall triangles, nine coordinates each: the fan of each face
  // from its vertex average, as its geometry is built.
  std::vector<double> mine;
  for (Index f = 0; f < nb; ++f) {
    if (!hm(f)) continue;
    Real avg[3] = {0.0, 0.0, 0.0};
    for (int t = 0; t < 4; ++t)
      for (int i = 0; i < 3; ++i) avg[i] += 0.25 * hc(4*f + t, i);
    for (int t = 0; t < 4; ++t) {
      const int u = (t + 1) % 4;
      for (int i = 0; i < 3; ++i) mine.push_back(avg[i]);
      for (int i = 0; i < 3; ++i) mine.push_back(hc(4*f + t, i));
      for (int i = 0; i < 3; ++i) mine.push_back(hc(4*f + u, i));
    }
  }
  // Every rank's, on every rank.
  std::vector<double> all = mine;
  if (comm.parallel()) {
    std::vector<std::vector<double>> send(static_cast<std::size_t>(comm.size()), mine);
    const auto got = comm.alltoallv(send);
    all.clear();
    for (const auto& g : got) all.insert(all.end(), g.begin(), g.end());
  }
  const Index nTri = static_cast<Index>(all.size() / 9);
  Kokkos::View<Real**, MemSpace> tri("wallTriangles", nTri, 9);
  {
    auto ht = Kokkos::create_mirror_view(tri);
    for (Index t = 0; t < nTri; ++t)
      for (int j = 0; j < 9; ++j) ht(t, j) = all[static_cast<std::size_t>(9 * t + j)];
    Kokkos::deep_copy(tri, ht);
  }
  ScalarField d("wallDistance", n);
  Kokkos::parallel_for("wallDistance", Kokkos::RangePolicy<ExecSpace>(0, n),
    KOKKOS_LAMBDA(const Index c) {
      const Real p[3] = {points(c, 0), points(c, 1), points(c, 2)};
      Real best = 1e300;
      for (Index t = 0; t < nTri; ++t) {
        const Real a[3] = {tri(t, 0), tri(t, 1), tri(t, 2)};
        const Real b[3] = {tri(t, 3), tri(t, 4), tri(t, 5)};
        const Real q[3] = {tri(t, 6), tri(t, 7), tri(t, 8)};
        best = Kokkos::fmin(best, triangleDistance(p, a, b, q));
      }
      d(c) = best;
    });
  Kokkos::fence();
  return d;
}

// ------------------------------------------------------------ set-up
void PisoSolver::enableTurbulence(const TurbulenceModel& model, const View1<int>& wallMask) {
  const Index nt = m_.nTotal(), nb = m_.nBoundaryFaces();
  turb_ = true;
  tm_ = model;
  k_ = ScalarField("k", nt); kOld_ = ScalarField("kOld", nt); kOld2_ = ScalarField("kOld2", nt);
  w_t_ = ScalarField("omega", nt); wOld_ = ScalarField("wOld", nt); wOld2_ = ScalarField("wOld2", nt);
  nut_ = ScalarField("nut", nt);
  nutB_ = ScalarField("nutB", nb);
  kwType_ = View1<int>("kwType", nb);
  Kokkos::deep_copy(kwType_, static_cast<int>(TurbulenceBC::ZeroGradient));
  kValue_ = ScalarField("kValue", nb);
  wValue_ = ScalarField("wValue", nb);
  kSrc_ = ScalarField("kSrc", nt);
  wSrc_ = ScalarField("wSrc", nt);
  dWall_ = vibeflow::wallDistance(m_, wallMask, m_.cellCentre(), nt, comm_);
  dWallB_ = vibeflow::wallDistance(m_, wallMask, m_.boundaryCentre(), nb, comm_);
  auto dB = dWallB_; auto mask = wallMask;
  Kokkos::parallel_for("dWallOnWall", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) { if (mask(f)) dB(f) = 0.0; });
  Kokkos::fence();
  bounded_ = 0;
  turbSteps_ = 0;
  nutValid_ = false;
}

void PisoSolver::setTurbulenceBoundary(const View1<int>& kind, const ScalarField& kValue,
                                       const ScalarField& wValue) {
  if (!turb_) throw std::runtime_error("setTurbulenceBoundary before enableTurbulence");
  Kokkos::deep_copy(kwType_, kind);
  Kokkos::deep_copy(kValue_, kValue);
  Kokkos::deep_copy(wValue_, wValue);
  nutValid_ = false;
}

void PisoSolver::setTurbulenceSource(const ScalarField& kSource, const ScalarField& wSource) {
  if (!turb_) throw std::runtime_error("setTurbulenceSource before enableTurbulence");
  Kokkos::deep_copy(kSrc_, kSource);
  Kokkos::deep_copy(wSrc_, wSource);
}

void PisoSolver::setTurbulence(const ScalarField& k, const ScalarField& w) {
  if (!turb_) throw std::runtime_error("setTurbulence before enableTurbulence");
  Kokkos::deep_copy(k_, k); Kokkos::deep_copy(kOld_, k); Kokkos::deep_copy(kOld2_, k);
  Kokkos::deep_copy(w_t_, w); Kokkos::deep_copy(wOld_, w); Kokkos::deep_copy(wOld2_, w);
  sync(k_); sync(kOld_); sync(kOld2_);
  sync(w_t_); sync(wOld_); sync(wOld2_);
  nutValid_ = false;          // set from these on the next step, with its uB
}

// ------------------------------------------------------------ the model
void PisoSolver::turbulenceBoundaryValues(ScalarField& kb, ScalarField& wb) const {
  const Index nb = m_.nBoundaryFaces();
  kb = ScalarField("kb", nb);
  wb = ScalarField("wb", nb);
  auto bc = m_.boundaryCell(); auto kind = kwType_;
  auto kv = kValue_; auto wv = wValue_; auto k = k_; auto w = w_t_; auto d = dWall_;
  const Real nu = nu_;
  auto kbv = kb; auto wbv = wb;
  Kokkos::parallel_for("kwB", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      const int t = kind(f);
      if (t == static_cast<int>(TurbulenceBC::Dirichlet)) {
        kbv(f) = kv(f); wbv(f) = wv(f);
      } else if (t == static_cast<int>(TurbulenceBC::Wall)) {
        const Real d1 = d(bc(f));
        kbv(f) = 0.0; wbv(f) = 60.0 * nu / (BETA1 * d1 * d1);
      } else {
        kbv(f) = k(bc(f)); wbv(f) = w(bc(f));
      }
    });
  Kokkos::fence();
}

void PisoSolver::velocityInvariants(const VectorField& uB, ScalarField& S, ScalarField& Om) const {
  // The velocity gradient with the boundary values the momentum equation
  // uses: the effective face value, or the cell's on a zero-gradient face.
  const Index nt = m_.nTotal(), nb = m_.nBoundaryFaces();
  auto bc = m_.boundaryCell(); auto bt = bcType_; auto u = u_;
  VectorField g0("gu0", nt, 3), g1("gu1", nt, 3), g2("gu2", nt, 3);
  VectorField* gs[3] = {&g0, &g1, &g2};
  for (int d = 0; d < 3; ++d) {
    ScalarField comp("uc", nt), compB("ucB", nb);
    Kokkos::parallel_for("uc", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { comp(c) = u(c, d); });
    Kokkos::parallel_for("ucB", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f) {
        compB(f) = (bt(f) == static_cast<int>(VelocityBC::ZeroGradient)) ? u(bc(f), d) : uB(f, d);
      });
    Kokkos::fence();
    grad_(comp, compB, *gs[d]);
  }
  S = ScalarField("S", nt);
  Om = ScalarField("Omega", nt);
  auto Sv = S; auto Ov = Om;
  Kokkos::parallel_for("invariants", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) {
      Real G[3][3];
      for (int j = 0; j < 3; ++j) { G[0][j] = g0(c, j); G[1][j] = g1(c, j); G[2][j] = g2(c, j); }
      Real s2 = 0.0, o2 = 0.0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          const Real sy = G[i][j] + G[j][i], sk = G[i][j] - G[j][i];
          s2 += sy * sy; o2 += sk * sk;
        }
      Sv(c) = Kokkos::sqrt(0.5 * s2);
      Ov(c) = Kokkos::sqrt(0.5 * o2);
    });
  Kokkos::fence();
}

void PisoSolver::updateEddyViscosity(const VectorField& uB) {
  const Index nt = m_.nTotal(), nb = m_.nBoundaryFaces();
  const SstConstants C = constantsOf(tm_.variant);
  ScalarField kb, wb;
  turbulenceBoundaryValues(kb, wb);
  VectorField gk("gk", nt, 3), gw("gw", nt, 3);
  grad_(k_, kb, gk);
  grad_(w_t_, wb, gw);
  ScalarField S, Om;
  velocityInvariants(uB, S, Om);
  auto k = k_; auto w = w_t_; auto d = dWall_; auto nut = nut_;
  const Real nu = nu_; const bool vort = C.vorticity; const Real cdF = C.cdFloor;
  Kokkos::parallel_for("nut", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) {
      const Real kw = gk(c,0)*gw(c,0) + gk(c,1)*gw(c,1) + gk(c,2)*gw(c,2);
      Real F1, F2, nt_;
      blending(k(c), w(c), kw, d(c), vort ? Om(c) : S(c), nu, cdF, F1, F2, nt_);
      nut(c) = nt_;
    });
  auto bc = m_.boundaryCell(); auto kind = kwType_; auto dB = dWallB_; auto nutB = nutB_;
  Kokkos::parallel_for("nutB", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      const Index c = bc(f);
      if (kind(f) == static_cast<int>(TurbulenceBC::ZeroGradient)) {
        const Real kw = gk(c,0)*gw(c,0) + gk(c,1)*gw(c,1) + gk(c,2)*gw(c,2);
        Real F1, F2, nt_;
        blending(k(c), w(c), kw, d(c), vort ? Om(c) : S(c), nu, cdF, F1, F2, nt_);
        nutB(f) = nt_;
      } else {
        const Real kw = gk(c,0)*gw(c,0) + gk(c,1)*gw(c,1) + gk(c,2)*gw(c,2);
        Real F1, F2, nt_;
        blending(kb(f), wb(f), kw, dB(f), vort ? Om(c) : S(c), nu, cdF, F1, F2, nt_);
        nutB(f) = nt_;
      }
    });
  Kokkos::fence();
  nutValid_ = true;
}

// One implicit solve of d(phi)/dt + div(F phi) = div(G grad phi) + rhsV -
// diagV phi, G a cell field interpolated to the faces (Gb on boundary faces).
void PisoSolver::transportSolve(ScalarField& phi, const ScalarField& old, const ScalarField& old2,
                                const ScalarField& phib, const VectorField& gphi,
                                const ScalarField& Gc, const ScalarField& Gb,
                                const ScalarField& diagV, const ScalarField& rhsV,
                                Real aPt, Real a1, Real a2, LinearSolver& solver) {
  const Index nt = m_.nTotal(), nf = m_.nInternalFaces(), nb = m_.nBoundaryFaces();
  auto own = m_.owner(); auto nei = m_.neighbour(); auto bc = m_.boundaryCell();
  auto vol = m_.cellVolume();
  auto a = diff_.aInt(); auto ab = diff_.aBnd(); auto wo = diff_.wOwner();
  auto k = diff_.kInt(); auto kb = diff_.kBnd();
  auto w = w_; auto sk = skew_; auto F = F_; auto Fb = Fb_;
  auto kind = kwType_;
  const bool deferred = ctl_.deferredCorrection && tm_.secondOrderAdvection;
  const bool dNonOrth = ctl_.diffusionNonOrth;

  LinearSystem sys(m_);
  sys.zero();
  auto diag = sys.diag(); auto up = sys.upper(); auto lo = sys.lower(); auto b = sys.source();
  auto P = phi;
  Kokkos::parallel_for("tVol", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) {
      diag(c) = (aPt + diagV(c)) * vol(c);
      b(c) = rhsV(c) * vol(c) - (a1 * old(c) + a2 * old2(c)) * vol(c);
    });
  Kokkos::fence();
  Kokkos::parallel_for("tInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      const Real G = w(f) * Gc(own(f)) + (1.0 - w(f)) * Gc(nei(f));
      const Real Fp = Kokkos::max(F(f), 0.0), Fn = Kokkos::max(-F(f), 0.0);
      Kokkos::atomic_add(&diag(own(f)), G * a(f) + Fp);
      Kokkos::atomic_add(&diag(nei(f)), G * a(f) + Fn);
      up(f) = -G * a(f) - Fn;
      lo(f) = -G * a(f) - Fp;
      Real nonorth = 0.0, ho = 0.0;
      for (int i = 0; i < 3; ++i) {
        const Real gf = w(f) * gphi(own(f), i) + (1.0 - w(f)) * gphi(nei(f), i);
        const Real gfo = wo(f) * gphi(own(f), i) + (1.0 - wo(f)) * gphi(nei(f), i);
        if (dNonOrth) nonorth += k(f, i) * gfo;
        ho += gf * sk(f, i);
      }
      ho += w(f) * P(own(f)) + (1.0 - w(f)) * P(nei(f));
      const Real ud = F(f) > 0.0 ? P(own(f)) : P(nei(f));
      const Real dc = deferred ? F(f) * (ho - ud) : 0.0;
      Kokkos::atomic_add(&b(own(f)), G * nonorth - dc);
      Kokkos::atomic_add(&b(nei(f)), -G * nonorth + dc);
    });
  Kokkos::parallel_for("tBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      if (kind(f) != static_cast<int>(TurbulenceBC::ZeroGradient)) {
        Real nonorth = 0.0;
        if (dNonOrth)
          for (int i = 0; i < 3; ++i) nonorth += kb(f, i) * gphi(bc(f), i);
        Kokkos::atomic_add(&diag(bc(f)), Gb(f) * ab(f));
        Kokkos::atomic_add(&b(bc(f)), Gb(f) * (ab(f) * phib(f) + nonorth) - Fb(f) * phib(f));
      } else {
        // Outflow carries the cell's value, implicitly; inflow too, explicitly.
        Kokkos::atomic_add(&diag(bc(f)), Kokkos::max(Fb(f), 0.0));
        Kokkos::atomic_add(&b(bc(f)), -Kokkos::min(Fb(f), 0.0) * P(bc(f)));
      }
    });
  Kokkos::fence();

  solver.notifyMatrixChanged();
  ScalarField x("phiNew", nt);
  Kokkos::deep_copy(x, phi);
  solver.solve(sys, x, 1e-13, 1e-300, 5000);
  phi = x;
}

void PisoSolver::solveTurbulence(LinearSolver& solver, const VectorField& uB, Real aPt,
                                 Real a1, Real a2) {
  const Index nt = m_.nTotal(), nc = m_.nCells(), nb = m_.nBoundaryFaces();
  const SstConstants C = constantsOf(tm_.variant);
  ScalarField kb, wb;
  turbulenceBoundaryValues(kb, wb);
  VectorField gk("gk", nt, 3), gw("gw", nt, 3);
  grad_(k_, kb, gk);
  grad_(w_t_, wb, gw);
  ScalarField S, Om;
  velocityInvariants(uB, S, Om);

  // Cell coefficients, from the current iterate.
  ScalarField Gk("Gk", nt), Gw("Gw", nt), dK("dK", nt), rK("rK", nt), dW("dW", nt), rW("rW", nt);
  auto k = k_; auto w = w_t_; auto d = dWall_; auto ks = kSrc_; auto ws = wSrc_;
  const Real nu = nu_; const bool vort = C.vorticity; const Real cdF = C.cdFloor;
  const Real g1 = C.g1, g2 = C.g2, plim = C.plim; const bool limitW = C.limitW;
  Kokkos::parallel_for("sstCells", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) {
      const Real kw = gk(c,0)*gw(c,0) + gk(c,1)*gw(c,1) + gk(c,2)*gw(c,2);
      Real F1, F2, nt_;
      blending(k(c), w(c), kw, d(c), vort ? Om(c) : S(c), nu, cdF, F1, F2, nt_);
      const Real sK = F1 * SIGMA_K1 + (1.0 - F1) * SIGMA_K2;
      const Real sW = F1 * SIGMA_W1 + (1.0 - F1) * SIGMA_W2;
      const Real beta = F1 * BETA1 + (1.0 - F1) * BETA2;
      const Real gamma = F1 * g1 + (1.0 - F1) * g2;
      const Real S2 = S(c) * S(c);
      const Real P = nt_ * S2;
      const Real Pk = Kokkos::fmin(P, plim * BETA_STAR * w(c) * k(c));
      // gamma P / nu_t = gamma S^2 exactly, P being nu_t S^2.
      const Real prodW = limitW ? gamma * Kokkos::fmin(S2, plim * BETA_STAR * w(c) * k(c) / nt_)
                                : gamma * S2;
      const Real cross = 2.0 * (1.0 - F1) * SIGMA_W2 * kw / w(c);
      const Real ksp = Kokkos::fmax(ks(c), 0.0), ksn = Kokkos::fmax(-ks(c), 0.0);
      const Real wsp = Kokkos::fmax(ws(c), 0.0), wsn = Kokkos::fmax(-ws(c), 0.0);
      Gk(c) = nu + sK * nt_;
      Gw(c) = nu + sW * nt_;
      dK(c) = BETA_STAR * w(c) + ksn / k(c);
      rK(c) = Pk + ksp;
      // beta w^2 linearised by Newton, 2 beta w_old w - beta w_old^2. The
      // Picard form beta w_old w makes the balance with production a
      // period-two map, w_new w_old = P/beta, which the stiff first cell on
      // a wall turned into a growing oscillation (ADR-042's revision).
      dW(c) = 2.0 * beta * w(c) + (cross < 0.0 ? -cross / w(c) : 0.0) + wsn / w(c);
      rW(c) = prodW + beta * w(c) * w(c) + (cross > 0.0 ? cross : 0.0) + wsp;
    });
  // Boundary diffusivities: nu_t and F1 from the face values where they are
  // fixed, the cell's where the face follows the cell.
  ScalarField GkB("GkB", nb), GwB("GwB", nb);
  auto bc = m_.boundaryCell(); auto kind = kwType_; auto dB = dWallB_;
  Kokkos::parallel_for("sstBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      const Index c = bc(f);
      const Real kw = gk(c,0)*gw(c,0) + gk(c,1)*gw(c,1) + gk(c,2)*gw(c,2);
      Real F1, F2, nt_;
      if (kind(f) == static_cast<int>(TurbulenceBC::ZeroGradient))
        blending(k(c), w(c), kw, d(c), vort ? Om(c) : S(c), nu, cdF, F1, F2, nt_);
      else
        blending(kb(f), wb(f), kw, dB(f), vort ? Om(c) : S(c), nu, cdF, F1, F2, nt_);
      GkB(f) = nu + (F1 * SIGMA_K1 + (1.0 - F1) * SIGMA_K2) * nt_;
      GwB(f) = nu + (F1 * SIGMA_W1 + (1.0 - F1) * SIGMA_W2) * nt_;
    });
  Kokkos::fence();

  ScalarField kNew("kNew", nt), wNew("wNew", nt);
  Kokkos::deep_copy(kNew, k_);
  Kokkos::deep_copy(wNew, w_t_);
  transportSolve(kNew, kOld_, kOld2_, kb, gk, Gk, GkB, dK, rK, aPt, a1, a2, solver);
  transportSolve(wNew, wOld_, wOld2_, wb, gw, Gw, GwB, dW, rW, aPt, a1, a2, solver);

  // Bound below; count the owned cells touched.
  const Real kF = tm_.kFloor, wF = tm_.wFloor;
  Index low = 0;
  Kokkos::parallel_reduce("sstBoundCount", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index c, Index& n) {
      if (kNew(c) < kF || wNew(c) < wF) ++n;
    }, low);
  Kokkos::parallel_for("sstBound", Kokkos::RangePolicy<ExecSpace>(0, nt),
    KOKKOS_LAMBDA(const Index c) {
      k(c) = Kokkos::fmax(kNew(c), kF);
      w(c) = Kokkos::fmax(wNew(c), wF);
    });
  Kokkos::fence();
  bounded_ += comm_.sum(low);
  sync(k_);
  sync(w_t_);
  updateEddyViscosity(uB);
}

void PisoSolver::advanceTurbulenceFrozen(const VectorField& uBoundary, const ScalarField& fBoundary,
                                         LinearSolver& solver) {
  if (!turb_) throw std::runtime_error("advanceTurbulenceFrozen before enableTurbulence");
  Kokkos::deep_copy(Fb_, fBoundary);
  if (hasSlip_) slipBoundaryVelocity(uBoundary);
  const VectorField& uB = hasSlip_ ? uBEff_ : uBoundary;
  sync(u_);
  Kokkos::deep_copy(kOld2_, kOld_); Kokkos::deep_copy(kOld_, k_);
  Kokkos::deep_copy(wOld2_, wOld_); Kokkos::deep_copy(wOld_, w_t_);
  if (!nutValid_) updateEddyViscosity(uB);
  const Real aPt = turbSteps_ == 0 ? 1.0 / dt_ : 1.5 / dt_;
  const Real a1 = turbSteps_ == 0 ? -1.0 / dt_ : -2.0 / dt_;
  const Real a2 = turbSteps_ == 0 ? 0.0 : 0.5 / dt_;
  solveTurbulence(solver, uB, aPt, a1, a2);
  ++turbSteps_;
}

}  // namespace vibeflow
