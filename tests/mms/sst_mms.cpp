// v2b gates 1-2, C++ side (ADR-042): the wall distance and the k-omega SST
// model equations.
//
// The port of prototype/sst_gates.py, one mesh finer -- 8/16/32 where Python
// runs 6/12/24 -- judged by ADR-042's criteria as revised before the result:
//
//   1. Wall distance. Walls at y = 0 and y = 1 on the orthogonal and smooth
//      families (8, 16) and the randomly perturbed fixture mesh (8): their
//      boundaries stay planar, so d = min(y, 1 - y) at every cell centre and
//      boundary face centre, to 1e-13.
//   2. Manufactured solutions MS-A (blending on, limiter off) and MS-B
//      (limiter on), both variants, the exact fields and sources generated
//      from prototype/sst_ms.py into sst_ms.hpp:
//        2a. frozen velocity: the exact velocity and a discretely
//            divergence-free face flux; k and omega solved. MS-A and MS-B.
//        2b. coupled: u, p, k and omega solved together. MS-A.
//      Orders of k and omega (and u) in [1.85, 2.15] on the orthogonal
//      meshes, [1.6, 2.3] and approaching 2 on the smooth distortion; steady
//      when the largest change per step of u, k and omega, each relative to
//      its largest value, is below 1e-8 (ADR-038's C++ floor); no cell
//      bounded.
//
// Run:  sst_mms <fixtures> [gates: any of 1 a b, default all] [grids, default 8 16 32]
//       VIBEFLOW_STEADY_TOL=<tol> changes the steady criterion (cross-check).

#include "mesh/HexMesh.hpp"
#include "discretization/FaceFlux.hpp"
#include "physics/Piso.hpp"
#include "physics/WallDistance.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#include "sst_ms.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

using namespace vibeflow;

namespace {

using ScalarFn = std::function<Real(const Vec3&)>;
using VecFn = std::function<Vec3(const Vec3&)>;

template <class V> auto host(const V& v) {
  return Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
}
template <class V> Vec3 centre(const V& cc, Index c) { return {cc(c,0), cc(c,1), cc(c,2)}; }

ScalarField cellScalar(const Mesh& m, const ScalarFn& fn) {
  auto cc = host(m.cellCentre());
  ScalarField s("cellScalar", m.nTotal());
  auto h = Kokkos::create_mirror_view(s);
  for (Index c = 0; c < m.nTotal(); ++c) h(c) = fn(centre(cc, c));
  Kokkos::deep_copy(s, h);
  return s;
}

VectorField cellVector(const Mesh& m, const VecFn& fn) {
  auto cc = host(m.cellCentre());
  VectorField v("cellVector", m.nTotal(), 3);
  auto h = Kokkos::create_mirror_view(v);
  for (Index c = 0; c < m.nTotal(); ++c) {
    const Vec3 x = fn(centre(cc, c));
    h(c,0) = x.x; h(c,1) = x.y; h(c,2) = x.z;
  }
  Kokkos::deep_copy(v, h);
  return v;
}

ScalarField faceAverage(const HexMesh& m, const ScalarFn& fn) {
  VectorField v;
  averageBoundaryValue(m, [&](const Vec3& q) { return Vec3{fn(q), 0.0, 0.0}; }, v);
  ScalarField s("faceAverage", m.nBoundaryFaces());
  Kokkos::parallel_for("fa", Kokkos::RangePolicy<ExecSpace>(0, m.nBoundaryFaces()),
    KOKKOS_LAMBDA(const Index f) { s(f) = v(f, 0); });
  Kokkos::fence();
  return s;
}

void adjustBoundaryFlux(const Mesh& mesh, ScalarField& fb) {
  auto bar = mesh.boundaryArea();
  Real total = 0.0, areaSum = 0.0;
  Kokkos::parallel_reduce("fbSum", Kokkos::RangePolicy<ExecSpace>(0, mesh.nBoundaryFaces()),
    KOKKOS_LAMBDA(const Index f, Real& a) { a += fb(f); }, total);
  Kokkos::parallel_reduce("areaSum", Kokkos::RangePolicy<ExecSpace>(0, mesh.nBoundaryFaces()),
    KOKKOS_LAMBDA(const Index f, Real& a) {
      a += Kokkos::sqrt(bar(f,0)*bar(f,0) + bar(f,1)*bar(f,1) + bar(f,2)*bar(f,2));
    }, areaSum);
  Kokkos::parallel_for("fbAdj", Kokkos::RangePolicy<ExecSpace>(0, mesh.nBoundaryFaces()),
    KOKKOS_LAMBDA(const Index f) {
      const Real a = Kokkos::sqrt(bar(f,0)*bar(f,0) + bar(f,1)*bar(f,1) + bar(f,2)*bar(f,2));
      fb(f) -= total * a / areaSum;
    });
  Kokkos::fence();
}

Real l2(const Mesh& m, const std::vector<Real>& e, int ncomp) {
  auto vol = host(m.cellVolume());
  Real num = 0.0, den = 0.0;
  for (Index c = 0; c < m.nCells(); ++c) {
    Real s = 0.0;
    for (int d = 0; d < ncomp; ++d) s += e[c*ncomp + d] * e[c*ncomp + d];
    num += s * vol(c);
    den += vol(c);
  }
  return std::sqrt(num / den);
}

std::vector<Real> toHost(const ScalarField& s, Index n) {
  auto h = host(s);
  std::vector<Real> out(n);
  for (Index c = 0; c < n; ++c) out[c] = h(c);
  return out;
}

std::vector<Real> toHost(const VectorField& v, Index n) {
  auto h = host(v);
  std::vector<Real> out(n * 3);
  for (Index c = 0; c < n; ++c)
    for (int d = 0; d < 3; ++d) out[c*3 + d] = h(c, d);
  return out;
}

Real relChange(const std::vector<Real>& a, const std::vector<Real>& b) {
  Real d = 0.0, s = 1e-300;
  for (std::size_t i = 0; i < a.size(); ++i) {
    d = std::max(d, std::abs(a[i] - b[i]));
    s = std::max(s, std::abs(a[i]));
  }
  return d / s;
}

std::vector<Real> orders(const std::vector<Real>& e, const std::vector<Real>& h) {
  std::vector<Real> o;
  for (std::size_t i = 1; i < e.size(); ++i)
    o.push_back(std::log(e[i-1] / e[i]) / std::log(h[i-1] / h[i]));
  return o;
}

bool verdict(const std::string& label, const std::vector<Real>& o, Real lo, Real hi,
             bool approaching) {
  const Real last = o.back();
  bool ok = last >= lo && last <= hi;
  std::printf("  -> %s: orders", label.c_str());
  for (std::size_t i = 0; i < o.size(); ++i) std::printf("%s %.3f", i ? "," : "", o[i]);
  std::printf("; last in [%g, %g]", lo, hi);
  if (approaching && o.size() > 1) {
    bool r = true;
    for (std::size_t i = 1; i < o.size(); ++i)
      r &= std::abs(o[i] - 2.0) <= std::abs(o[i-1] - 2.0) + 0.02;
    ok &= r;
    std::printf(", %s", r ? "approaching 2" : "MOVING AWAY FROM 2");
  }
  std::printf(": %s\n", ok ? "PASS" : "FAIL");
  std::fflush(stdout);
  return ok;
}

HexMesh family(Index n, Real skew) {
  return HexMesh::generate(n, skew, skew == 0.0 ? "none" : "smooth");
}

// The steady criterion: ADR-042's 1e-8 for the order runs; the cross-check
// (crosscheck_v2b.py) asks for 1e-12 through VIBEFLOW_STEADY_TOL.
const Real STEADY = [] {
  const char* e = std::getenv("VIBEFLOW_STEADY_TOL");
  return e ? std::atof(e) : 1e-8;
}();

// ------------------------------------------------------------ 1. wall distance
bool gateWallDistance(const std::string& fixtures) {
  std::printf("\n1. wall distance: walls at y = 0 and y = 1, d = min(y, 1 - y)\n");
  bool ok = true;
  struct Case { std::string tag; HexMesh mesh; };
  std::vector<Case> cases;
  for (Index n : {8, 16}) {
    cases.push_back({"orthogonal         n=" + std::to_string(n), family(n, 0.0)});
    cases.push_back({"smooth distortion  n=" + std::to_string(n), family(n, 0.25)});
  }
  cases.push_back({"perturbed          n=8 ",
                   HexMesh::fromVertexFile(8, fixtures + "/vertices_n8_s25.txt")});
  for (const Case& cs : cases) {
    const HexMesh& m = cs.mesh;
    const Index nb = m.nBoundaryFaces();
    auto side = host(m.boundarySide());
    View1<int> wall("wall", nb);
    auto hw = Kokkos::create_mirror_view(wall);
    for (Index f = 0; f < nb; ++f) hw(f) = (side(f) == 2 || side(f) == 3) ? 1 : 0;
    Kokkos::deep_copy(wall, hw);
    const ScalarField d = wallDistance(m, wall, m.cellCentre(), m.nCells());
    const ScalarField db = wallDistance(m, wall, m.boundaryCentre(), nb);
    auto hd = host(d); auto hdb = host(db);
    auto cc = host(m.cellCentre()); auto bc = host(m.boundaryCentre());
    Real e = 0.0, eb = 0.0;
    for (Index c = 0; c < m.nCells(); ++c)
      e = std::max(e, std::abs(hd(c) - std::min(cc(c,1), 1.0 - cc(c,1))));
    for (Index f = 0; f < nb; ++f)
      eb = std::max(eb, std::abs(hdb(f) - std::min(bc(f,1), 1.0 - bc(f,1))));
    const bool pass = e <= 1e-13 && eb <= 1e-13;
    ok &= pass;
    std::printf("  %s  max|d - d_exact| cells %.1e, boundary faces %.1e  %s\n",
                cs.tag.c_str(), e, eb, pass ? "PASS" : "FAIL");
  }
  return ok;
}

// ------------------------------------------------------------ 2. manufactured
struct Ms {
  ScalarFn k, w, p;
  VecFn u, a, srcU;
  ScalarFn srcK, srcW;
  Real nu;
};

Ms solution(char name, SstVariant v) {
  using namespace sst_ms;
  const bool y03 = v == SstVariant::Menter2003;
  Ms s;
  if (name == 'A') {
    s.k = [](const Vec3& q) { return k_A(q.x, q.y, q.z); };
    s.w = [](const Vec3& q) { return w_A(q.x, q.y, q.z); };
    s.p = [](const Vec3& q) { return p_A(q.x, q.y, q.z); };
    s.u = [](const Vec3& q) { return Vec3{u0_A(q.x, q.y, q.z), u1_A(q.x, q.y, q.z), u2_A(q.x, q.y, q.z)}; };
    s.a = [](const Vec3& q) { return Vec3{a0_A(q.x, q.y, q.z), a1_A(q.x, q.y, q.z), a2_A(q.x, q.y, q.z)}; };
    if (y03) {
      s.srcK = [](const Vec3& q) { return srck_A_2003(q.x, q.y, q.z); };
      s.srcW = [](const Vec3& q) { return srcw_A_2003(q.x, q.y, q.z); };
      s.srcU = [](const Vec3& q) { return Vec3{srcu0_A_2003(q.x, q.y, q.z), srcu1_A_2003(q.x, q.y, q.z), srcu2_A_2003(q.x, q.y, q.z)}; };
    } else {
      s.srcK = [](const Vec3& q) { return srck_A_1994(q.x, q.y, q.z); };
      s.srcW = [](const Vec3& q) { return srcw_A_1994(q.x, q.y, q.z); };
      s.srcU = [](const Vec3& q) { return Vec3{srcu0_A_1994(q.x, q.y, q.z), srcu1_A_1994(q.x, q.y, q.z), srcu2_A_1994(q.x, q.y, q.z)}; };
    }
    s.nu = nu_A;
  } else {
    s.k = [](const Vec3& q) { return k_B(q.x, q.y, q.z); };
    s.w = [](const Vec3& q) { return w_B(q.x, q.y, q.z); };
    s.p = [](const Vec3& q) { return p_B(q.x, q.y, q.z); };
    s.u = [](const Vec3& q) { return Vec3{u0_B(q.x, q.y, q.z), u1_B(q.x, q.y, q.z), u2_B(q.x, q.y, q.z)}; };
    s.a = [](const Vec3& q) { return Vec3{a0_B(q.x, q.y, q.z), a1_B(q.x, q.y, q.z), a2_B(q.x, q.y, q.z)}; };
    if (y03) {
      s.srcK = [](const Vec3& q) { return srck_B_2003(q.x, q.y, q.z); };
      s.srcW = [](const Vec3& q) { return srcw_B_2003(q.x, q.y, q.z); };
      s.srcU = [](const Vec3& q) { return Vec3{srcu0_B_2003(q.x, q.y, q.z), srcu1_B_2003(q.x, q.y, q.z), srcu2_B_2003(q.x, q.y, q.z)}; };
    } else {
      s.srcK = [](const Vec3& q) { return srck_B_1994(q.x, q.y, q.z); };
      s.srcW = [](const Vec3& q) { return srcw_B_1994(q.x, q.y, q.z); };
      s.srcU = [](const Vec3& q) { return Vec3{srcu0_B_1994(q.x, q.y, q.z), srcu1_B_1994(q.x, q.y, q.z), srcu2_B_1994(q.x, q.y, q.z)}; };
    }
    s.nu = nu_B;
  }
  return s;
}

// The wall at y = 0 (side 2), every face Dirichlet with the exact values.
void setupModel(PisoSolver& solver, const HexMesh& m, const Ms& s, SstVariant v) {
  const Index nb = m.nBoundaryFaces();
  auto side = host(m.boundarySide());
  View1<int> wall("wall", nb);
  auto hw = Kokkos::create_mirror_view(wall);
  for (Index f = 0; f < nb; ++f) hw(f) = side(f) == 2 ? 1 : 0;
  Kokkos::deep_copy(wall, hw);
  TurbulenceModel tm;
  tm.variant = v;
  tm.secondOrderAdvection = true;     // the order these gates judge
  solver.enableTurbulence(tm, wall);
  View1<int> kind("kind", nb);                       // all Dirichlet
  solver.setTurbulenceBoundary(kind, faceAverage(m, s.k), faceAverage(m, s.w));
  solver.setTurbulenceSource(cellScalar(m, s.srcK), cellScalar(m, s.srcW));
  solver.setTurbulence(cellScalar(m, s.k), cellScalar(m, s.w));
}

struct Result { Real eu = 0, ek = 0, ew = 0, change = 1e300; int steps = 0; long long bounded = 0; };

std::pair<Real, Real> kwErrors(const HexMesh& m, const PisoSolver& solver, const Ms& s) {
  const Index nc = m.nCells();
  auto k = toHost(solver.turbulentKineticEnergy(), nc);
  auto w = toHost(solver.specificDissipation(), nc);
  auto cc = host(m.cellCentre());
  std::vector<Real> ek(nc), ew(nc);
  for (Index c = 0; c < nc; ++c) {
    ek[c] = k[c] - s.k(centre(cc, c));
    ew[c] = w[c] - s.w(centre(cc, c));
  }
  return {l2(m, ek, 1), l2(m, ew, 1)};
}

Result runFrozen(char name, SstVariant v, const HexMesh& m, Real dt = 0.5, int maxSteps = 4000) {
  const Ms s = solution(name, v);
  const Index nc = m.nCells(), nb = m.nBoundaryFaces();
  PisoSolver solver(m, s.nu, dt);
  ScalarField fi, fb;
  fluxFromPotential(m, s.a, fi, fb);
  ScalarField p0("p0", m.nTotal());
  solver.setState(cellVector(m, s.u), p0, fi);
  setupModel(solver, m, s, v);
  VectorField ub;
  averageBoundaryValue(m, s.u, ub);
  NativeBiCGStab scalar(m);
  Result r;
  auto k = toHost(solver.turbulentKineticEnergy(), nc);
  auto w = toHost(solver.specificDissipation(), nc);
  for (int it = 0; it < maxSteps; ++it) {
    solver.advanceTurbulenceFrozen(ub, fb, scalar);
    auto kn = toHost(solver.turbulentKineticEnergy(), nc);
    auto wn = toHost(solver.specificDissipation(), nc);
    r.change = std::max(relChange(kn, k), relChange(wn, w));
    r.steps = it + 1;
    k = kn; w = wn;
    if (!(r.change >= STEADY) || !std::isfinite(k[0])) break;
  }
  (void)nb;
  std::tie(r.ek, r.ew) = kwErrors(m, solver, s);
  r.bounded = solver.boundedCells();
  return r;
}

Result runCoupled(SstVariant v, const HexMesh& m, Real dt = 0.5, int maxSteps = 4000) {
  const Ms s = solution('A', v);
  const Index nc = m.nCells();
  PisoControls ctl;
  ctl.outer = 3;
  ctl.correctors = 2;
  ctl.nonOrthTol = 1e-12;          // as heat_transfer's gate 2
  ctl.pressureSolveTol = 1e-13;
  PisoSolver solver(m, s.nu, dt, ctl);
  ScalarField fi, fbPot;
  fluxFromPotential(m, s.a, fi, fbPot);
  ScalarField p0 = cellScalar(m, s.p);
  {
    auto hp = host(p0);
    Real mean = 0.0;
    for (Index c = 0; c < nc; ++c) mean += hp(c);
    mean /= nc;
    for (Index c = 0; c < m.nTotal(); ++c) hp(c) -= mean;
    Kokkos::deep_copy(p0, hp);
  }
  solver.setState(cellVector(m, s.u), p0, fi);
  setupModel(solver, m, s, v);
  VectorField ub; ScalarField fb;
  averageBoundaryValue(m, s.u, ub);
  integrateBoundaryFlux(m, s.u, fb);
  adjustBoundaryFlux(m, fb);
  VectorField src = cellVector(m, s.srcU);
  NativeBiCGStab momentum(m);
  NativeCG pressure(m, Comm(), true);
  Result r;
  auto u = toHost(solver.velocity(), nc);
  auto k = toHost(solver.turbulentKineticEnergy(), nc);
  auto w = toHost(solver.specificDissipation(), nc);
  for (int it = 0; it < maxSteps; ++it) {
    solver.advance(ub, fb, src, momentum, pressure);
    auto un = toHost(solver.velocity(), nc);
    auto kn = toHost(solver.turbulentKineticEnergy(), nc);
    auto wn = toHost(solver.specificDissipation(), nc);
    r.change = std::max({relChange(un, u), relChange(kn, k), relChange(wn, w)});
    r.steps = it + 1;
    u = un; k = kn; w = wn;
    if (!(r.change >= STEADY) || !std::isfinite(u[0])) break;
  }
  auto cc = host(m.cellCentre());
  std::vector<Real> eu(nc * 3);
  for (Index c = 0; c < nc; ++c) {
    const Vec3 ue = s.u(centre(cc, c));
    eu[c*3] = u[c*3] - ue.x; eu[c*3+1] = u[c*3+1] - ue.y; eu[c*3+2] = u[c*3+2] - ue.z;
  }
  r.eu = l2(m, eu, 3);
  std::tie(r.ek, r.ew) = kwErrors(m, solver, s);
  r.bounded = solver.boundedCells();
  return r;
}

const char* variantName(SstVariant v) { return v == SstVariant::Menter2003 ? "2003" : "1994"; }

template <class Run>
bool study(const std::string& title, const std::vector<Index>& grids, bool withU, Run run) {
  std::printf("\n%s\n", title.c_str());
  std::fflush(stdout);
  bool ok = true;
  struct Fam { Real skew, lo, hi; bool approaching; const char* tag; };
  for (const Fam& fm : {Fam{0.0, 1.85, 2.15, false, "orthogonal"},
                        Fam{0.25, 1.6, 2.3, true, "smooth distortion"}}) {
    std::vector<Real> eu, ek, ew, hs;
    for (Index n : grids) {
      const auto t0 = std::chrono::steady_clock::now();
      const HexMesh m = family(n, fm.skew);
      const Result r = run(m);
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      ok &= r.change < STEADY && r.bounded == 0;
      eu.push_back(r.eu); ek.push_back(r.ek); ew.push_back(r.ew);
      hs.push_back(1.0 / n);
      std::printf("  %-18s n=%-3d steps %-5d ", fm.tag, n, r.steps);
      if (withU) std::printf("L2(u) %.10e  ", r.eu);
      std::printf("L2(k) %.10e  L2(w) %.10e  change %.0e  bounded %lld  (%.0f s)\n",
                  r.ek, r.ew, r.change, r.bounded, secs);
      std::fflush(stdout);
    }
    if (grids.size() > 1) {
      if (withU) ok &= verdict(std::string("u, ") + fm.tag, orders(eu, hs), fm.lo, fm.hi, fm.approaching);
      ok &= verdict(std::string("k, ") + fm.tag, orders(ek, hs), fm.lo, fm.hi, fm.approaching);
      ok &= verdict(std::string("w, ") + fm.tag, orders(ew, hs), fm.lo, fm.hi, fm.approaching);
    }
  }
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const std::string fixtures = argc > 1 ? argv[1] : "tests/fixtures";
    const std::string which = argc > 2 ? argv[2] : "1ab";
    std::vector<Index> grids;
    for (int i = 3; i < argc; ++i) grids.push_back(static_cast<Index>(std::atoi(argv[i])));
    if (grids.empty()) grids = {8, 16, 32};
    bool ok = true;
    auto runGate = [&](char key, const std::function<bool()>& fn) {
      if (which.find(key) == std::string::npos) return;
      try {
        ok &= fn();
      } catch (const std::exception& e) {
        std::printf("  FAIL: %s\n", e.what());
        ok = false;
      }
    };
    runGate('1', [&] { return gateWallDistance(fixtures); });
    runGate('a', [&] {
      bool g = true;
      for (char name : {'A', 'B'})
        for (SstVariant v : {SstVariant::Menter2003, SstVariant::Menter1994})
          g &= study(std::string("2a. frozen velocity, MS-") + name + ", SST-" + variantName(v),
                     grids, false, [&](const HexMesh& m) { return runFrozen(name, v, m); });
      return g;
    });
    runGate('b', [&] {
      bool g = true;
      for (SstVariant v : {SstVariant::Menter2003, SstVariant::Menter1994})
        g &= study(std::string("2b. coupled, MS-A, SST-") + variantName(v), grids, true,
                   [&](const HexMesh& m) { return runCoupled(v, m); });
      return g;
    });
    std::printf("\nv2b SST GATE (C++): %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
  return rc;
}
