// Momentum's convected face value, verified where convection matters
// (ADR-042, the revision on momentum advection).
//
// Ethier-Steinman, the v1 order gate, runs two steps from the exact field, so
// the error it measures is mostly the projection's: its linear and
// linear-upwind errors differ by 1.5e-4 relative, and it cannot tell one face
// value from the other. This gate runs the steady manufactured flow of
// steady_dt instead -- the solenoidal field
//
//   u = (sin px (cos py - cos pz), sin py (cos pz - cos px), sin pz (cos px - cos py)),
//   p = cos px cos py cos pz,   s = (u . grad) u - nu laplacian(u) + grad p,
//
// (p = pi times the coordinate), both face values, both mesh families, each
// marched from the exact field to a steady state (the largest change of u or
// p over a step below 1e-8, as heat_transfer's order runs) at a Courant
// number of 3.2, dt = 1.6 / n, on 6^3, 12^3 and 24^3.
//
// First run at nu = 0.02 -- cell Peclet numbers 17, 8 and 4 -- and failed by
// its own criteria for both face values, the long-verified linear one too:
// the meshes are short of the asymptotic range there, the orders come down
// towards 2 from above (linear 2.30, 2.14; linear upwind 2.60, 2.35 on the
// orthogonal family), and the linear-upwind error is 1.7 to 2.5 times
// smaller than the linear one on every mesh. Kept as `--nu 0.02`, reported.
// Revised after it (ADR-042), the gate runs at nu = 0.1, steady_dt's and the
// v2a steady gate's viscosity (cell Peclet 3.3, 1.7 and 0.8), where the v1
// and v2a gates show the linear face value asymptotic on these meshes, with a
// condition that the check sees the face value at all:
//
//   * the order of u between the two finest meshes in [1.85, 2.15] on the
//     orthogonal family, and in [1.6, 2.3] approaching 2 on the smooth
//     distortion (ADR-038's form -- the first run took v1's "rising"), for
//     each face value;
//   * on every mesh the two face values' errors at least 5% apart;
//   * every run steady.
//
// Run:  convection_order [--nu NU, default 0.1] [grids, default 6 12 24]

#include "mesh/HexMesh.hpp"
#include "discretization/FaceFlux.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {

constexpr Real PI = 3.14159265358979323846;
Real NU = 0.1;   // --nu

Vec3 velocity(const Vec3& q) {
  const Real sx = std::sin(PI*q.x), sy = std::sin(PI*q.y), sz = std::sin(PI*q.z);
  const Real cx = std::cos(PI*q.x), cy = std::cos(PI*q.y), cz = std::cos(PI*q.z);
  return {sx * (cy - cz), sy * (cz - cx), sz * (cx - cy)};
}

// g[i][j] = d u_i / d x_j; component i is sin(pi x_i)(cos(pi x_j) - cos(pi x_k)),
// (i, j, k) cyclic.
void velocityGradient(const Vec3& q, Real g[3][3]) {
  const Real x[3] = {q.x, q.y, q.z};
  for (int i = 0; i < 3; ++i) {
    const int j = (i + 1) % 3, k = (i + 2) % 3;
    const Real si = std::sin(PI*x[i]), ci = std::cos(PI*x[i]);
    const Real sj = std::sin(PI*x[j]), cj = std::cos(PI*x[j]);
    const Real sk = std::sin(PI*x[k]), ck = std::cos(PI*x[k]);
    g[i][i] = PI * ci * (cj - ck);
    g[i][j] = -PI * si * sj;
    g[i][k] = PI * si * sk;
  }
}

Real pressureExact(const Vec3& q) {
  return std::cos(PI*q.x) * std::cos(PI*q.y) * std::cos(PI*q.z);
}

// (u . grad) u - nu laplacian(u) + grad p, laplacian(u) = -2 pi^2 u.
Vec3 source(const Vec3& q) {
  const Vec3 u = velocity(q);
  const Real uu[3] = {u.x, u.y, u.z};
  Real g[3][3];
  velocityGradient(q, g);
  Real s[3];
  for (int i = 0; i < 3; ++i) {
    s[i] = 2.0 * PI * PI * NU * uu[i];
    for (int j = 0; j < 3; ++j) s[i] += uu[j] * g[i][j];
  }
  const Real sx = std::sin(PI*q.x), sy = std::sin(PI*q.y), sz = std::sin(PI*q.z);
  const Real cx = std::cos(PI*q.x), cy = std::cos(PI*q.y), cz = std::cos(PI*q.z);
  s[0] -= PI * sx * cy * cz;
  s[1] -= PI * cx * sy * cz;
  s[2] -= PI * cx * cy * sz;
  return {s[0], s[1], s[2]};
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

struct Run { Real l2 = 0.0, change = 1e300, seconds = 0.0; int steps = 0; };

Run steady(const HexMesh& mesh, ConvectionScheme scheme, Real dt, Real tol, Real maxTime) {
  const auto t0 = std::chrono::steady_clock::now();
  const Index nc = mesh.nCells(), nt = mesh.nTotal();
  PisoControls ctl;
  ctl.outer = 2;
  ctl.correctors = 2;
  ctl.nonOrthTol = 1e-12;          // heat_transfer's order runs: the state then
  ctl.pressureSolveTol = 1e-13;    // wobbles far below the 1e-8 called steady
  ctl.convection = scheme;
  PisoSolver solver(mesh, NU, dt, ctl);

  auto cc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellCentre());
  VectorField src("src", nt, 3), u0("u0", nt, 3);
  ScalarField p0("p0", nt), F0("F0", mesh.nInternalFaces());
  {
    auto hs = Kokkos::create_mirror_view(src);
    auto hu = Kokkos::create_mirror_view(u0);
    auto hp = Kokkos::create_mirror_view(p0);
    for (Index i = 0; i < nt; ++i) {
      const Vec3 q{cc(i,0), cc(i,1), cc(i,2)};
      const Vec3 s = source(q), v = velocity(q);
      hs(i,0) = s.x; hs(i,1) = s.y; hs(i,2) = s.z;
      hu(i,0) = v.x; hu(i,1) = v.y; hu(i,2) = v.z;
      hp(i) = pressureExact(q);
    }
    Kokkos::deep_copy(src, hs); Kokkos::deep_copy(u0, hu); Kokkos::deep_copy(p0, hp);
  }
  {
    // The face flux from the cells, weighted as ethier_steinman's is.
    auto own = mesh.owner(); auto nei = mesh.neighbour();
    auto fa = mesh.faceArea(); auto fc = mesh.faceCentre(); auto ccd = mesh.cellCentre();
    Kokkos::parallel_for("F0", Kokkos::RangePolicy<ExecSpace>(0, mesh.nInternalFaces()),
      KOKKOS_LAMBDA(const Index f) {
        Real lo = 0.0, ln = 0.0;
        for (int i = 0; i < 3; ++i) {
          const Real ro = fc(f, i) - ccd(own(f), i), rn = fc(f, i) - ccd(nei(f), i);
          lo += ro * ro; ln += rn * rn;
        }
        const Real w = Kokkos::sqrt(ln) / (Kokkos::sqrt(lo) + Kokkos::sqrt(ln));
        Real s = 0.0;
        for (int i = 0; i < 3; ++i) s += (w * u0(own(f), i) + (1.0 - w) * u0(nei(f), i)) * fa(f, i);
        F0(f) = s;
      });
    Kokkos::fence();
  }
  solver.setState(u0, p0, F0);
  VectorField ub; ScalarField fb;
  averageBoundaryValue(mesh, velocity, ub);
  integrateBoundaryFlux(mesh, velocity, fb);
  adjustBoundaryFlux(mesh, fb);

  NativeBiCGStab momentum(mesh);
  NativeCG pressure(mesh, Comm(), true);
  std::vector<Real> prevU(nc * 3, 0.0), prevP(nc, 0.0);
  Run r;
  const int maxSteps = static_cast<int>(std::lround(maxTime / dt));
  for (int k = 0; k < maxSteps; ++k) {
    solver.advance(ub, fb, src, momentum, pressure);
    r.steps = k + 1;
    auto hu = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), solver.velocity());
    auto hp = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), solver.pressure());
    Real change = 0.0;
    for (Index c = 0; c < nc; ++c) {
      for (int d = 0; d < 3; ++d) {
        change = std::max(change, std::abs(hu(c,d) - prevU[c*3 + d]));
        prevU[c*3 + d] = hu(c,d);
      }
      change = std::max(change, std::abs(hp(c) - prevP[c]));
      prevP[c] = hp(c);
    }
    if (k > 0) {
      r.change = change;
      if (change < tol || !std::isfinite(change)) break;
    }
  }
  auto vol = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellVolume());
  Real e = 0.0, v = 0.0;
  for (Index c = 0; c < nc; ++c) {
    const Vec3 ue = velocity({cc(c,0), cc(c,1), cc(c,2)});
    const Real d0 = prevU[c*3] - ue.x, d1 = prevU[c*3+1] - ue.y, d2 = prevU[c*3+2] - ue.z;
    e += (d0*d0 + d1*d1 + d2*d2) * vol(c);
    v += vol(c);
  }
  r.l2 = std::sqrt(e / v);
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::vector<Index> grids;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--nu" && i + 1 < argc) { NU = std::atof(argv[++i]); continue; }
      grids.push_back(std::stoi(a));
    }
    if (grids.empty()) grids = {6, 12, 24};
    const Real tol = 1e-8, maxTime = 200.0;
    std::printf("Momentum's face value, steady manufactured flow at nu = %.3g "
                "(Courant 3.2, steady below %.0e)\n", NU, tol);
    struct Fam { Real skew; const char* mode; Real lo, hi; bool approaching; const char* tag; };
    struct Sch { ConvectionScheme s; const char* tag; };
    const Fam fams[2] = {Fam{0.0, "none", 1.85, 2.15, false, "orthogonal"},
                         Fam{0.25, "smooth", 1.6, 2.3, true, "smooth distortion"}};
    const Sch schemes[2] = {Sch{ConvectionScheme::Linear, "linear"},
                            Sch{ConvectionScheme::LinearUpwind, "linear upwind"}};
    std::vector<Real> err[2][2];   // [scheme][family], one per grid
    bool ok = true;
    for (int si = 0; si < 2; ++si) {
      for (int fi = 0; fi < 2; ++fi) {
        const Sch& sc = schemes[si];
        const Fam& fm = fams[fi];
        std::printf("\n  %s, %s\n", sc.tag, fm.tag);
        std::vector<Real>& e = err[si][fi];
        std::vector<Real> h, orders;
        bool steadyAll = true;
        for (Index n : grids) {
          const HexMesh mesh = HexMesh::generate(n, fm.skew, fm.mode);
          const Run r = steady(mesh, sc.s, 1.6 / static_cast<Real>(n), tol, maxTime);
          e.push_back(r.l2); h.push_back(1.0 / n);
          char o[16] = "     -";
          if (e.size() > 1) {
            orders.push_back(std::log(e[e.size()-2] / e.back()) / std::log(h[h.size()-2] / h.back()));
            std::snprintf(o, sizeof o, "%6.3f", orders.back());
          }
          steadyAll &= r.change < tol;
          std::printf("    n %3d  steps %5d  last change %.1e  L2(u) %.10e  order %s  (%.0f s)\n",
                      static_cast<int>(n), r.steps, r.change, r.l2, o, r.seconds);
          std::fflush(stdout);
        }
        bool pass = steadyAll && !orders.empty();
        if (!orders.empty()) {
          const Real last = orders.back();
          pass &= last >= fm.lo && last <= fm.hi;
          bool approach = true;
          if (fm.approaching)
            for (std::size_t i = 1; i < orders.size(); ++i)
              approach &= std::abs(orders[i] - 2.0) <= std::abs(orders[i-1] - 2.0) + 0.02;
          pass &= approach;
          std::printf("    -> order %.3f in [%.2f, %.2f]%s%s: %s\n", last, fm.lo, fm.hi,
                      fm.approaching ? (approach ? ", approaching 2" : ", MOVING AWAY FROM 2") : "",
                      steadyAll ? "" : ", NOT STEADY", pass ? "PASS" : "FAIL");
        }
        ok &= pass;
      }
    }
    // The check must see the face value: Ethier-Steinman's two errors are
    // 1.5e-4 apart.
    std::printf("\n  linear upwind against linear, L2(u) ratio per mesh\n");
    for (int fi = 0; fi < 2; ++fi) {
      bool apart = true;
      std::printf("    %-18s", fams[fi].tag);
      for (std::size_t g = 0; g < grids.size(); ++g) {
        const Real ratio = err[1][fi][g] / err[0][fi][g];
        apart &= std::abs(ratio - 1.0) >= 0.05;
        std::printf("  n %d: %.4f", static_cast<int>(grids[g]), ratio);
      }
      std::printf("  -> at least 5%% apart on every mesh: %s\n", apart ? "PASS" : "FAIL");
      ok &= apart;
    }
    std::printf("\nv2b momentum face value GATE (C++): %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
  return rc;
}
