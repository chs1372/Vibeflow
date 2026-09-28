// v2a gate 4 (ADR-038): the onset of Rayleigh-Benard convection.
//
// Rigid, isothermal plates one unit apart -- hot (T = 1) at z = 0, cold
// (T = 0) at z = 1 -- and slip, adiabatic side walls pi/k_c apart. Those walls
// hold exactly one roll of the infinite layer's critical mode: u_x ~
// sin(k_c x), u_z and T ~ cos(k_c x) meet the slip and zero-flux conditions
// at x = 0 and x = pi/k_c. The roll is two-dimensional, so the domain is a
// one-cell slab in y with slip, adiabatic faces.
//
// Pr = 1, in units of the gap d and the diffusion time d^2/kappa: kappa = nu
// = 1 and betaG = (0, 0, -Ra). The reference stratification is the
// conduction profile T = 1 - z, which makes the conduction state an exact
// discrete fixed point (gate 3), so a small perturbation grows or decays as
// the linear modes of the DISCRETE equations. Its growth rate at Ra = 1600,
// 1700 and 1800 is measured from a temperature perturbation of 1e-6 in the
// critical mode's shape, on meshes of 16, 24 and 32 cells across the layer;
// each mesh's critical Ra is the zero of the quadratic through its three
// (Ra, growth rate) points.
//
// The zero does not depend on the time step or on the outer-iteration count.
// A neutral mode is a steady solution of the linearised discrete equations,
// and the solver's steady states do not depend on either (ADR-037) -- so the
// marching parameters move the measured growth rates a little away from the
// zero and not at all at it.
//
// Reference: Ra_c = 1707.762 at k_c = 3.117 (Chandrasekhar; Scholarpedia's
// Rayleigh-Benard article). Pass: the finest mesh within 1%, the observed
// order in [1.5, 2.6], the Richardson extrapolation within 0.3%.
//
// Run:  rayleigh_benard [N...]      default 16 24 32

#include "mesh/HexMesh.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {

constexpr Real PI = 3.14159265358979323846;
constexpr Real KC = 3.117;
constexpr Real RA_C = 1707.762;

template <class V> auto host(const V& v) {
  return Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
}

struct Growth { Real sigma, sigmaEarly, sigmaLate; int steps; };

// Growth rate of the perturbation's velocity norm, d ln|u| / dt, fitted by
// least squares over [t0, t1]; also over each half of it, as a check that the
// transient is gone.
Growth growthRate(Index N, Real Ra, Real dt = 0.01, Real t0 = 1.5, Real t1 = 3.5) {
  const Real Lx = PI / KC;
  const HexMesh mesh = HexMesh::box(N, 1, N, Lx, 1.0 / N, 1.0);
  const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();

  PisoControls ctl;
  ctl.outer = 4;           // fixed: the neutral point does not depend on it
  ctl.outerTol = 0.0;
  ctl.correctors = 2;
  PisoSolver solver(mesh, 1.0, dt, ctl);
  EnergyModel em;
  em.kappa = 1.0;
  em.betaG = {0.0, 0.0, -Ra};
  em.tRef = 1.0;
  em.tRefGrad = {0.0, 0.0, -1.0};
  solver.enableEnergy(em);

  // Sides: 0 x-, 1 x+, 2 y-, 3 y+, 4 z- (hot plate), 5 z+ (cold plate).
  auto side = host(mesh.boundarySide());
  View1<int> uType("uType", nb), tType("tType", nb);
  ScalarField tValue("tValue", nb);
  auto hu = Kokkos::create_mirror_view(uType);
  auto ht = Kokkos::create_mirror_view(tType);
  auto hv = Kokkos::create_mirror_view(tValue);
  for (Index f = 0; f < nb; ++f) {
    const bool plate = side(f) >= 4;
    hu(f) = static_cast<int>(plate ? VelocityBC::Dirichlet : VelocityBC::Slip);
    ht(f) = static_cast<int>(plate ? TemperatureBC::FixedValue : TemperatureBC::FixedFlux);
    hv(f) = side(f) == 4 ? 1.0 : 0.0;       // T = 1 below, 0 above, no flux elsewhere
  }
  Kokkos::deep_copy(uType, hu); Kokkos::deep_copy(tType, ht); Kokkos::deep_copy(tValue, hv);
  solver.setBoundaryTypes(uType);
  solver.setTemperatureBoundary(tType, tValue);

  auto cc = host(mesh.cellCentre());
  ScalarField T0("T0", nt);
  {
    auto h = Kokkos::create_mirror_view(T0);
    for (Index c = 0; c < nt; ++c)
      h(c) = 1.0 - cc(c,2) + 1e-6 * std::cos(PI * cc(c,0) / Lx) * std::sin(PI * cc(c,2));
    Kokkos::deep_copy(T0, h);
  }
  solver.setTemperature(T0);

  VectorField ub("ub", nb, 3), src("src", nt, 3);
  ScalarField fb("fb", nb);
  NativeBiCGStab momentum(mesh);
  NativeCG pressure(mesh, Comm(), true);
  auto vol = host(mesh.cellVolume());

  std::vector<Real> ts, ls;
  const int steps = static_cast<int>(std::lround(t1 / dt));
  for (int k = 0; k < steps; ++k) {
    solver.advance(ub, fb, src, momentum, pressure);
    const Real t = (k + 1) * dt;
    if (t < t0 - 1e-9) continue;
    auto u = host(solver.velocity());
    Real e = 0.0;
    for (Index c = 0; c < nc; ++c)
      e += (u(c,0)*u(c,0) + u(c,1)*u(c,1) + u(c,2)*u(c,2)) * vol(c);
    ts.push_back(t);
    ls.push_back(0.5 * std::log(e));
  }
  auto slope = [&](std::size_t a, std::size_t b) {
    Real st = 0.0, sl = 0.0, stt = 0.0, stl = 0.0;
    const Real n = static_cast<Real>(b - a);
    for (std::size_t i = a; i < b; ++i) {
      st += ts[i]; sl += ls[i]; stt += ts[i] * ts[i]; stl += ts[i] * ls[i];
    }
    return (n * stl - st * sl) / (n * stt - st * st);
  };
  const std::size_t m = ts.size() / 2;
  return {slope(0, ts.size()), slope(0, m), slope(m, ts.size()), steps};
}

// Zero of the quadratic through three (x, y) points, nearest to x[1].
Real quadraticZero(const Real x[3], const Real y[3]) {
  // Newton form around x1: y = y1 + b (x - x1) + c (x - x1)^2
  const Real d0 = x[0] - x[1], d2 = x[2] - x[1];
  const Real c = ((y[2] - y[1]) / d2 - (y[0] - y[1]) / d0) / (d2 - d0);
  const Real b = (y[2] - y[1]) / d2 - c * d2;
  if (std::abs(c) < 1e-300) return x[1] - y[1] / b;
  const Real disc = b * b - 4.0 * c * y[1];
  if (disc < 0.0) return std::nan("");
  const Real r1 = (-b + std::sqrt(disc)) / (2.0 * c), r2 = (-b - std::sqrt(disc)) / (2.0 * c);
  return x[1] + (std::abs(r1) < std::abs(r2) ? r1 : r2);
}

// Observed order from three meshes with arbitrary refinement ratios:
// (f1 - f2)/(f2 - f3) = (h1^p - h2^p)/(h2^p - h3^p), solved by bisection.
Real observedOrder(const Real h[3], const Real f[3]) {
  const Real d1 = f[0] - f[1], d2 = f[1] - f[2];
  if (d1 * d2 <= 0.0) return std::nan("");
  const Real target = d1 / d2;
  auto g = [&](Real p) {
    return (std::pow(h[0], p) - std::pow(h[1], p)) / (std::pow(h[1], p) - std::pow(h[2], p))
           - target;
  };
  Real lo = 0.05, hi = 12.0;
  if (g(lo) * g(hi) > 0.0) return std::nan("");
  for (int i = 0; i < 200; ++i) {
    const Real mid = 0.5 * (lo + hi);
    (g(lo) * g(mid) <= 0.0 ? hi : lo) = mid;
  }
  return 0.5 * (lo + hi);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::vector<Index> grids;
    for (int i = 1; i < argc; ++i) grids.push_back(std::stoi(argv[i]));
    if (grids.empty()) grids = {16, 24, 32};
    const Real ras[3] = {1600.0, 1700.0, 1800.0};
    std::printf("Rayleigh-Benard onset: rigid plates, one roll between slip walls pi/k_c "
                "apart, Pr = 1\n");
    bool ok = true;
    std::vector<Real> hs, rcs;
    try {
      for (Index N : grids) {
        Real sig[3];
        for (int i = 0; i < 3; ++i) {
          const Growth g = growthRate(N, ras[i]);
          sig[i] = g.sigma;
          std::printf("  N=%-3d Ra %6.0f  growth rate %+.6e   halves %+.6e %+.6e\n",
                      N, ras[i], g.sigma, g.sigmaEarly, g.sigmaLate);
          // The fit window must be past the transient: its two halves agree.
          if (std::abs(g.sigmaEarly - g.sigmaLate) > 1e-3) {
            std::printf("  FAIL: growth rate not settled over the fit window\n");
            ok = false;
          }
        }
        const Real rc_ = quadraticZero(ras, sig);
        hs.push_back(1.0 / N);
        rcs.push_back(rc_);
        std::printf("  N=%-3d critical Ra %.3f   (%+.3f%% from %.3f)\n", N, rc_,
                    100.0 * (rc_ - RA_C) / RA_C, RA_C);
      }
    } catch (const std::exception& e) {
      std::printf("  FAIL: %s\n", e.what());
      ok = false;
    }
    if (ok && rcs.size() == 3) {
      const Real fine = rcs[2];
      const bool fineOk = std::abs(fine - RA_C) / RA_C <= 0.01;
      const Real h[3] = {hs[0], hs[1], hs[2]}, f[3] = {rcs[0], rcs[1], rcs[2]};
      const Real p = observedOrder(h, f);
      const bool orderOk = std::isfinite(p) && p >= 1.5 && p <= 2.6;
      Real ext = std::nan("");
      if (std::isfinite(p)) ext = f[2] + (f[2] - f[1]) / (std::pow(h[1] / h[2], p) - 1.0);
      const bool extOk = std::isfinite(ext) && std::abs(ext - RA_C) / RA_C <= 0.003;
      std::printf("  -> finest mesh %.3f, %+.3f%% (within 1%%): %s\n", fine,
                  100.0 * (fine - RA_C) / RA_C, fineOk ? "PASS" : "FAIL");
      std::printf("  -> observed order %.3f (in [1.5, 2.6]): %s\n", p, orderOk ? "PASS" : "FAIL");
      std::printf("  -> Richardson extrapolation %.3f, %+.3f%% (within 0.3%%): %s\n", ext,
                  100.0 * (ext - RA_C) / RA_C, extOk ? "PASS" : "FAIL");
      ok = fineOk && orderOk && extOk;
    } else if (ok) {
      std::printf("  (fewer than three meshes: reported, not judged)\n");
    }
    std::printf("\nv2a Rayleigh-Benard onset GATE (C++): %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
  return rc;
}
