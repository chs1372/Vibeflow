#pragma once
// The Rayleigh-Benard setup shared by the onset gate (ADR-038) and the
// transient-coupling gate (ADR-040).
//
// Rigid, isothermal plates one unit apart -- hot (T = 1) at z = 0, cold
// (T = 0) at z = 1 -- and slip, adiabatic side walls pi/k_c apart, which hold
// one roll of the infinite layer's critical mode; a one-cell slab in y with
// slip, adiabatic faces. Pr = 1 in units of the gap and the diffusion time:
// kappa = nu = 1, betaG = (0, 0, -Ra), the reference stratification the
// conduction profile, so the conduction state is an exact discrete fixed
// point. A temperature perturbation of 1e-6 in the critical mode's shape
// grows or decays as the linear modes of the discrete equations.

#include "mesh/HexMesh.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace vibeflow::rb {

constexpr Real PI = 3.14159265358979323846;
constexpr Real KC = 3.117;

template <class V> auto host(const V& v) {
  return Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
}

// The buoyancy form the gate runs (ADR-041); VIBEFLOW_BUOYANCY=cell selects
// ADR-038's cell force, the recorded baseline.
inline BuoyancyForm gateForm() {
  const char* e = std::getenv("VIBEFLOW_BUOYANCY");
  return (e && std::string(e) == "cell") ? BuoyancyForm::Cell : BuoyancyForm::Balanced;
}

// How the march is run. The onset gate's values are the defaults.
struct Marching {
  Real dt = 0.01;
  int outer = 4;
  Real outerTol = 0.0;
  Real pressureSolveTol = -1.0;     // < 0: the solver's default
};

struct Growth {
  Real sigma, sigmaEarly, sigmaLate;
  int steps;
  int maxOuterUsed;                 // most outer iterations any step used
};

// Growth rate of the perturbation's velocity norm, d ln|u| / dt, fitted by
// least squares over [t0, t1]; also over each half of it, as a check that the
// transient is gone.
inline Growth growthRate(Index N, Real Ra, const Marching& mk = {}, Real t0 = 1.5,
                         Real t1 = 3.5) {
  const Real Lx = PI / KC;
  const HexMesh mesh = HexMesh::box(N, 1, N, Lx, 1.0 / N, 1.0);
  const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();

  PisoControls ctl;
  ctl.outer = mk.outer;
  ctl.outerTol = mk.outerTol;
  ctl.correctors = 2;
  if (mk.pressureSolveTol > 0.0) ctl.pressureSolveTol = mk.pressureSolveTol;
  PisoSolver solver(mesh, 1.0, mk.dt, ctl);
  EnergyModel em;
  em.kappa = 1.0;
  em.betaG = {0.0, 0.0, -Ra};
  em.tRef = 1.0;
  em.tRefGrad = {0.0, 0.0, -1.0};
  em.form = gateForm();
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
  const int steps = static_cast<int>(std::lround(t1 / mk.dt));
  int maxOuter = 0;
  for (int k = 0; k < steps; ++k) {
    const StepReport rep = solver.advance(ub, fb, src, momentum, pressure);
    maxOuter = std::max(maxOuter, rep.outerUsed);
    const Real t = (k + 1) * mk.dt;
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
  return {slope(0, ts.size()), slope(0, m), slope(m, ts.size()), steps, maxOuter};
}

// Observed order from three meshes with arbitrary refinement ratios:
// (f1 - f2)/(f2 - f3) = (h1^p - h2^p)/(h2^p - h3^p), solved by bisection.
inline Real observedOrder(const Real h[3], const Real f[3]) {
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

}  // namespace vibeflow::rb
