// v2a gate 5 (ADR-038): the differentially heated square cavity of de Vahl
// Davis (1983).
//
// Air, Pr = 0.71, Ra = 1e3, 1e4, 1e5 and 1e6. The hot wall x = 0 is at T = 1,
// the cold wall x = 1 at T = 0, the top and bottom are adiabatic, and every
// wall is no-slip. The problem is two-dimensional and runs as a one-cell-thick
// slab in z with slip, adiabatic faces. In units of the side L and the
// velocity kappa/L -- de Vahl Davis's -- kappa = 1, nu = Pr and betaG =
// (0, -Ra Pr, 0) with a constant reference T_ref = 0.5.
//
// Each case runs from rest in the conduction profile to a steady state on
// uniform meshes of 32^2, 64^2 and 128^2. Pass:
//   * the hot wall's mean Nusselt number, Richardson-extrapolated from the
//     three meshes, within 0.5% of 1.1178, 2.2448, 4.5216 and 8.8252 (Wang et
//     al., Hortmann et al., Le Quere, as tabulated by a lattice Boltzmann
//     validation study that cites them) -- the extrapolation only with an
//     observed order in [0.5, 4], as ADR-033 used it; outside that the
//     differences carry no order and the check fails;
//   * the same Nusselt number on 128^2 within 1%;
//   * on 128^2, the largest horizontal velocity on the vertical mid-line and
//     the largest vertical velocity on the horizontal mid-line within 1% of
//     de Vahl Davis's 3.649 / 3.697, 16.178 / 19.617, 34.73 / 68.59 and
//     64.63 / 219.36.
// The Nusselt number is read from the solver's own wall heat flux -- the
// discrete flux the energy equation applies there -- not from a separately
// differenced gradient. With the mesh's two columns straddling the mid-line
// averaged, each maximum is the vertex of the parabola through the largest
// sample and its two neighbours, as de Vahl Davis located his.
//
// Steady: over a step, the largest change of T and of u / u_ref (u_ref the
// reference vertical maximum), each divided by dt, below 1e-5 -- a state
// within about 1e-5 / (slowest decay rate) of the steady one, far inside the
// tolerances above. The time step is a Courant number of about one on the
// reference velocity; the steady state does not depend on it (ADR-037).
//
// Run:  heated_cavity [Ra ...] [-n N ...]    default all four, 32 64 128

#include "mesh/HexMesh.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {

struct Reference { Real ra, nu, uMax, vMax; };
const Reference REF[] = {
  {1e3, 1.1178, 3.649, 3.697},
  {1e4, 2.2448, 16.178, 19.617},
  {1e5, 4.5216, 34.73, 68.59},
  {1e6, 8.8252, 64.63, 219.36},
};
constexpr Real PR = 0.71;

template <class V> auto host(const V& v) {
  return Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
}

struct Result { Real nusselt, uMax, vMax; int steps; Real resid; double seconds; };

// Vertex of the parabola through (i-1, i, i+1) of a uniformly spaced profile.
Real peak(const std::vector<Real>& v) {
  std::size_t i = 0;
  for (std::size_t k = 1; k < v.size(); ++k) if (v[k] > v[i]) i = k;
  if (i == 0 || i + 1 == v.size()) return v[i];
  const Real a = v[i-1], b = v[i], c = v[i+1];
  const Real den = a - 2.0 * b + c;
  if (den == 0.0) return b;
  const Real s = 0.5 * (a - c) / den;
  return b - 0.25 * (a - c) * s;
}

Result run(Index N, const Reference& ref, bool verbose) {
  const auto t0 = std::chrono::steady_clock::now();
  const HexMesh mesh = HexMesh::box(N, N, 1, 1.0, 1.0, 1.0 / N);
  const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();
  const Real h = 1.0 / N;
  const Real dt = h / ref.vMax;          // Courant number about one

  PisoControls ctl;
  ctl.outer = 3;
  ctl.outerTol = 1e-8;
  ctl.correctors = 2;
  PisoSolver solver(mesh, PR, dt, ctl);
  EnergyModel em;
  em.kappa = 1.0;
  em.betaG = {0.0, -ref.ra * PR, 0.0};
  em.tRef = 0.5;
  solver.enableEnergy(em);

  // Sides: 0 x- (hot), 1 x+ (cold), 2 y-, 3 y+ (adiabatic), 4 z-, 5 z+ (slab).
  auto side = host(mesh.boundarySide());
  View1<int> uType("uType", nb), tType("tType", nb);
  ScalarField tValue("tValue", nb);
  View1<int> hot("hot", nb);
  auto hu = Kokkos::create_mirror_view(uType);
  auto ht = Kokkos::create_mirror_view(tType);
  auto hv = Kokkos::create_mirror_view(tValue);
  auto hh = Kokkos::create_mirror_view(hot);
  for (Index f = 0; f < nb; ++f) {
    const int s = side(f);
    hu(f) = static_cast<int>(s >= 4 ? VelocityBC::Slip : VelocityBC::Dirichlet);
    ht(f) = static_cast<int>(s <= 1 ? TemperatureBC::FixedValue : TemperatureBC::FixedFlux);
    hv(f) = s == 0 ? 1.0 : 0.0;
    hh(f) = s == 0 ? 1 : 0;
  }
  Kokkos::deep_copy(uType, hu); Kokkos::deep_copy(tType, ht);
  Kokkos::deep_copy(tValue, hv); Kokkos::deep_copy(hot, hh);
  solver.setBoundaryTypes(uType);
  solver.setTemperatureBoundary(tType, tValue);

  auto cc = host(mesh.cellCentre());
  {
    ScalarField T0("T0", nt);
    auto h0 = Kokkos::create_mirror_view(T0);
    for (Index c = 0; c < nt; ++c) h0(c) = 1.0 - cc(c,0);    // conduction profile
    Kokkos::deep_copy(T0, h0);
    solver.setTemperature(T0);
  }

  VectorField ub("ub", nb, 3), src("src", nt, 3);
  ScalarField fb("fb", nb);
  NativeBiCGStab momentum(mesh);
  NativeCG pressure(mesh, Comm(), true);

  std::vector<Real> pu(nc * 3, 0.0), pT(nc, 0.0);
  {
    auto T = host(solver.temperature());
    for (Index c = 0; c < nc; ++c) pT[c] = T(c);
  }
  const int maxSteps = 400000;
  int step = 0;
  Real resid = 1e300;
  for (; step < maxSteps; ++step) {
    solver.advance(ub, fb, src, momentum, pressure);
    auto u = host(solver.velocity());
    auto T = host(solver.temperature());
    Real du = 0.0, dT = 0.0;
    for (Index c = 0; c < nc; ++c) {
      for (int d = 0; d < 3; ++d) {
        du = std::max(du, std::abs(u(c,d) - pu[c*3 + d]));
        pu[c*3 + d] = u(c,d);
      }
      dT = std::max(dT, std::abs(T(c) - pT[c]));
      pT[c] = T(c);
    }
    resid = std::max(du / ref.vMax, dT) / dt;
    if (!std::isfinite(resid)) break;
    if (verbose && step % 500 == 0)
      std::printf("    N=%d Ra=%.0e step %6d  t %.4f  residual %.3e\n", N, ref.ra, step,
                  (step + 1) * dt, resid);
    if (resid < 1e-5) { ++step; break; }
  }

  // Hot-wall Nusselt number: the heat flux into the domain through x = 0,
  // over the wall's area, in units of kappa dT / L.
  auto q = host(solver.boundaryHeatFlux());
  Real heat = 0.0, area = 0.0;
  auto ba = host(mesh.boundaryArea());
  for (Index f = 0; f < nb; ++f) {
    if (!hh(f)) continue;
    heat += q(f);
    area += std::sqrt(ba(f,0)*ba(f,0) + ba(f,1)*ba(f,1) + ba(f,2)*ba(f,2));
  }
  // Mid-line profiles: the two columns (rows) straddling x = 0.5 (y = 0.5).
  auto u = host(solver.velocity());
  std::vector<Real> uLine(N, 0.0), vLine(N, 0.0);
  for (Index c = 0; c < nc; ++c) {
    const Index i = static_cast<Index>(std::floor(cc(c,0) * N));
    const Index j = static_cast<Index>(std::floor(cc(c,1) * N));
    if (i == N / 2 - 1 || i == N / 2) uLine[j] += 0.5 * u(c,0);
    if (j == N / 2 - 1 || j == N / 2) vLine[i] += 0.5 * u(c,1);
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return {heat / area, peak(uLine), peak(vLine), step, resid, secs};
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::vector<Real> ras;
    std::vector<Index> grids;
    bool meshes = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-n") { meshes = true; continue; }
      if (meshes) grids.push_back(std::stoi(a)); else ras.push_back(std::atof(a.c_str()));
    }
    if (grids.empty()) grids = {32, 64, 128};
    const bool verbose = std::getenv("VIBEFLOW_VERBOSE") != nullptr;
    std::printf("De Vahl Davis differentially heated cavity, Pr = %.2f\n", PR);
    bool ok = true;
    try {
      for (const Reference& ref : REF) {
        if (!ras.empty() &&
            std::find(ras.begin(), ras.end(), ref.ra) == ras.end()) continue;
        std::vector<Result> rs;
        for (Index N : grids) {
          rs.push_back(run(N, ref, verbose));
          const Result& r = rs.back();
          std::printf("  Ra %.0e  N=%-4d steps %7d  residual %.1e  Nu %.5f  u_max %.4f  "
                      "v_max %.4f  (%.0f s)\n", ref.ra, N, r.steps, r.resid, r.nusselt,
                      r.uMax, r.vMax, r.seconds);
          std::fflush(stdout);
          if (!(r.resid < 1e-5)) { std::printf("  FAIL: not steady\n"); ok = false; }
        }
        const Result& fine = rs.back();
        const bool nuFine = std::abs(fine.nusselt - ref.nu) / ref.nu <= 0.01;
        const bool uOk = std::abs(fine.uMax - ref.uMax) / ref.uMax <= 0.01;
        const bool vOk = std::abs(fine.vMax - ref.vMax) / ref.vMax <= 0.01;
        std::printf("  -> Ra %.0e: Nu on N=%d %.5f vs %.4f (%+.2f%%, within 1%%) %s; "
                    "u_max %+.2f%%, v_max %+.2f%% (within 1%%) %s\n",
                    ref.ra, grids.back(), fine.nusselt, ref.nu,
                    100.0 * (fine.nusselt - ref.nu) / ref.nu, nuFine ? "PASS" : "FAIL",
                    100.0 * (fine.uMax - ref.uMax) / ref.uMax,
                    100.0 * (fine.vMax - ref.vMax) / ref.vMax,
                    (uOk && vOk) ? "PASS" : "FAIL");
        ok &= nuFine && uOk && vOk;
        if (rs.size() == 3) {
          const Real r = static_cast<Real>(grids[1]) / grids[0];
          const Real d1 = rs[0].nusselt - rs[1].nusselt, d2 = rs[1].nusselt - rs[2].nusselt;
          bool extOk = false;
          if (d1 * d2 > 0.0) {
            const Real p = std::log(d1 / d2) / std::log(r);
            if (p >= 0.5 && p <= 4.0) {
              const Real ext = rs[2].nusselt - d2 / (std::pow(r, p) - 1.0);
              extOk = std::abs(ext - ref.nu) / ref.nu <= 0.005;
              std::printf("  -> Ra %.0e: observed order %.2f, Richardson Nu %.5f (%+.3f%%, "
                          "within 0.5%%) %s\n", ref.ra, p, ext,
                          100.0 * (ext - ref.nu) / ref.nu, extOk ? "PASS" : "FAIL");
            } else {
              std::printf("  -> Ra %.0e: observed order %.2f outside [0.5, 4]: FAIL\n",
                          ref.ra, p);
            }
          } else {
            std::printf("  -> Ra %.0e: Nusselt differences change sign, no order: FAIL\n",
                        ref.ra);
          }
          ok &= extOk;
        }
      }
    } catch (const std::exception& e) {
      std::printf("  FAIL: %s\n", e.what());
      ok = false;
    }
    std::printf("\nv2a heated-cavity GATE (C++): %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
  return rc;
}
