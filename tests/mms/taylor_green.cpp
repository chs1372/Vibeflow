// ADR-039, C++ side: slip walls, gated by the Taylor-Green vortex.
//
//     u = ( sin(pi x) cos(pi y), -cos(pi x) sin(pi y), 0 ) exp(-2 pi^2 nu t)
//     p = ( cos(2 pi x) + cos(2 pi y) ) / 4 * exp(-4 pi^2 nu t)
//
// solves Navier-Stokes exactly and, in the unit cube, meets the slip condition
// on every wall: zero normal velocity, zero normal derivative of the
// tangential velocity. Slip on all six faces, zero boundary flux, the exact
// initial state, nu = 0.05, dt = 2e-4, two steps -- the port of
// prototype/taylor_green.py on 8/16/32 where Python runs 6/12/24. As revised
// before any C++ code (ADR-039):
//
//   1. on the smooth distortion -- walls planar and axis-aligned, the cells
//      next to them not -- the observed order in [1.6, 2.3] and approaching 2;
//   2. on both families, at every mesh, the slip error no more than 1.1 times
//      the error of the same run with the exact wall velocity prescribed.
//
// The orthogonal family's convergence is reported, not gated: on a uniform
// mesh this problem's interior error is tiny and a slower wall-layer error
// rules it whatever the boundary (ADR-039).
//
// Run:  taylor_green [grids, default 8 16 32]

#include "mesh/HexMesh.hpp"
#include "discretization/FaceFlux.hpp"
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

Vec3 velocity(const Vec3& q, Real t, Real nu) {
  const Real e = std::exp(-2.0 * PI * PI * nu * t);
  return {std::sin(PI*q.x) * std::cos(PI*q.y) * e,
          -std::cos(PI*q.x) * std::sin(PI*q.y) * e, 0.0};
}

Real pressure(const Vec3& q, Real t, Real nu) {
  return 0.25 * (std::cos(2*PI*q.x) + std::cos(2*PI*q.y)) * std::exp(-4.0 * PI * PI * nu * t);
}

template <class V> auto host(const V& v) {
  return Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
}

Real run(Index n, Real skew, bool slip, Real dt = 2e-4, int nsteps = 2,
         Real nu = 0.05, int outer = 6) {
  const HexMesh mesh = HexMesh::generate(n, skew, skew == 0.0 ? "none" : "smooth");
  const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();
  PisoControls ctl;
  ctl.outer = outer;
  ctl.correctors = 2;
  PisoSolver solver(mesh, nu, dt, ctl);

  auto cc = host(mesh.cellCentre());
  VectorField u0("u0", nt, 3);
  ScalarField p0("p0", nt);
  {
    auto hu = Kokkos::create_mirror_view(u0);
    auto hp = Kokkos::create_mirror_view(p0);
    Real mean = 0.0;
    for (Index c = 0; c < nt; ++c) {
      const Vec3 q{cc(c,0), cc(c,1), cc(c,2)};
      const Vec3 v = velocity(q, 0.0, nu);
      hu(c,0) = v.x; hu(c,1) = v.y; hu(c,2) = v.z;
      hp(c) = pressure(q, 0.0, nu);
      if (c < nc) mean += hp(c);
    }
    mean /= nc;                       // numpy's plain mean, as the Python gate
    for (Index c = 0; c < nt; ++c) hp(c) -= mean;
    Kokkos::deep_copy(u0, hu);
    Kokkos::deep_copy(p0, hp);
  }
  ScalarField F0("F0", mesh.nInternalFaces());
  {
    auto own = mesh.owner(); auto nei = mesh.neighbour();
    auto fa = mesh.faceArea(); auto fc = mesh.faceCentre(); auto cd = mesh.cellCentre();
    Kokkos::parallel_for("F0", Kokkos::RangePolicy<ExecSpace>(0, mesh.nInternalFaces()),
      KOKKOS_LAMBDA(const Index f) {
        Real lo = 0.0, ln = 0.0;
        for (int i = 0; i < 3; ++i) {
          const Real ro = fc(f, i) - cd(own(f), i);
          const Real rn = fc(f, i) - cd(nei(f), i);
          lo += ro * ro; ln += rn * rn;
        }
        const Real w = Kokkos::sqrt(ln) / (Kokkos::sqrt(lo) + Kokkos::sqrt(ln));
        Real s = 0.0;
        for (int i = 0; i < 3; ++i)
          s += (w * u0(own(f), i) + (1.0 - w) * u0(nei(f), i)) * fa(f, i);
        F0(f) = s;
      });
    Kokkos::fence();
  }
  if (slip) {
    View1<int> bt("bt", nb);
    Kokkos::deep_copy(bt, static_cast<int>(VelocityBC::Slip));
    solver.setBoundaryTypes(bt);
  }
  solver.setState(u0, p0, F0);

  VectorField src("src", nt, 3), ubZero("ub", nb, 3);
  ScalarField fb("fb", nb);                  // nothing crosses a slip wall
  NativeBiCGStab momentum(mesh);
  NativeCG pressureSolver(mesh, Comm(), true);
  for (int k = 0; k < nsteps; ++k) {
    if (slip) {
      solver.advance(ubZero, fb, src, momentum, pressureSolver);
    } else {                                 // the control: exact wall velocity
      const Real t = (k + 1) * dt;
      VectorField ub;
      averageBoundaryValue(mesh, [t, nu](const Vec3& q) { return velocity(q, t, nu); }, ub);
      solver.advance(ub, fb, src, momentum, pressureSolver);
    }
  }
  auto u = host(solver.velocity());
  auto vol = host(mesh.cellVolume());
  Real num = 0.0, den = 0.0;
  for (Index c = 0; c < nc; ++c) {
    const Vec3 e = velocity({cc(c,0), cc(c,1), cc(c,2)}, nsteps * dt, nu);
    const Real d0 = u(c,0) - e.x, d1 = u(c,1) - e.y, d2 = u(c,2) - e.z;
    num += (d0*d0 + d1*d1 + d2*d2) * vol(c);
    den += vol(c);
  }
  return std::sqrt(num / den);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::vector<Index> grids;
    for (int i = 1; i < argc; ++i) grids.push_back(std::stoi(argv[i]));
    if (grids.empty()) grids = {8, 16, 32};
    bool ok = true;
    std::printf("Taylor-Green vortex, slip walls on all six faces (nu = 0.05, dt = 2e-4, 2 steps)\n");
    struct Fam { Real skew; const char* tag; bool gated; };
    for (const Fam& fm : {Fam{0.0, "orthogonal", false}, Fam{0.25, "smooth distortion", true}}) {
      std::vector<Real> slip, exact;
      try {
        for (Index n : grids) slip.push_back(run(n, fm.skew, true));
      } catch (const std::exception& e) {
        std::printf("  FAIL: %s\n", e.what());
        ok = false;
        continue;
      }
      for (Index n : grids) exact.push_back(run(n, fm.skew, false));
      for (std::size_t i = 0; i < grids.size(); ++i) {
        const Real ratio = slip[i] / exact[i];
        const bool good = ratio <= 1.1;
        ok &= good;
        std::printf("  %-18s n=%-3d L2(u) slip %.10e   exact walls %.10e   ratio %.3f (<= 1.1) %s\n",
                    fm.tag, grids[i], slip[i], exact[i], ratio, good ? "ok" : "TOO LARGE");
      }
      if (grids.size() < 3) continue;
      std::vector<Real> so, eo;
      for (std::size_t i = 1; i < grids.size(); ++i) {
        const Real r = std::log(static_cast<Real>(grids[i]) / grids[i-1]);
        so.push_back(std::log(slip[i-1] / slip[i]) / r);
        eo.push_back(std::log(exact[i-1] / exact[i]) / r);
      }
      std::printf("  -> %s: order slip %.3f then %.3f, exact walls %.3f then %.3f",
                  fm.tag, so[0], so[1], eo[0], eo[1]);
      if (fm.gated) {
        const bool good = so[1] >= 1.6 && so[1] <= 2.3 &&
                          std::abs(so[1] - 2.0) <= std::abs(so[0] - 2.0) + 0.02;
        ok &= good;
        std::printf("; slip in [1.6, 2.3] and approaching 2: %s\n", good ? "PASS" : "FAIL");
      } else {
        std::printf("  (reported, not gated)\n");
      }
    }
    std::printf("\nv2a slip-wall GATE (C++): %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
  return rc;
}
