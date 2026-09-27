// v1 gate, C++ side of ADR-010 and ADR-037: the Rhie-Chow flux must not let
// the time step change a steady state.
//
// A steady problem has no temporal discretisation error -- the BDF2 terms
// cancel exactly once u stops changing -- so the same steady problem run with
// different time steps must reach the same discrete state. Whatever separates
// them is dt leaking into the spatial discretisation through the face flux,
// which is the trap ADR-010 described in the first week of v1 and asked a
// gate to cover.
//
// This is the port of prototype/ethier_steinman.py:gate_dt_independence, on
// the same meshes (the n = 8 vertex fixtures, which are the Python meshes
// written out) and the same two steady manufactured problems:
//
//   u = (sin px (cos py - cos pz), sin py (cos pz - cos px), sin pz (cos px - cos py))
//   p = 0, or p = cos px cos py cos pz          (p = pi times the coordinate)
//   s = (u . grad) u - nu laplacian(u) + grad(p),   laplacian(u) = -2 pi^2 u
//
// each run from rest until u and p stop changing, at dt = 0.02, 0.2 and 2.0.
// The spread is the largest L2 difference between two of those states,
// relative to the discretisation error of the dt = 2.0 run. Bound 1e-6: a
// formulation with no dt in its steady equations meets it at the iteration
// tolerance.
//
// Run:  steady_dt <fixtures>        VIBEFLOW_OLD_FLUX=v1 selects the v1 form,
//                                    which fails (ADR-037).

#include "mesh/HexMesh.hpp"
#include "discretization/FaceFlux.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {

constexpr Real PI = 3.14159265358979323846;

Vec3 velocity(const Vec3& q) {
  const Real sx = std::sin(PI*q.x), sy = std::sin(PI*q.y), sz = std::sin(PI*q.z);
  const Real cx = std::cos(PI*q.x), cy = std::cos(PI*q.y), cz = std::cos(PI*q.z);
  return {sx * (cy - cz), sy * (cz - cx), sz * (cx - cy)};
}

// g[i][j] = d u_i / d x_j. Component i is sin(pi x_i)(cos(pi x_j) - cos(pi x_k))
// with (i, j, k) cyclic.
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

Real pressureExact(const Vec3& q, bool withPressure) {
  if (!withPressure) return 0.0;
  return std::cos(PI*q.x) * std::cos(PI*q.y) * std::cos(PI*q.z);
}

Vec3 source(const Vec3& q, Real nu, bool withPressure) {
  const Vec3 u = velocity(q);
  const Real uu[3] = {u.x, u.y, u.z};
  Real g[3][3];
  velocityGradient(q, g);
  Real s[3];
  for (int i = 0; i < 3; ++i) {
    s[i] = 2.0 * PI * PI * nu * uu[i];
    for (int j = 0; j < 3; ++j) s[i] += uu[j] * g[i][j];
  }
  if (withPressure) {
    const Real sx = std::sin(PI*q.x), sy = std::sin(PI*q.y), sz = std::sin(PI*q.z);
    const Real cx = std::cos(PI*q.x), cy = std::cos(PI*q.y), cz = std::cos(PI*q.z);
    s[0] -= PI * sx * cy * cz;
    s[1] -= PI * cx * sy * cz;
    s[2] -= PI * cx * cy * sz;
  }
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

struct State {
  std::vector<Real> u, p;   // host copies: nc x 3 and nc
  int steps = 0;
  Real change = 0.0;
};

State steadyState(const HexMesh& mesh, Real nu, Real dt, bool withPressure,
                  bool v1, Real tol, Real maxTime) {
  const Index nc = mesh.nCells(), nt = mesh.nTotal();
  PisoControls ctl;
  ctl.outer = 3;              // as the Python gate: n_outer = 3, 2 correctors
  ctl.correctors = 2;
  // The gate judges the formulation, not the solver tolerances: tight enough
  // that tolerance-limited convergence stays far below the bound.
  ctl.nonOrthTol = 1e-11;
  ctl.pressureSolveTol = 1e-13;
  if (v1) ctl.oldFlux = OldFluxForm::V1;
  PisoSolver solver(mesh, nu, dt, ctl);

  VectorField src("src", nt, 3);
  {
    auto cc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellCentre());
    auto hs = Kokkos::create_mirror_view(src);
    for (Index i = 0; i < nt; ++i) {
      const Vec3 s = source({cc(i,0), cc(i,1), cc(i,2)}, nu, withPressure);
      hs(i,0) = s.x; hs(i,1) = s.y; hs(i,2) = s.z;
    }
    Kokkos::deep_copy(src, hs);
  }
  VectorField ub; ScalarField fb;
  averageBoundaryValue(mesh, velocity, ub);
  integrateBoundaryFlux(mesh, velocity, fb);
  adjustBoundaryFlux(mesh, fb);

  NativeBiCGStab momentum(mesh);
  NativeCG pressure(mesh, Comm(), true);

  State st;
  std::vector<Real> prevU(nc * 3, 0.0), prevP(nc, 0.0);
  bool havePrev = false;
  const int maxSteps = static_cast<int>(std::lround(maxTime / dt));
  st.change = 1e300;
  for (int k = 0; k < maxSteps; ++k) {
    solver.advance(ub, fb, src, momentum, pressure);
    st.steps = k + 1;
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
    if (havePrev) {
      st.change = change;
      if (change < tol) break;
    }
    havePrev = true;
  }
  st.u = prevU; st.p = prevP;
  return st;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const std::string dir = argc > 1 ? argv[1] : "tests/fixtures";
    const Real nu = 0.1, bound = 1e-6;
    const Real tol = 1e-12;       // max change per step that counts as steady
    const Real maxTime = 400.0;
    std::vector<Real> dts = {0.02, 0.2, 2.0};
    // Exploration only (ADR-037): VIBEFLOW_STEADY_DTS="0.002,0.02,0.2,2.0"
    // replaces the time steps. The gate itself is judged on the three above.
    if (const char* e = std::getenv("VIBEFLOW_STEADY_DTS")) {
      dts.clear();
      std::string list(e);
      std::size_t pos = 0;
      while (pos < list.size()) {
        const std::size_t comma = list.find(',', pos);
        dts.push_back(std::atof(list.substr(pos, comma - pos).c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
    }
    const char* of = std::getenv("VIBEFLOW_OLD_FLUX");
    const bool v1 = of && std::string(of) == "v1";

    std::string dtList;
    for (Real dt : dts) dtList += (dtList.empty() ? "" : ", ") + std::to_string(dt).substr(0, 6);
    std::printf("Rhie-Chow steady state independent of dt (steady MMS, n=8, nu=%.1f, "
                "dt = %s; old-flux form %s)\n", nu, dtList.c_str(), v1 ? "v1" : "exact");
    std::printf("  %-12s %-11s %16s %10s %10s %10s %10s\n",
                "problem", "mesh", "steps", "spread u", "spread p", "L2 u", "L2 p");
    bool ok = true;
    for (bool withPressure : {false, true}) {
      for (const char* tag : {"s0", "s25"}) {
        const HexMesh mesh = HexMesh::fromVertexFile(
            8, dir + "/vertices_n8_" + tag + ".txt");
        const Index nc = mesh.nCells();
        auto vol = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellVolume());
        auto cc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellCentre());
        Real volSum = 0.0;
        for (Index c = 0; c < nc; ++c) volSum += vol(c);

        std::vector<State> states;
        bool steady = true;
        std::string steps;
        for (Real dt : dts) {
          states.push_back(steadyState(mesh, nu, dt, withPressure, v1, tol, maxTime));
          steady &= states.back().change < tol;
          steps += (steps.empty() ? "" : "/") + std::to_string(states.back().steps);
        }
        auto centred = [&](const std::vector<Real>& p) {
          Real m = 0.0;
          for (Index c = 0; c < nc; ++c) m += p[c] * vol(c);
          m /= volSum;
          std::vector<Real> out(p);
          for (auto& v : out) v -= m;
          return out;
        };
        std::vector<Real> pex(nc);
        for (Index c = 0; c < nc; ++c) pex[c] = pressureExact({cc(c,0), cc(c,1), cc(c,2)}, withPressure);
        const auto pexC = centred(pex);
        const State& ref = states.back();
        const auto refP = centred(ref.p);
        Real eu = 0.0, ep = 0.0;
        for (Index c = 0; c < nc; ++c) {
          const Vec3 ue = velocity({cc(c,0), cc(c,1), cc(c,2)});
          const Real d0 = ref.u[c*3] - ue.x, d1 = ref.u[c*3+1] - ue.y, d2 = ref.u[c*3+2] - ue.z;
          eu += (d0*d0 + d1*d1 + d2*d2) * vol(c);
          ep += (refP[c] - pexC[c]) * (refP[c] - pexC[c]) * vol(c);
        }
        eu = std::sqrt(eu / volSum); ep = std::sqrt(ep / volSum);
        Real su = 0.0, sp = 0.0;
        for (std::size_t a = 0; a < states.size(); ++a) {
          const auto pa = centred(states[a].p);
          for (std::size_t b = a + 1; b < states.size(); ++b) {
            const auto pb = centred(states[b].p);
            Real du = 0.0, dp = 0.0;
            for (Index c = 0; c < nc; ++c) {
              for (int d = 0; d < 3; ++d) {
                const Real e = states[a].u[c*3+d] - states[b].u[c*3+d];
                du += e * e * vol(c);
              }
              dp += (pa[c] - pb[c]) * (pa[c] - pb[c]) * vol(c);
            }
            su = std::max(su, std::sqrt(du / volSum) / eu);
            sp = std::max(sp, std::sqrt(dp / volSum) / ep);
          }
        }
        const bool passed = steady && su <= bound && sp <= bound;
        ok &= passed;
        std::printf("  %-12s %-11s %16s %10.1e %10.1e %10.2e %10.2e  %s\n",
                    withPressure ? "grad p" : "constant p",
                    std::string(tag) == "s0" ? "orthogonal" : "skewed", steps.c_str(),
                    su, sp, eu, ep,
                    passed ? "PASS" : (steady ? "FAIL" : "FAIL (not steady)"));
      }
    }
    std::printf("  -> every spread <= %.0e: %s\n", bound, ok ? "PASS" : "FAIL");
    std::printf("\nv1 Rhie-Chow dt-independence GATE (C++): %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
  return rc;
}
