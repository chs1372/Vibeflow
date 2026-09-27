// v1 gate, C++ side: the Ethier-Steinman exact solution of the 3D unsteady
// incompressible Navier-Stokes equations.
//
// Unlike a manufactured solution this needs NO source term -- it satisfies
// the equations exactly. Verified symbolically in tests/mms/verify_exact.py.
//
// The distorted case runs on the SMOOTH mesh family (ADR-013). A randomly
// perturbed family redraws itself at every resolution, so its mesh quality
// never converges and the measured order tracks the mesh rather than the
// scheme.

#include "mesh/HexMesh.hpp"
#include "discretization/FaceFlux.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#ifdef VIBEFLOW_HAVE_PETSC
#include "linalg/PetscSolver.hpp"
#include <petscsys.h>
#endif
#include <memory>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {

constexpr Real PI = 3.14159265358979323846;
constexpr Real A = PI / 4.0;
constexpr Real D = PI / 2.0;

Vec3 velocity(const Vec3& p, Real t, Real nu) {
  const Real e = std::exp(-D * D * nu * t);
  return {-A * (std::exp(A*p.x) * std::sin(A*p.y + D*p.z)
                + std::exp(A*p.z) * std::cos(A*p.x + D*p.y)) * e,
          -A * (std::exp(A*p.y) * std::sin(A*p.z + D*p.x)
                + std::exp(A*p.x) * std::cos(A*p.y + D*p.z)) * e,
          -A * (std::exp(A*p.z) * std::sin(A*p.x + D*p.y)
                + std::exp(A*p.y) * std::cos(A*p.z + D*p.x)) * e};
}

// With Dirichlet velocity everywhere the pressure Poisson problem is solvable
// only if the boundary fluxes sum to zero. The exact field is divergence-free
// but the quadrature leaves a small imbalance; spreading it over the boundary
// area is what OpenFOAM's adjustPhi does.
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

struct Row {
  Index n; Real h, dt, l2, cont; int outer, nonOrth;
  Real momSec, preSec; int momIt, preIt, preSolves;
  std::string backend;
  // Work over the whole run, for costing a change from counts rather than
  // seconds (ADR-034, ADR-036): outer iterations summed over steps, and the
  // momentum solves those took.
  long long outerTotal{}, momSolves{};
};

// Backend selection, so the same case can be timed with either stack. The
// discretisation does not change, so the L2 errors must not either -- that is
// what makes this a safe optimisation rather than a rewrite.
std::string backendName() {
  const char* e = std::getenv("VIBEFLOW_PRESSURE");
  return e ? e : "native";
}

std::unique_ptr<LinearSolver> makePressureSolver(const Mesh& mesh) {
  const std::string cfg = backendName();
  // The pressure operator is pure Neumann, hence singular.
  if (cfg == "native") return std::make_unique<NativeCG>(mesh, Comm(), true);
#ifdef VIBEFLOW_HAVE_PETSC
  int interval = 1;
  if (const char* e = std::getenv("VIBEFLOW_PC_REUSE")) interval = std::stoi(e);
  return std::make_unique<PetscSolver>(mesh, Comm(), cfg,
                                       std::vector<Index>{}, interval);
#else
  std::fprintf(stderr, "built without PETSc; VIBEFLOW_PRESSURE=%s unavailable\n",
               cfg.c_str());
  std::exit(2);
#endif
}

Row run(Index n, Real dt, int nsteps, Real nu, Real skew, int outer = 6) {
  auto mesh = HexMesh::generate(n, skew, skew == 0.0 ? "none" : "smooth");
  const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();

  PisoControls ctl;
  ctl.outer = outer;
  // The non-orthogonal loop's cost is the solver's single biggest, and its
  // tolerance is the knob. An order study is where loosening it would show
  // first, so the knob is exposed here and the sweep is recorded in ADR-025.
  if (const char* e = std::getenv("VIBEFLOW_NONORTH_TOL")) ctl.nonOrthTol = std::atof(e);
  // The boundary-pressure extrapolation's sweep count and warm start, exposed
  // for the same reason: ADR-034 measures both here before the cylinder.
  if (const char* e = std::getenv("VIBEFLOW_PEXTRAP")) ctl.pressureExtrapSweeps = std::atoi(e);
  if (std::getenv("VIBEFLOW_PB_WARM")) ctl.pressureExtrapWarmStart = true;
  // The v1 old-flux form, for the record (ADR-037).
  if (const char* e = std::getenv("VIBEFLOW_OLD_FLUX"))
    if (std::string(e) == "v1") ctl.oldFlux = OldFluxForm::V1;
  // Momentum predictor sweeps per outer iteration (ADR-036).
  if (const char* e = std::getenv("VIBEFLOW_CONV_SWEEPS")) ctl.convectionSweeps = std::atoi(e);
  if (const char* e = std::getenv("VIBEFLOW_CONV_TOL")) ctl.convectionSweepTol = std::atof(e);
  PisoSolver solver(mesh, nu, dt, ctl);

  VectorField u0("u0", nt, 3);
  ScalarField p0("p0", nt), F0("F0", mesh.nInternalFaces());
  auto cc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellCentre());
  auto hu = Kokkos::create_mirror_view(u0);
  for (Index i = 0; i < nt; ++i) {
    const Vec3 v = velocity({cc(i,0), cc(i,1), cc(i,2)}, 0.0, nu);
    hu(i,0) = v.x; hu(i,1) = v.y; hu(i,2) = v.z;
  }
  Kokkos::deep_copy(u0, hu);
  {
    // Weighted by the face's own interpolation weight, not a plain 0.5
    // average: on a distorted mesh the two differ, and the initial flux sets
    // where the Rhie-Chow residual starts from.
    auto own = mesh.owner(); auto nei = mesh.neighbour();
    auto fa = mesh.faceArea(); auto fc = mesh.faceCentre(); auto cc = mesh.cellCentre();
    Kokkos::parallel_for("F0", Kokkos::RangePolicy<ExecSpace>(0, mesh.nInternalFaces()),
      KOKKOS_LAMBDA(const Index f) {
        Real lo = 0.0, ln = 0.0;
        for (int i = 0; i < 3; ++i) {
          const Real ro = fc(f, i) - cc(own(f), i);
          const Real rn = fc(f, i) - cc(nei(f), i);
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
  solver.setState(u0, p0, F0);

  VectorField src("src", nt, 3);
  NativeBiCGStab momentum(mesh);
  auto pressurePtr = makePressureSolver(mesh);
  LinearSolver& pressure = *pressurePtr;
  Real cont = 0.0;
  int outerUsed = 0, nonOrthUsed = 0;
  long long outerTotal = 0;
  for (int k = 0; k < nsteps; ++k) {
    const Real t = (k + 1) * dt;
    auto uFn = [t, nu](const Vec3& q) { return velocity(q, t, nu); };
    VectorField ub;
    ScalarField fb;
    averageBoundaryValue(mesh, uFn, ub);
    integrateBoundaryFlux(mesh, uFn, fb);
    adjustBoundaryFlux(mesh, fb);
    const auto rep = solver.advance(ub, fb, src, momentum, pressure);
    cont = std::max(cont, rep.continuityError);
    outerUsed = rep.outerUsed;
    outerTotal += rep.outerUsed;
    nonOrthUsed = rep.nonOrthSweeps;
  }

  const Real tEnd = nsteps * dt;
  auto u = solver.velocity();
  VectorField ex("ex", nt, 3);
  auto he = Kokkos::create_mirror_view(ex);
  for (Index i = 0; i < nt; ++i) {
    const Vec3 v = velocity({cc(i,0), cc(i,1), cc(i,2)}, tEnd, nu);
    he(i,0) = v.x; he(i,1) = v.y; he(i,2) = v.z;
  }
  Kokkos::deep_copy(ex, he);

  auto vol = mesh.cellVolume();
  Real num = 0.0, den = 0.0;
  Kokkos::parallel_reduce("l2", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index i, Real& a) {
      Real s = 0.0;
      for (int d = 0; d < 3; ++d) { const Real e = u(i,d) - ex(i,d); s += e * e; }
      a += s * vol(i);
    }, num);
  Kokkos::parallel_reduce("vol", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index i, Real& a) { a += vol(i); }, den);

  return {n, 1.0 / n, dt, std::sqrt(num / den), cont, outerUsed, nonOrthUsed,
          momentum.totalSeconds(), pressure.totalSeconds(),
          momentum.totalIterations(), pressure.totalIterations(),
          pressure.solveCount(), pressure.backendName(),
          outerTotal, static_cast<long long>(momentum.solveCount())};
}

bool report(const char* label, const std::vector<Row>& rows, Real lo, Real hi,
            bool requireRising) {
  std::printf("\n%s\n", label);
  std::printf("  %4s %10s %19s %11s %7s %9s %8s %9s %9s %8s\n",
              "N", "h", "L2(u) error", "max div", "outer", "nonOrth", "order",
              "mom s", "pres s", "pres it");
  std::vector<Real> orders;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    char o[16] = "       -";
    if (i) {
      orders.push_back(std::log(rows[i-1].l2 / rows[i].l2)
                       / std::log(rows[i-1].h / rows[i].h));
      std::snprintf(o, sizeof o, "%8.3f", orders.back());
    }
    std::printf("  %4d %10.5f %19.12e %11.2e %7d %9d %s %9.1f %9.1f %8d\n",
                rows[i].n, rows[i].h, rows[i].l2, rows[i].cont, rows[i].outer,
                rows[i].nonOrth, o, rows[i].momSec, rows[i].preSec, rows[i].preIt);
  }
  // Whole-run work, deterministic: "outer" above is the last step's count.
  for (const Row& r : rows)
    std::printf("  work  N %3d: %lld outer iterations, %lld momentum solves / %d iters, "
                "%d pressure solves / %d iters\n",
                r.n, r.outerTotal, r.momSolves, r.momIt, r.preSolves, r.preIt);
  const Real last = orders.empty() ? 0.0 : orders.back();
  bool ok = last >= lo && last <= hi;
  std::printf("  -> order %.3f in [%.2f, %.2f]", last, lo, hi);
  if (requireRising && orders.size() > 1) {
    bool rising = true;
    for (std::size_t i = 1; i < orders.size(); ++i)
      rising &= orders[i] >= orders[i-1] - 0.02;
    ok = ok && rising;
    std::printf(", trend %s", rising ? "rising" : "FALLING");
  }
  std::printf(": %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef VIBEFLOW_HAVE_PETSC
  PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::vector<Index> grids;
    for (int i = 1; i < argc; ++i) grids.push_back(std::stoi(argv[i]));
    if (grids.empty()) grids = {8, 16, 32};
    int outer = 6;
    if (const char* e = std::getenv("VIBEFLOW_OUTER")) outer = std::stoi(e);

    bool ok = true;
    std::printf("pressure backend: %s (pc rebuild interval %s)\n",
                backendName().c_str(),
                std::getenv("VIBEFLOW_PC_REUSE") ? std::getenv("VIBEFLOW_PC_REUSE") : "1");
    std::vector<Row> ortho, dist;
    for (Index n : grids) ortho.push_back(run(n, 2e-4, 2, 0.05, 0.0, outer));
    ok &= report("spatial order / orthogonal  (dt=2e-4, 2 steps)", ortho, 1.85, 2.15, false);
    for (Index n : grids) dist.push_back(run(n, 2e-4, 2, 0.05, 0.25, outer));
    ok &= report("spatial order / smooth distortion  (dt=2e-4, 2 steps)",
                 dist, 1.6, 2.3, true);

    std::printf("\nv1 Ethier-Steinman GATE (C++): %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
#ifdef VIBEFLOW_HAVE_PETSC
  PetscFinalize();
#endif
  return rc;
}
