// Lid-driven cavity against Ghia, Ghia & Shin (1982).
//
// This is the first gate in the suite that checks the solver against MEASURED
// reference data rather than against a solution we constructed ourselves. An
// exact solution proves the discretisation converges to the equations we
// wrote down; a benchmark proves those are the equations everyone else solves.
//
// Ghia's case is 2D. It runs here as a one-cell-thick slab with zero-gradient
// velocity on the two z faces, which reproduces the 2D problem exactly because
// the flow is z-invariant.
//
// Reference values are Table I (u on the vertical centreline) and Table II
// (v on the horizontal centreline) of that paper, transcribed from
// https://gist.github.com/ivan-pi/3e9326d18a366ffe6a8e5bfda6353219 and
// .../caa6c6737d36a9140fbcf2ea59c78b3c

#include "mesh/HexMesh.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#ifdef VIBEFLOW_HAVE_PETSC
#include "linalg/PetscSolver.hpp"
#include <petscsys.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {

// Table I: u along x = 0.5, bottom to top.
const Real GHIA_Y[] = {0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                       0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                       0.9688, 0.9766, 1.0000};
const Real GHIA_U100[] = {0.00000, -0.03717, -0.04192, -0.04775, -0.06434,
                          -0.10150, -0.15662, -0.21090, -0.20581, -0.13641,
                          0.00332, 0.23151, 0.68717, 0.73722, 0.78871,
                          0.84123, 1.00000};
const Real GHIA_U1000[] = {0.00000, -0.18109, -0.20196, -0.22220, -0.29730,
                           -0.38289, -0.27805, -0.10648, -0.06080, 0.05702,
                           0.18719, 0.33304, 0.46604, 0.51117, 0.57492,
                           0.65928, 1.00000};

// Table II: v along y = 0.5, left to right.
const Real GHIA_X[] = {0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                       0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                       0.9609, 0.9688, 1.0000};
const Real GHIA_V100[] = {0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                          0.17507, 0.17527, 0.05454, -0.24533, -0.22445,
                          -0.16914, -0.10313, -0.08864, -0.07391, -0.05906,
                          0.00000};
const Real GHIA_V1000[] = {0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095,
                           0.33075, 0.32235, 0.02526, -0.31966, -0.42665,
                           -0.51500, -0.39188, -0.33714, -0.27669, -0.21388,
                           0.00000};
constexpr int NREF = 17;

std::unique_ptr<LinearSolver> makePressureSolver(const Mesh& mesh) {
  const char* e = std::getenv("VIBEFLOW_PRESSURE");
  const std::string cfg = e ? e : "native";
  if (cfg == "native") return std::make_unique<NativeCG>(mesh, Comm(), true);
#ifdef VIBEFLOW_HAVE_PETSC
  return std::make_unique<PetscSolver>(mesh, Comm(), cfg);
#else
  std::fprintf(stderr, "built without PETSc\n"); std::exit(2);
#endif
}

// Linear interpolation of a centreline profile onto the reference points.
Real sampleLine(const std::vector<Real>& pos, const std::vector<Real>& val, Real at) {
  if (at <= pos.front()) return val.front();
  if (at >= pos.back()) return val.back();
  const auto it = std::lower_bound(pos.begin(), pos.end(), at);
  const std::size_t i = static_cast<std::size_t>(it - pos.begin());
  const Real t = (at - pos[i - 1]) / (pos[i] - pos[i - 1]);
  return val[i - 1] + t * (val[i] - val[i - 1]);
}

struct Result { Real rmsU, rmsV, maxU, maxV; int steps; Real resid; };

Result run(Index N, Real Re, Real dt, int maxSteps, Real tol, bool verbose) {
  const Real nu = 1.0 / Re;
  auto mesh = HexMesh::box(N, N, 1, 1.0, 1.0, 1.0 / N);
  const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();

  PisoControls ctl;
  ctl.outer = 3;
  ctl.outerTol = 1e-8;
  PisoSolver solver(mesh, nu, dt, ctl);

  // Sides: 0 = x-, 1 = x+, 2 = y-, 3 = y+ (the lid), 4 = z-, 5 = z+.
  auto side = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                  mesh.boundarySide());
  View1<int> bcType("bcType", nb);
  VectorField ub("ub", nb, 3);
  ScalarField fb("fb", nb);
  auto hbt = Kokkos::create_mirror_view(bcType);
  auto hub = Kokkos::create_mirror_view(ub);
  for (Index f = 0; f < nb; ++f) {
    const bool slip = side(f) >= 4;
    hbt(f) = static_cast<int>(slip ? VelocityBC::ZeroGradient : VelocityBC::Dirichlet);
    hub(f, 0) = (side(f) == 3) ? 1.0 : 0.0;   // lid moves in +x
    hub(f, 1) = 0.0;
    hub(f, 2) = 0.0;
  }
  Kokkos::deep_copy(bcType, hbt);
  Kokkos::deep_copy(ub, hub);
  Kokkos::deep_copy(fb, 0.0);          // no through-flow anywhere
  solver.setBoundaryTypes(bcType);

  VectorField src("src", nt, 3), uPrev("uPrev", nt, 3);
  NativeBiCGStab momentum(mesh);
  auto pressurePtr = makePressureSolver(mesh);

  int step = 0;
  Real resid = 1.0;
  for (; step < maxSteps; ++step) {
    Kokkos::deep_copy(uPrev, solver.velocity());
    solver.advance(ub, fb, src, momentum, *pressurePtr);
    auto u = solver.velocity();
    Real d = 0.0;
    Kokkos::parallel_reduce("resid", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& a) {
        for (int k = 0; k < 3; ++k) a = Kokkos::max(a, Kokkos::abs(u(c, k) - uPrev(c, k)));
      }, Kokkos::Max<Real>(d));
    resid = d / dt;               // steady when the time derivative vanishes
    if (verbose && step % 50 == 0)
      std::printf("    step %5d  du/dt %.3e\n", step, resid);
    if (resid < tol) { ++step; break; }
  }

  // Centreline profiles. The cells nearest x = 0.5 and y = 0.5 are used; with
  // N even the centre falls on a face, so two columns straddle it and are
  // averaged.
  auto cc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellCentre());
  auto hu = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), solver.velocity());

  // The profile must span the walls. Ghia's tables include y = 0 and y = 1,
  // where u is exactly 0 and 1; a profile made of cell centres stops half a
  // cell short, and clamping to the last cell value reports the lid as moving
  // at 0.77 instead of 1. That is a comparison artefact, not a solver error,
  // and it swamped the real differences.
  std::vector<Real> yPos{0.0}, uVal{0.0}, xPos{0.0}, vVal{0.0};
  {
    // cell id = (i*ny + j)*nz + k with nz = 1
    const Index half = N / 2;
    for (Index j = 0; j < N; ++j) {
      const Index a = (half - 1) * N + j, b = half * N + j;
      yPos.push_back(0.5 * (cc(a, 1) + cc(b, 1)));
      uVal.push_back(0.5 * (hu(a, 0) + hu(b, 0)));
    }
    yPos.push_back(1.0); uVal.push_back(1.0);      // the lid
    for (Index i = 0; i < N; ++i) {
      const Index a = i * N + (half - 1), b = i * N + half;
      xPos.push_back(0.5 * (cc(a, 0) + cc(b, 0)));
      vVal.push_back(0.5 * (hu(a, 1) + hu(b, 1)));
    }
    xPos.push_back(1.0); vVal.push_back(0.0);      // the right wall
  }

  const Real* refU = (Re < 500.0) ? GHIA_U100 : GHIA_U1000;
  const Real* refV = (Re < 500.0) ? GHIA_V100 : GHIA_V1000;
  Real su = 0.0, sv = 0.0, mu = 0.0, mv = 0.0;
  std::printf("    %8s %10s %10s %8s | %8s %10s %10s %8s\n",
              "y", "u (ours)", "u (Ghia)", "diff", "x", "v (ours)", "v (Ghia)", "diff");
  for (int i = 0; i < NREF; ++i) {
    const Real u = sampleLine(yPos, uVal, GHIA_Y[i]);
    const Real v = sampleLine(xPos, vVal, GHIA_X[i]);
    const Real du = u - refU[i], dv = v - refV[i];
    su += du * du; sv += dv * dv;
    mu = std::max(mu, std::abs(du)); mv = std::max(mv, std::abs(dv));
    std::printf("    %8.4f %10.5f %10.5f %8.4f | %8.4f %10.5f %10.5f %8.4f\n",
                GHIA_Y[i], u, refU[i], du, GHIA_X[i], v, refV[i], dv);
  }
  return {std::sqrt(su / NREF), std::sqrt(sv / NREF), mu, mv, step, resid};
}

}  // namespace

int main(int argc, char** argv) {
#ifdef VIBEFLOW_HAVE_PETSC
  PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    Index N = 64;
    if (argc > 1 && argv[1][0] != '-') N = std::stoi(argv[1]);
    const bool verbose = std::getenv("VIBEFLOW_VERBOSE") != nullptr;

    struct Case { Real Re, dt, tolRms, tolSteady; int maxSteps; };
    // Ghia used a 129x129 uniform grid. At a coarser one the thin near-wall
    // layers are under-resolved, so the tolerance is set by what this grid
    // can resolve, not by what the solver can do. MEASURED at 64x64:
    //   Re = 100   rms(u) 0.0016  rms(v) 0.0045  max 0.0088, steady in 1123 steps
    //   Re = 1000  rms(u) 0.0114  rms(v) 0.0117  max 0.0212, 8000 steps
    // Re = 1000 is 88% of the runtime, so it is opt-in: pass --full.
    const bool full = [&]{ for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--full") return true; return false; }();
    std::vector<Case> cases{{100.0, 0.02, 0.012, 1e-5, 4000}};
    if (full) cases.push_back({1000.0, 0.01, 0.030, 2e-4, 12000});

    bool ok = true;
    for (const auto& c : cases) {
      std::printf("\nlid-driven cavity, Re = %.0f, %dx%d cells\n", c.Re,
                  static_cast<int>(N), static_cast<int>(N));
      const Result r = run(N, c.Re, c.dt, c.maxSteps, c.tolSteady, verbose);
      // Both conditions matter. A profile read off a state that is still
      // evolving is not a steady-state result, however well it happens to
      // match, so the residual is gated too rather than just reported.
      const bool steady = r.resid < c.tolSteady;
      const bool pass = r.rmsU < c.tolRms && r.rmsV < c.tolRms && steady;
      std::printf("    %d steps, du/dt %.1e (steady below %.0e: %s); "
                  "rms(u) %.4f  rms(v) %.4f  max|du| %.4f  max|dv| %.4f\n",
                  r.steps, r.resid, c.tolSteady, steady ? "yes" : "NO",
                  r.rmsU, r.rmsV, r.maxU, r.maxV);
      std::printf("    -> rms against Ghia below %.3f and steady: %s\n", c.tolRms,
                  pass ? "PASS" : "FAIL");
      ok &= pass;
    }
    std::printf("\ncavity benchmark GATE: %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
#ifdef VIBEFLOW_HAVE_PETSC
  PetscFinalize();
#endif
  return rc;
}
