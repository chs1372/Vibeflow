// Vortex shedding behind a circular cylinder at Re = 100.
//
// The first UNSTEADY benchmark, and the first case on a body-fitted
// unstructured mesh from an external generator. What it adds over the cavity:
//
//  * it tests time accuracy against an external number. The cavity is steady,
//    so it says nothing about BDF2 beyond "it reaches the right fixed point".
//  * it needs an outflow, so it exercises the pressure Dirichlet path and the
//    boundary flux being solved for rather than prescribed.
//  * the mesh comes from gmsh, with 41 degrees of non-orthogonality and cells
//    spanning three orders of magnitude in volume.
//
// Reference: Williamson's correlation for the laminar shedding regime
// (50 < Re < 180),  St = -3.3265/Re + 0.1816 + 1.6e-4 Re,  giving
// St = 0.164 at Re = 100. Reported drag lies between about 1.32 and 1.36 and
// the lift amplitude between about 0.30 and 0.35 across published studies;
// both are sensitive to domain size and blockage, so the gate uses bands, not
// point values.

#include "mesh/PolyMesh.hpp"
#include "physics/Piso.hpp"
#include "io/VtuWriter.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#ifdef NSFLOW_HAVE_PETSC
#include "linalg/PetscSolver.hpp"
#include <petscsys.h>
#endif
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace nsflow;

namespace {

constexpr Real D = 1.0, R = 0.5, U_IN = 1.0;
// Domain extents are READ FROM THE MESH, not declared here. Hard-coding them
// and then running a different mesh classified a far-field face as
// unclassified and aborted; deriving them means the case cannot disagree with
// its own grid.
Real X_IN = 0.0, X_OUT = 0.0, Y_HALF = 0.0;

Real williamsonSt(Real Re) { return -3.3265 / Re + 0.1816 + 1.6e-4 * Re; }

// Largest convective Courant number measured stable for this case, from the
// sweep in tests/benchmark/courant_limit. Not a theoretical bound and not a
// guess: an earlier version of this file warned above 2, which was an estimate
// from cell size and free-stream speed, and the case then ran happily at 7.2.
constexpr Real COURANT_WARN = 8.0;

enum Patch { INLET, OUTLET, FARFIELD, CYLINDER, SPANWISE };

std::unique_ptr<LinearSolver> makePressureSolver(const Mesh& mesh) {
  const char* e = std::getenv("NSFLOW_PRESSURE");
  const std::string cfg = e ? e : "cg+hypre";
  if (cfg == "native") return std::make_unique<NativeCG>(mesh, Comm(), false);
#ifdef NSFLOW_HAVE_PETSC
  return std::make_unique<PetscSolver>(mesh, Comm(), cfg);
#else
  return std::make_unique<NativeCG>(mesh, Comm(), false);
#endif
}

// Strouhal number from the lift signal, by averaging the interval between
// upward zero crossings over whole cycles. Counting peaks instead would be
// noisier: a peak is one sample, a crossing is interpolated between two.
Real strouhalFromLift(const std::vector<Real>& t, const std::vector<Real>& cl,
                      std::size_t skip, int& cycles) {
  std::vector<Real> crossings;
  for (std::size_t i = skip + 1; i < cl.size(); ++i) {
    if (cl[i - 1] <= 0.0 && cl[i] > 0.0) {
      const Real frac = -cl[i - 1] / (cl[i] - cl[i - 1]);
      crossings.push_back(t[i - 1] + frac * (t[i] - t[i - 1]));
    }
  }
  cycles = static_cast<int>(crossings.size()) - 1;
  if (cycles < 2) return 0.0;
  const Real period = (crossings.back() - crossings.front()) / cycles;
  return D / (U_IN * period);
}

}  // namespace

int main(int argc, char** argv) {
#ifdef NSFLOW_HAVE_PETSC
  PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const std::string meshPath = argc > 1 ? argv[1] : "cases/cylinder/cylinder.hex";
    const Real Re = 100.0, nu = U_IN * D / Re;
    const Real dt = (argc > 2) ? std::atof(argv[2]) : 0.05;
    const Real tEnd = (argc > 3) ? std::atof(argv[3]) : 200.0;
    const Real tStats = 0.6 * tEnd;      // discard the start-up transient

    auto mesh = PolyMesh::fromHexFile(meshPath);
    const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();
    std::printf("cylinder Re=%.0f  cells %d  non-orth %.1f deg  skew %.2f\n",
                Re, static_cast<int>(nc), mesh.maxNonOrthogonality(),
                mesh.maxSkewness());

    // Classify boundary faces geometrically. An unstructured mesh gives no
    // patch names through this path, and position is more trustworthy than a
    // mesher's naming anyway.
    auto bcen = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                    mesh.boundaryCentre());
    {
      Real xmin = 1e30, xmax = -1e30, ymax = -1e30;
      for (Index f = 0; f < nb; ++f) {
        xmin = std::min(xmin, bcen(f, 0));
        xmax = std::max(xmax, bcen(f, 0));
        ymax = std::max(ymax, std::abs(bcen(f, 1)));
      }
      X_IN = xmin; X_OUT = xmax; Y_HALF = ymax;
      std::printf("  domain x [%.2f, %.2f]  |y| <= %.2f\n", X_IN, X_OUT, Y_HALF);
    }
    auto barea = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                     mesh.boundaryArea());
    View1<int> uType("uType", nb), pType("pType", nb), cylMask("cylMask", nb);
    VectorField ub("ub", nb, 3);
    ScalarField fb("fb", nb), pval("pval", nb);
    auto hUT = Kokkos::create_mirror_view(uType);
    auto hPT = Kokkos::create_mirror_view(pType);
    auto hCM = Kokkos::create_mirror_view(cylMask);
    auto hUB = Kokkos::create_mirror_view(ub);
    auto hFB = Kokkos::create_mirror_view(fb);
    auto hPV = Kokkos::create_mirror_view(pval);

    int nIn = 0, nOut = 0, nFar = 0, nCyl = 0, nSpan = 0;
    for (Index f = 0; f < nb; ++f) {
      const Real x = bcen(f, 0), y = bcen(f, 1);
      const Real r = std::hypot(x, y);
      const bool spanwise = std::abs(barea(f, 2)) > 0.9 *
          std::sqrt(barea(f,0)*barea(f,0) + barea(f,1)*barea(f,1) + barea(f,2)*barea(f,2));
      Patch p;
      if (spanwise) p = SPANWISE;
      else if (std::abs(x - X_IN) < 1e-9 * (X_OUT - X_IN) + 1e-9) p = INLET;
      else if (std::abs(x - X_OUT) < 1e-9 * (X_OUT - X_IN) + 1e-9) p = OUTLET;
      else if (std::abs(std::abs(y) - Y_HALF) < 1e-9 * Y_HALF + 1e-9) p = FARFIELD;
      else if (r < 1.5 * R) p = CYLINDER;
      else { std::printf("unclassified face at (%.3f, %.3f)\n", x, y); std::exit(2); }

      hUB(f, 0) = hUB(f, 1) = hUB(f, 2) = 0.0;
      hFB(f) = 0.0;
      hPV(f) = 0.0;
      hCM(f) = 0;
      switch (p) {
        case INLET:
          hUT(f) = static_cast<int>(VelocityBC::Dirichlet);
          hPT(f) = static_cast<int>(PressureBC::FixedFlux);
          hUB(f, 0) = U_IN;
          hFB(f) = U_IN * barea(f, 0);
          ++nIn; break;
        case OUTLET:
          hUT(f) = static_cast<int>(VelocityBC::ZeroGradient);
          hPT(f) = static_cast<int>(PressureBC::FixedValue);
          ++nOut; break;
        case FARFIELD:                     // slip: free stream, no through-flow
          hUT(f) = static_cast<int>(VelocityBC::Dirichlet);
          hPT(f) = static_cast<int>(PressureBC::FixedFlux);
          hUB(f, 0) = U_IN;
          ++nFar; break;
        case CYLINDER:
          hUT(f) = static_cast<int>(VelocityBC::Dirichlet);
          hPT(f) = static_cast<int>(PressureBC::FixedFlux);
          hCM(f) = 1;
          ++nCyl; break;
        case SPANWISE:
          hUT(f) = static_cast<int>(VelocityBC::ZeroGradient);
          hPT(f) = static_cast<int>(PressureBC::FixedFlux);
          ++nSpan; break;
      }
    }
    Kokkos::deep_copy(uType, hUT); Kokkos::deep_copy(pType, hPT);
    Kokkos::deep_copy(cylMask, hCM); Kokkos::deep_copy(ub, hUB);
    Kokkos::deep_copy(fb, hFB); Kokkos::deep_copy(pval, hPV);
    std::printf("  patches: inlet %d  outlet %d  farfield %d  cylinder %d  spanwise %d\n",
                nIn, nOut, nFar, nCyl, nSpan);

    Real span = 0.0;
    {
      auto bar = mesh.boundaryArea();
      auto cm = cylMask;
      Kokkos::parallel_reduce("cylArea", Kokkos::RangePolicy<ExecSpace>(0, nb),
        KOKKOS_LAMBDA(const Index f, Real& a) {
          if (cm(f)) a += Kokkos::sqrt(bar(f,0)*bar(f,0) + bar(f,1)*bar(f,1));
        }, span);
      span /= (M_PI * D);      // wetted area / circumference = spanwise thickness
    }
    std::printf("  span %.4f (forces normalised by it)\n", span);

    PisoControls ctl;
    ctl.outer = std::atoi(std::getenv("CYL_OUTER") ? std::getenv("CYL_OUTER") : "3");
    ctl.outerTol = 1e-7;
    if (const char* e = std::getenv("CYL_NONORTH")) ctl.nonOrthCorrectors = std::atoi(e);
    if (const char* e = std::getenv("CYL_NONORTH_TOL")) ctl.nonOrthTol = std::atof(e);
    if (std::getenv("CYL_UPWIND")) ctl.deferredCorrection = false;
    if (std::getenv("CYL_NO_DIFF_NONORTH")) ctl.diffusionNonOrth = false;
    if (std::getenv("CYL_NAIVE_RC")) ctl.consistentRhieChow = false;
    if (std::getenv("CYL_ZG_PRESSURE")) ctl.pressureExtrapolation = false;
    if (const char* e = std::getenv("CYL_PEXTRAP")) ctl.pressureExtrapSweeps = std::atoi(e);
    std::printf("  controls: outer %d  nonOrth <= %d sweeps to %.0e%s\n",
                ctl.outer, ctl.nonOrthCorrectors, ctl.nonOrthTol,
                ctl.deferredCorrection ? "" : "  [FIRST-ORDER UPWIND]");
    if (!ctl.diffusionNonOrth)
      std::printf("  controls: diffusion non-orthogonal correction OFF\n");
    PisoSolver solver(mesh, nu, dt, ctl);
    solver.setBoundaryTypes(uType);
    solver.setPressureBoundary(pType, pval);

    // Start from the free stream with an asymmetric kick, so shedding begins
    // from a physical instability rather than from round-off. The size of the
    // kick only sets how long the transient lasts: at Re = 100 the shedding
    // limit cycle is a global attractor, so the measured Strouhal number,
    // drag and lift amplitude do not depend on it -- and the statistics
    // window starts at 0.6 t_end, well after saturation, with a minimum cycle
    // count enforced besides.
    VectorField u0("u0", nt, 3);
    auto cc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                  mesh.cellCentre());
    auto hu = Kokkos::create_mirror_view(u0);
    for (Index c = 0; c < nt; ++c) {
      const Real x = cc(c, 0), y = cc(c, 1);
      const Real r = std::hypot(x, y);
      hu(c, 0) = (r > R * 1.001) ? U_IN : 0.0;
      hu(c, 1) = (r > R && r < 3.0 * R && x > 0.0) ? 0.20 * U_IN : 0.0;
      hu(c, 2) = 0.0;
    }
    Kokkos::deep_copy(u0, hu);
    ScalarField p0("p0", nt), F0("F0", mesh.nInternalFaces());
    {
      auto own = mesh.owner(); auto nei = mesh.neighbour(); auto fa = mesh.faceArea();
      Kokkos::parallel_for("F0", Kokkos::RangePolicy<ExecSpace>(0, mesh.nInternalFaces()),
        KOKKOS_LAMBDA(const Index f) {
          Real s = 0.0;
          for (int i = 0; i < 3; ++i) s += 0.5 * (u0(own(f), i) + u0(nei(f), i)) * fa(f, i);
          F0(f) = s;
        });
      Kokkos::fence();
    }
    solver.setState(u0, p0, F0);

    VectorField src("src", nt, 3);
    NativeBiCGStab momentum(mesh);
    auto pressure = makePressureSolver(mesh);
    std::printf("  pressure backend: %s\n", pressure->backendName().c_str());

    std::vector<Real> th, cdh, clh;
    const auto tWall0 = std::chrono::steady_clock::now();
    const Real qA = 0.5 * U_IN * U_IN * D * span;     // dynamic pressure x area
    const int nSteps = static_cast<int>(tEnd / dt);
    const int reportLines =
        std::atoi(std::getenv("CYL_REPORT") ? std::getenv("CYL_REPORT") : "25");
    Real maxCont = 0.0, maxCo = 0.0;
    bool coWarned = false;
    for (int k = 0; k < nSteps; ++k) {
      const auto rep = solver.advance(ub, fb, src, momentum, *pressure);
      const Real t = (k + 1) * dt;
      const Vec3 F = solver.boundaryForce(cylMask);
      th.push_back(t);
      cdh.push_back(F.x / qA);
      clh.push_back(F.y / qA);
      if (t > tStats) maxCont = std::max(maxCont, rep.continuityError);
      maxCo = std::max(maxCo, rep.courant);
      if (rep.courant > COURANT_WARN && !coWarned) {
        coWarned = true;
        std::printf("    NOTE: Courant %.1f, above the largest value measured "
                    "stable for this case (%.0f). The convection scheme's "
                    "deferred correction is explicit, so there is a limit; "
                    "where it sits is a measurement, not a rule of thumb.\n",
                    rep.courant, COURANT_WARN);
      }
      if (k % std::max(1, nSteps / reportLines) == 0)
        std::printf("    t %7.2f  Cd %8.4f  Cl %8.4f  div %.1e  Co %5.2f  "
                    "|u|max %6.2f at (%6.2f,%6.2f) r=%5.2f  nonOrth %d\n",
                    t, cdh.back(), clh.back(), rep.continuityError, rep.courant,
                    rep.uMax, rep.uMaxAt[0], rep.uMaxAt[1],
                    std::hypot(rep.uMaxAt[0], rep.uMaxAt[1]), rep.nonOrthSweeps);
    }

    {
      const Real wall = std::chrono::duration<Real>(
          std::chrono::steady_clock::now() - tWall0).count();
      std::printf("\n  cost: %.0f s wall for %d steps (%.2f s/step)\n",
                  wall, nSteps, wall / std::max(1, nSteps));
      std::printf("    momentum  %7.0f s  %8d solves  %10d iters\n",
                  momentum.totalSeconds(), momentum.solveCount(),
                  momentum.totalIterations());
      std::printf("    pressure  %7.0f s  %8d solves  %10d iters\n",
                  pressure->totalSeconds(), pressure->solveCount(),
                  pressure->totalIterations());
      const auto& t = solver.timings();
      std::printf("    phases (s): assemble %.0f  gradient %.0f (%d calls)  "
                  "boundaryP %.0f (%d calls)\n"
                  "                HbyA %.0f  RhieChow %.0f  pressureStage %.0f"
                  "  advance total %.0f\n",
                  t.assemble, t.gradient, t.gradCalls, t.boundaryP,
                  t.boundaryPCalls, t.hbya, t.rhieChow, t.pressureAssembly,
                  t.total);
    }

    std::size_t skip = 0;
    while (skip < th.size() && th[skip] < tStats) ++skip;
    int cycles = 0;
    const Real St = strouhalFromLift(th, clh, skip, cycles);
    Real cdMean = 0.0, clMax = -1e30, clMin = 1e30;
    for (std::size_t i = skip; i < th.size(); ++i) {
      cdMean += cdh[i];
      clMax = std::max(clMax, clh[i]);
      clMin = std::min(clMin, clh[i]);
    }
    cdMean /= static_cast<Real>(th.size() - skip);
    const Real clAmp = 0.5 * (clMax - clMin);

    {
      VtuWriter w(mesh.points(), mesh.hexes());
      w.addCellField("U", solver.velocity());
      w.addCellField("p", solver.pressure());
      std::printf("  wrote %s\n", w.write("cylinder_final").c_str());
    }

    const Real stRef = williamsonSt(Re);
    struct Check { const char* name; Real value, lo, hi; };
    const Check checks[] = {
      {"Strouhal number", St, 0.150, 0.178},
      {"mean drag Cd", cdMean, 1.25, 1.45},
      {"lift amplitude", clAmp, 0.25, 0.42},
    };
    std::printf("\n  %d shedding cycles measured after t = %.0f, max div %.1e, "
                "max Courant %.2f\n", cycles, tStats, maxCont, maxCo);
    std::printf("  %-18s %10s %18s\n", "quantity", "value", "accepted band");
    bool ok = cycles >= 3;
    if (cycles < 3) std::printf("  fewer than 3 shedding cycles: not a measurement\n");
    for (const auto& c : checks) {
      const bool pass = c.value >= c.lo && c.value <= c.hi;
      ok &= pass;
      std::printf("  %-18s %10.4f   [%.3f, %.3f]  %s\n",
                  c.name, c.value, c.lo, c.hi, pass ? "ok" : "OUT");
    }
    std::printf("  Williamson correlation gives St = %.4f; ours differs by %.1f%%\n",
                stRef, 100.0 * (St - stRef) / stRef);
    std::printf("\ncylinder wake benchmark GATE: %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
#ifdef NSFLOW_HAVE_PETSC
  PetscFinalize();
#endif
  return rc;
}
