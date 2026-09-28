// The zero-pressure-gradient flat plate, NASA TMR's 2DZP verification case,
// with the k-omega SST model (ADR-042, gate 4). One grid per run; the gate's
// comparisons across grids and against TMR's data are flat_plate_gate.py's.
//
// Re = 5e6 per unit length, U = 1, nu = 2e-7. The TMR grid, extruded one cell
// in z (cases/flatplate/make_mesh.py), spans -1/3 <= x <= 2, 0 <= y <= 1:
//   inlet    x = -1/3   u = (1, 0, 0), k = 2.25e-7, omega = 125 (TMR's
//                       free stream, nu_t/nu = 0.009)
//   symmetry y = 0, x < 0   slip; k, omega zero gradient
//   plate    y = 0, x > 0   no slip; k = 0, omega Menter's wall value
//   top      y = 1      pressure outlet p = 0, so the displacement flow leaves
//   outlet   x = 2      pressure outlet p = 0
//   z faces              zero gradient, as the cylinder's slab
// From a uniform stream, marched to a steady state -- the step ramped from
// dt/64, doubling every 100 steps, because the impulsive start puts
// S ~ 1e5 in the first cell above the wall and a full step there drives k and
// omega through zero -- until Cf at
// x = 0.97008 and the peak of nu_t/nu there changing by less than 1e-6
// relative over the last tenth of the march, and u, k and omega changing by
// less than 1e-6 of their size per step.
//
// Written out, for the gate:
//   <out>.profile   y, u, nu_t/nu at x = 0.97008 (the two cell columns on
//                   either side of that grid line, interpolated linearly)
//   <out>.cf        x, Cf along the plate, per wall face
//   and one line   FINAL Cf <Cf(0.97008)> CD <drag> tau <wall stress at 0.97008> ...
//
// Run:  flat_plate <mesh.hex> <out prefix> [variant 1994|2003, default 1994]
//       [dt, default 0.002] [max steps, default 20000]

#include "mesh/PolyMesh.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#ifdef VIBEFLOW_HAVE_PETSC
#include "linalg/PetscSolver.hpp"
#include <petscsys.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {

constexpr Real NU = 2e-7, U_IN = 1.0, K_IN = 2.25e-7, W_IN = 125.0;
constexpr Real X_CF = 0.97008;     // TMR's station; a grid line on every level

template <class V> auto host(const V& v) {
  return Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
}

std::unique_ptr<LinearSolver> makePressureSolver(const Mesh& mesh) {
  const char* e = std::getenv("VIBEFLOW_PRESSURE");
#ifdef VIBEFLOW_HAVE_PETSC
  const std::string cfg = e ? e : "cg+hypre";
#else
  const std::string cfg = e ? e : "native";
#endif
  if (cfg == "native") return std::make_unique<NativeCG>(mesh, Comm(), false);
#ifdef VIBEFLOW_HAVE_PETSC
  return std::make_unique<PetscSolver>(mesh, Comm(), cfg);
#else
  std::fprintf(stderr, "built without PETSc; VIBEFLOW_PRESSURE=%s unavailable\n", cfg.c_str());
  std::exit(2);
#endif
}

enum Patch { INLET, OUTLET, TOP, SYMMETRY, WALL, SPAN };

struct Wall { std::vector<Index> face; std::vector<Real> x; };   // sorted by x

// The wall stress on each plate face from the same two-point gradient the
// momentum equation uses there (the mesh is orthogonal): nu u_x / d_n.
std::vector<Real> wallStress(const PolyMesh& mesh, const PisoSolver& solver, const Wall& w) {
  auto u = host(solver.velocity());
  auto bc = host(mesh.boundaryCell()); auto bcen = host(mesh.boundaryCentre());
  auto ba = host(mesh.boundaryArea()); auto cc = host(mesh.cellCentre());
  std::vector<Real> tau(w.face.size());
  for (std::size_t i = 0; i < w.face.size(); ++i) {
    const Index f = w.face[i], c = bc(f);
    const Real mag = std::sqrt(ba(f,0)*ba(f,0) + ba(f,1)*ba(f,1) + ba(f,2)*ba(f,2));
    Real dn = 0.0;
    for (int k = 0; k < 3; ++k) dn += (bcen(f,k) - cc(c,k)) * ba(f,k) / mag;
    tau[i] = NU * u(c, 0) / std::abs(dn);
  }
  return tau;
}

Real interpAt(const std::vector<Real>& x, const std::vector<Real>& v, Real at) {
  for (std::size_t i = 1; i < x.size(); ++i)
    if (x[i] >= at) {
      const Real t = (at - x[i-1]) / (x[i] - x[i-1]);
      return (1.0 - t) * v[i-1] + t * v[i];
    }
  return v.back();
}

// The cells of the two columns on either side of x = X_CF, bottom to top.
struct Columns { std::vector<Index> left, right; Real xl = 0, xr = 0; };

Columns columnsAt(const PolyMesh& mesh) {
  auto cc = host(mesh.cellCentre());
  Real xl = -1e300, xr = 1e300;
  for (Index c = 0; c < mesh.nCells(); ++c) {
    const Real x = cc(c, 0);
    if (x < X_CF && x > xl) xl = x;
    if (x > X_CF && x < xr) xr = x;
  }
  Columns col; col.xl = xl; col.xr = xr;
  for (Index c = 0; c < mesh.nCells(); ++c) {
    if (std::abs(cc(c,0) - xl) < 1e-12) col.left.push_back(c);
    if (std::abs(cc(c,0) - xr) < 1e-12) col.right.push_back(c);
  }
  auto byY = [&](Index a, Index b) { return cc(a,1) < cc(b,1); };
  std::sort(col.left.begin(), col.left.end(), byY);
  std::sort(col.right.begin(), col.right.end(), byY);
  return col;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef VIBEFLOW_HAVE_PETSC
  PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    if (argc < 3) {
      std::printf("usage: flat_plate <mesh.hex> <out prefix> [1994|2003] [dt] [max steps]\n");
      Kokkos::finalize();
      return 2;
    }
    const std::string meshPath = argv[1], out = argv[2];
    const SstVariant variant = (argc > 3 && std::string(argv[3]) == "2003")
                                   ? SstVariant::Menter2003 : SstVariant::Menter1994;
    const Real dt = argc > 4 ? std::atof(argv[4]) : 0.002;
    const int maxSteps = argc > 5 ? std::atoi(argv[5]) : 20000;
    const auto t0 = std::chrono::steady_clock::now();

    const PolyMesh mesh = PolyMesh::fromHexFile(meshPath);
    const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();


    auto bcen = host(mesh.boundaryCentre()); auto ba = host(mesh.boundaryArea());
    // The domain's extent from its points: TMR writes the inlet at -0.33333.
    Real X_IN = 1e300, X_OUT = -1e300, Y_TOP = -1e300;
    for (const auto& q : mesh.points()) {
      X_IN = std::min(X_IN, q.x); X_OUT = std::max(X_OUT, q.x); Y_TOP = std::max(Y_TOP, q.y);
    }
    View1<int> uType("uType", nb), pType("pType", nb), kind("kind", nb), wallMask("wall", nb);
    VectorField ub("ub", nb, 3);
    ScalarField fb("fb", nb), pval("pval", nb), kB("kB", nb), wB("wB", nb);
    auto hUT = Kokkos::create_mirror_view(uType); auto hPT = Kokkos::create_mirror_view(pType);
    auto hK = Kokkos::create_mirror_view(kind); auto hW = Kokkos::create_mirror_view(wallMask);
    auto hUB = Kokkos::create_mirror_view(ub); auto hFB = Kokkos::create_mirror_view(fb);
    auto hKB = Kokkos::create_mirror_view(kB); auto hWB = Kokkos::create_mirror_view(wB);
    Wall wall;
    int count[6] = {0, 0, 0, 0, 0, 0};
    for (Index f = 0; f < nb; ++f) {
      const Real x = bcen(f,0), y = bcen(f,1);
      const Real mag = std::sqrt(ba(f,0)*ba(f,0) + ba(f,1)*ba(f,1) + ba(f,2)*ba(f,2));
      Patch p;
      if (std::abs(ba(f,2)) > 0.9 * mag) p = SPAN;
      else if (std::abs(x - X_IN) < 1e-9) p = INLET;
      else if (std::abs(x - X_OUT) < 1e-9) p = OUTLET;
      else if (std::abs(y - Y_TOP) < 1e-9) p = TOP;
      else if (std::abs(y) < 1e-12) p = x < 0.0 ? SYMMETRY : WALL;
      else { std::printf("unclassified boundary face at (%.6f, %.6f)\n", x, y); std::exit(2); }
      ++count[p];
      hUB(f,0) = hUB(f,1) = hUB(f,2) = 0.0; hFB(f) = 0.0;
      hKB(f) = 0.0; hWB(f) = 0.0; hW(f) = 0;
      hPT(f) = static_cast<int>(PressureBC::FixedFlux);
      hK(f) = static_cast<int>(TurbulenceBC::ZeroGradient);
      switch (p) {
        case INLET:
          hUT(f) = static_cast<int>(VelocityBC::Dirichlet);
          hUB(f,0) = U_IN; hFB(f) = U_IN * ba(f,0);
          hK(f) = static_cast<int>(TurbulenceBC::Dirichlet); hKB(f) = K_IN; hWB(f) = W_IN;
          break;
        case OUTLET: case TOP:
          hUT(f) = static_cast<int>(VelocityBC::ZeroGradient);
          hPT(f) = static_cast<int>(PressureBC::FixedValue);
          break;
        case SYMMETRY:
          hUT(f) = static_cast<int>(VelocityBC::Slip);
          break;
        case WALL:
          hUT(f) = static_cast<int>(VelocityBC::Dirichlet);
          hK(f) = static_cast<int>(TurbulenceBC::Wall); hW(f) = 1;
          wall.face.push_back(f);
          break;
        case SPAN:
          hUT(f) = static_cast<int>(VelocityBC::ZeroGradient);
          break;
      }
    }
    std::sort(wall.face.begin(), wall.face.end(),
              [&](Index a, Index b) { return bcen(a,0) < bcen(b,0); });
    for (Index f : wall.face) wall.x.push_back(bcen(f,0));
    std::printf("  patches: inlet %d  outlet %d  top %d  symmetry %d  wall %d  span %d\n",
                count[INLET], count[OUTLET], count[TOP], count[SYMMETRY], count[WALL], count[SPAN]);
    Kokkos::deep_copy(uType, hUT); Kokkos::deep_copy(pType, hPT); Kokkos::deep_copy(kind, hK);
    Kokkos::deep_copy(wallMask, hW); Kokkos::deep_copy(ub, hUB); Kokkos::deep_copy(fb, hFB);
    Kokkos::deep_copy(kB, hKB); Kokkos::deep_copy(wB, hWB);

    PisoControls ctl;
    ctl.outer = 1;
    ctl.correctors = 2;
    if (const char* e = std::getenv("VIBEFLOW_OUTER")) ctl.outer = std::atoi(e);
    // Momentum's face value: linear upwind, because central differencing let
    // the leading edge set the odd-even mode going upstream along the
    // symmetry plane (ADR-042). VIBEFLOW_CONVECTION=linear, or =upwind for
    // first order, reproduces the record.
    ctl.convection = ConvectionScheme::LinearUpwind;
    if (const char* e = std::getenv("VIBEFLOW_CONVECTION")) {
      const std::string c(e);
      if (c == "linear") ctl.convection = ConvectionScheme::Linear;
      else if (c == "upwind") ctl.deferredCorrection = false;
    }
    std::printf("flat plate, SST-%s, %s: %d cells, dt %g, momentum %s\n",
                variant == SstVariant::Menter2003 ? "2003" : "1994", meshPath.c_str(),
                static_cast<int>(nc), dt,
                !ctl.deferredCorrection ? "upwind"
                : ctl.convection == ConvectionScheme::LinearUpwind ? "linear upwind" : "linear");
    Real dtNow = dt / 64.0;
    PisoSolver solver(mesh, NU, dtNow, ctl);
    solver.setBoundaryTypes(uType);
    solver.setPressureBoundary(pType, pval);
    {
      VectorField u0("u0", nt, 3);
      Kokkos::deep_copy(u0, 0.0);
      auto hu = host(u0);
      for (Index c = 0; c < nt; ++c) hu(c, 0) = U_IN;
      Kokkos::deep_copy(u0, hu);
      ScalarField p0("p0", nt), F0("F0", mesh.nInternalFaces());
      auto fa = mesh.faceArea();
      Kokkos::parallel_for("F0", Kokkos::RangePolicy<ExecSpace>(0, mesh.nInternalFaces()),
        KOKKOS_LAMBDA(const Index f) { F0(f) = U_IN * fa(f, 0); });
      Kokkos::fence();
      solver.setState(u0, p0, F0);
    }
    TurbulenceModel tm;
    tm.variant = variant;
    solver.enableTurbulence(tm, wallMask);
    bool wallDistOk = true;
    {
      // Gate 1 on this grid (ADR-042): d = y above the plate and the distance
      // to its leading edge, sqrt(x^2 + y^2), ahead of it.
      auto d = host(solver.wallDistance()); auto dB = host(solver.boundaryWallDistance());
      auto cc = host(mesh.cellCentre());
      auto exact = [](Real x, Real y) { return x >= 0.0 ? y : std::sqrt(x * x + y * y); };
      Real eC = 0.0, eB = 0.0;
      for (Index c = 0; c < nc; ++c) eC = std::max(eC, std::abs(d(c) - exact(cc(c,0), cc(c,1))));
      for (Index f = 0; f < nb; ++f) eB = std::max(eB, std::abs(dB(f) - exact(bcen(f,0), bcen(f,1))));
      std::printf("  wall distance: max|d - d_exact| cells %.1e, boundary faces %.1e\n", eC, eB);
      wallDistOk = eC <= 1e-13 && eB <= 1e-13;
    }
    // VIBEFLOW_FP_WALLDIST_ONLY: that check alone, for grids marched before
    // it was added.
    if (std::getenv("VIBEFLOW_FP_WALLDIST_ONLY")) {
      rc = wallDistOk ? 0 : 1;
    } else {
    solver.setTurbulenceBoundary(kind, kB, wB);
    {
      ScalarField k0("k0", nt), w0("w0", nt);
      Kokkos::deep_copy(k0, K_IN);
      Kokkos::deep_copy(w0, W_IN);
      solver.setTurbulence(k0, w0);
    }

    // Momentum, k and omega share one solver: the native Jacobi BiCGStab, or
    // a PETSc configuration (VIBEFLOW_MOMENTUM=bicgstab+ilu). The steady
    // state does not depend on which (ADR-042).
    std::unique_ptr<LinearSolver> momentumPtr;
    {
      const char* e = std::getenv("VIBEFLOW_MOMENTUM");
      const std::string cfg = e ? e : "native";
      if (cfg == "native") momentumPtr = std::make_unique<NativeBiCGStab>(mesh);
#ifdef VIBEFLOW_HAVE_PETSC
      else momentumPtr = std::make_unique<PetscSolver>(mesh, Comm(), cfg);
#else
      else { std::fprintf(stderr, "built without PETSc\n"); std::exit(2); }
#endif
      std::printf("  momentum, k and omega solver: %s\n", cfg.c_str());
    }
    LinearSolver& momentum = *momentumPtr;
    auto pressure = makePressureSolver(mesh);
    const Columns col = columnsAt(mesh);
    const Real tcol = (X_CF - col.xl) / (col.xr - col.xl);

    auto nutPeak = [&]() {
      auto nut = host(solver.eddyViscosity());
      Real best = 0.0;
      for (std::size_t j = 0; j < col.left.size(); ++j)
        best = std::max(best, ((1.0 - tcol) * nut(col.left[j]) + tcol * nut(col.right[j])) / NU);
      return best;
    };
    std::vector<Real> cfHist, peakHist;
    std::vector<Real> uPrev, kPrev, wPrev;
    auto snapshot = [&](std::vector<Real>& uu, std::vector<Real>& kk, std::vector<Real>& ww) {
      auto u = host(solver.velocity()); auto k = host(solver.turbulentKineticEnergy());
      auto w = host(solver.specificDissipation());
      uu.resize(nc * 3); kk.resize(nc); ww.resize(nc);
      for (Index c = 0; c < nc; ++c) {
        for (int d = 0; d < 3; ++d) uu[c*3 + d] = u(c, d);
        kk[c] = k(c); ww[c] = w(c);
      }
    };
    // The largest change of a field over 50 steps relative to its largest
    // size, and the cell it is in.
    struct Change { Real rel = 0.0; Index cell = 0; };
    auto relChange = [](const std::vector<Real>& a, const std::vector<Real>& b, int stride) {
      Real d = 0.0, s = 1e-300;
      Change c;
      for (std::size_t i = 0; i < a.size(); ++i) {
        const Real di = std::abs(a[i] - b[i]);
        if (di > d) { d = di; c.cell = static_cast<Index>(i / stride); }
        s = std::max(s, std::abs(a[i]));
      }
      c.rel = d / s;
      return c;
    };
    auto ccH = host(mesh.cellCentre());
    bool steady = false;
    int step = 0;
    Real change = 1.0, cf = 0.0;
    std::vector<Real> uN, kN, wN;
    snapshot(uPrev, kPrev, wPrev);
    VectorField src("src", nt, 3);
    Real time = 0.0;
    for (step = 1; step <= maxSteps; ++step) {
      if (step > 1 && (step - 1) % 100 == 0 && dtNow < dt) {
        dtNow = std::min(2.0 * dtNow, dt);
        solver.setTimeStep(dtNow);
      }
      solver.advance(ub, fb, src, momentum, *pressure);
      time += dtNow;
      const auto tau = wallStress(mesh, solver, wall);
      cf = 2.0 * interpAt(wall.x, tau, X_CF);
      cfHist.push_back(cf);
      peakHist.push_back(nutPeak());
      if (std::getenv("VIBEFLOW_FP_DEBUG") && (step <= 20 || step % 10 == 0)) {
        // Exploration only: where k and omega go, cell by cell.
        auto k = host(solver.turbulentKineticEnergy()); auto w = host(solver.specificDissipation());
        auto nut = host(solver.eddyViscosity()); auto cc = host(mesh.cellCentre());
        Index ik = 0, iw = 0, iwx = 0, in = 0;
        for (Index c = 0; c < nc; ++c) {
          if (k(c) < k(ik)) ik = c;
          if (w(c) < w(iw)) iw = c;
          if (w(c) > w(iwx)) iwx = c;
          if (nut(c) > nut(in)) in = c;
        }
        std::printf("    dbg %4d  k min %.3e at (%.4f, %.2e)  w min %.3e at (%.4f, %.2e)  "
                    "w max %.3e at (%.4f, %.2e)  nut/nu max %.3e at (%.4f, %.2e)  bounded %lld\n",
                    step, k(ik), cc(ik,0), cc(ik,1), w(iw), cc(iw,0), cc(iw,1),
                    w(iwx), cc(iwx,0), cc(iwx,1), nut(in) / NU, cc(in,0), cc(in,1),
                    solver.boundedCells());
      }
      if (!std::isfinite(cf)) { std::printf("  step %d: not finite\n", step); break; }
      if (step % 50 == 0 || step == maxSteps) {
        snapshot(uN, kN, wN);
        const Change cu = relChange(uN, uPrev, 3), ck = relChange(kN, kPrev, 1),
                     cw = relChange(wN, wPrev, 1);
        change = std::max({cu.rel, ck.rel, cw.rel});
        uPrev = uN; kPrev = kN; wPrev = wN;
        // Over the last tenth of the march, how far Cf and the nu_t peak wandered.
        const std::size_t n0 = cfHist.size() - std::max<std::size_t>(cfHist.size() / 10, 1);
        Real dcf = 0.0, dpk = 0.0;
        for (std::size_t i = n0; i < cfHist.size(); ++i) {
          dcf = std::max(dcf, std::abs(cfHist[i] - cf) / std::abs(cf));
          dpk = std::max(dpk, std::abs(peakHist[i] - peakHist.back()) / peakHist.back());
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        // change is over 50 steps; per step, at most that.
        std::printf("  step %6d  t %8.3f  dt %.1e  Cf %.8e  nut/nu peak %.4f  drift %.1e %.1e  "
                    "change/50 steps %.1e  bounded %lld  (%.0f s)\n",
                    step, time, dtNow, cf, peakHist.back(), dcf, dpk, change,
                    solver.boundedCells(), secs);
        std::printf("      change: u %.1e at (%.4f, %.2e)  k %.1e at (%.4f, %.2e)  "
                    "omega %.1e at (%.4f, %.2e)\n",
                    cu.rel, ccH(cu.cell,0), ccH(cu.cell,1), ck.rel, ccH(ck.cell,0), ccH(ck.cell,1),
                    cw.rel, ccH(cw.cell,0), ccH(cw.cell,1));
        std::fflush(stdout);
        if (dcf < 1e-6 && dpk < 1e-6 && change < 1e-6 && dtNow == dt) { steady = true; break; }
      }
    }
    if (step > maxSteps) step = maxSteps;

    // The profile at x = 0.97008 and Cf along the plate.
    {
      auto u = host(solver.velocity()); auto nut = host(solver.eddyViscosity());
      auto cc = host(mesh.cellCentre());
      std::FILE* fp = std::fopen((out + ".profile").c_str(), "w");
      std::fprintf(fp, "# y  u  nu_t/nu  at x = %.5f (flat_plate, SST-%s)\n", X_CF,
                   variant == SstVariant::Menter2003 ? "2003" : "1994");
      for (std::size_t j = 0; j < col.left.size(); ++j) {
        const Index a = col.left[j], b = col.right[j];
        std::fprintf(fp, "%.10e %.10e %.10e\n", cc(a,1),
                     (1.0 - tcol) * u(a,0) + tcol * u(b,0),
                     ((1.0 - tcol) * nut(a) + tcol * nut(b)) / NU);
      }
      std::fclose(fp);
      const auto tau = wallStress(mesh, solver, wall);
      std::FILE* fc = std::fopen((out + ".cf").c_str(), "w");
      std::fprintf(fc, "# x  Cf\n");
      Real drag = 0.0;
      auto baH = host(mesh.boundaryArea());
      for (std::size_t i = 0; i < tau.size(); ++i) {
        const Index f = wall.face[i];
        const Real area = std::sqrt(baH(f,0)*baH(f,0) + baH(f,1)*baH(f,1) + baH(f,2)*baH(f,2));
        std::fprintf(fc, "%.10e %.10e\n", wall.x[i], 2.0 * tau[i]);
        drag += tau[i] * area;
      }
      std::fclose(fc);
      if (std::getenv("VIBEFLOW_FP_DUMP")) {
        // Exploration only: every cell's x, y, u, v, k, omega, nu_t/nu.
        auto k = host(solver.turbulentKineticEnergy()); auto w = host(solver.specificDissipation());
        auto pp = host(solver.pressure());
        std::FILE* fd = std::fopen((out + ".cells").c_str(), "w");
        for (Index c = 0; c < nc; ++c)
          std::fprintf(fd, "%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", cc(c,0), cc(c,1),
                       u(c,0), u(c,1), k(c), w(c), nut(c) / NU, pp(c));
        std::fclose(fd);
      }
      // CD = drag / (1/2 rho U^2 L_ref b): L_ref = 2, the plate, and b the
      // slab's thickness -- TMR's definition, (1/2) of the integral of Cf.
      Real zmin = 1e300, zmax = -1e300;
      for (const auto& p : mesh.points()) { zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z); }
      const Real cd = drag / (0.5 * U_IN * U_IN * 2.0 * (zmax - zmin));
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      // Where the time went: the solver's own phases; the rest of "total" is
      // mostly the k and omega solves.
      const PisoTimings& ts = solver.timings();
      const Real known = ts.assemble + ts.gradient + ts.boundaryP + ts.hbya + ts.rhieChow
                         + ts.pressureAssembly + ts.momentumSolve + ts.pressureSolve;
      std::printf("timings: total %.0f s  pressure solve %.0f  momentum solve %.0f  "
                  "assembly %.0f  gradients %.0f  other (k, omega) %.0f\n",
                  ts.total, ts.pressureSolve, ts.momentumSolve, ts.assemble + ts.pressureAssembly,
                  ts.gradient + ts.boundaryP, ts.total - known);
      std::printf("linear solves: momentum, k and omega %.0f s (%d solves, %d iterations); "
                  "pressure %.0f s (%d solves, %d iterations)\n",
                  momentum.totalSeconds(), momentum.solveCount(), momentum.totalIterations(),
                  pressure->totalSeconds(), pressure->solveCount(), pressure->totalIterations());
      std::printf("FINAL Cf %.10e CD %.10e tau %.10e nutPeak %.6f steps %d steady %s "
                  "bounded %lld seconds %.0f\n",
                  cf, cd, 0.5 * cf, peakHist.empty() ? 0.0 : peakHist.back(), step,
                  steady ? "yes" : "no", solver.boundedCells(), secs);
    }
    rc = steady && wallDistOk ? 0 : 1;
    }   // not VIBEFLOW_FP_WALLDIST_ONLY
  }
  Kokkos::finalize();
#ifdef VIBEFLOW_HAVE_PETSC
  PetscFinalize();
#endif
  return rc;
}
