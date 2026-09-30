// The backward-facing step, NASA TMR's 2D validation case, with the k-omega
// SST model (ADR-045). One grid per run; the gate's comparisons across grids
// and against TMR's CFL3D results are backstep_gate.py's.
//
// Re_H = 36,000 on the reference velocity U = 1, nu = 1/36,000; the step
// height H = 1. The TMR grid (cases/backstep/make_mesh.py), four zones merged
// and extruded one cell in z, spans -130 <= x <= 50; 1 <= y <= 9 upstream of
// the step at x = 0, 0 <= y <= 9 behind it:
//   inlet     x = -130   u = (U_in, 0, 0), k = 5.49e-7, omega = 2.197 (TMR's
//                        free stream for SST at M = 0.128, nu_t/nu = 0.009)
//   walls     y = 1 (x < 0), y = 0 (x > 0), the step's face x = 0 and the top
//             y = 9: slip for x < -110, as TMR's grids mark them, no slip
//             after; k = 0, omega Menter's wall value on the no-slip part
//   outlet    x = 50     pressure outlet p = 0
//   z faces              zero gradient, as the flat plate's slab
// U_in is the inlet speed, given on the command line: ADR-045 sets it once,
// on level 3, so that the channel's centre velocity at x = -4 is CFL3D's.
//
// From the uniform stream, or from a coarser grid's state (VIBEFLOW_BS_INIT,
// each cell taking the state of the nearest coarse cell), marched from dt/64
// with a ramp that follows the largest cell Courant number C (ADR-045's third
// revision): every 100 steps dt doubles, never past the level's dt, if C at
// the doubled dt is at most 8; whenever C exceeds 12 it halves and holds for
// 100 steps. Once the march has held its dt for 1,000 steps -- the level's, or
// the one the cap refused to double -- it ends at the steady criteria of the
// flat plate (u, k and omega changing by less than 1e-6 of their size over 50
// steps) or ADR-045's quasi-steady one: the reattachment point and Cf at
// x = -4 moving by no more than 0.2% over the march's last 20%. Otherwise it
// ends after the given number of steps at that dt, or 20,000 in all.
//
// Written out, for the gate:
//   <out>.cf        x, Cf on the bottom wall (y = 1 upstream, y = 0 behind),
//                   per wall face, on (1/2) rho U^2
//   <out>.cp        x, Cp there, from the wall cells' pressure, shifted to 0 at
//                   x = 40 as TMR shifts its own
//   <out>.prof      x_station, y, u at x = -4, 1, 4, 6, 10: the two cell
//                   columns on either side of each station, interpolated
//   <out>.state     x y u v w p k omega per cell (VIBEFLOW_BS_SAVE)
//   and one line    FINAL xr ... cf4 ... uc4 ... bubble ... steps ... settled ...
//
// Run:  backstep <mesh.hex> <out prefix> [dt, default 0.08] [max steps at the
//       full dt, default 5000] [U_in, default 1]
//
// Exploration only, not used by the gate: VIBEFLOW_BS_DEBUG (each step's
// extremes and where they are), VIBEFLOW_BS_SAVE_AT=<step>[,...] (the state
// after those steps too), VIBEFLOW_BS_STOP=<step>, VIBEFLOW_BS_RAMP=<n> (start
// at dt/n), VIBEFLOW_BS_CUP / VIBEFLOW_BS_CDOWN (the ramp's thresholds),
// VIBEFLOW_BS_FROZEN, VIBEFLOW_CONVECTION, VIBEFLOW_NO_NONORTH,
// VIBEFLOW_CORRECTORS.

#include "mesh/PolyMesh.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#ifdef VIBEFLOW_HAVE_PETSC
#include "linalg/PetscSolver.hpp"
#include <petscsys.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {

constexpr Real NU = 1.0 / 36000.0, K_IN = 5.493e-7, W_IN = 2.197;
constexpr Real X_SLIP = -110.0;               // slip walls upstream of this
constexpr Real X_CF4 = -4.0;                  // the upstream station
const Real STATIONS[] = {-4.0, 1.0, 4.0, 6.0, 10.0};

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

enum Patch { INLET, OUTLET, SLIP, WALL, SPAN };

// The bottom wall's faces, upstream (y = 1) and behind the step (y = 0),
// sorted by x.
struct Bottom { std::vector<Index> face; std::vector<Real> x; };

// Wall stress on each bottom face from the two-point gradient the momentum
// equation uses there: nu u_x / d_n, signed, so the recirculation is negative.
std::vector<Real> wallStress(const PolyMesh& mesh, const PisoSolver& solver, const Bottom& w) {
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

// The reattachment point: where Cf last changes sign from negative to
// positive behind the step, linear between faces; the corner bubble: the
// first two sign changes behind the step, if Cf starts positive there.
struct Separation { Real xr = -1.0, c0 = -1.0, c1 = -1.0; };
Separation separation(const std::vector<Real>& x, const std::vector<Real>& cf) {
  Separation s;
  std::vector<Real> zs;
  for (std::size_t i = 1; i < x.size(); ++i) {
    if (x[i-1] < 0.0) continue;
    if ((cf[i-1] < 0.0) != (cf[i] < 0.0))
      zs.push_back(x[i-1] - cf[i-1] * (x[i] - x[i-1]) / (cf[i] - cf[i-1]));
  }
  for (std::size_t i = 1; i < x.size(); ++i) {
    if (x[i-1] < 0.0) continue;
    if (cf[i-1] < 0.0 && cf[i] >= 0.0)
      s.xr = x[i-1] - cf[i-1] * (x[i] - x[i-1]) / (cf[i] - cf[i-1]);
  }
  if (zs.size() >= 3) { s.c0 = zs[0]; s.c1 = zs[1]; }
  return s;
}

// A coarser grid's state and a bucket grid over its cell centres, for the
// nearest coarse cell to a point.
struct CoarseState {
  std::vector<std::array<Real, 8>> v;          // x y u v w p k omega
  Real x0 = 0, y0 = 0, hx = 1, hy = 1;
  int nx = 1, ny = 1;
  std::vector<std::vector<Index>> bucket;
};

CoarseState readState(const std::string& path) {
  CoarseState cs;
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f) { std::printf("cannot read %s\n", path.c_str()); std::exit(2); }
  long n = 0;
  if (std::fscanf(f, "%ld", &n) != 1) { std::printf("bad state file %s\n", path.c_str()); std::exit(2); }
  cs.v.resize(n);
  Real xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
  for (long c = 0; c < n; ++c) {
    double a[8];
    for (int i = 0; i < 8; ++i)
      if (std::fscanf(f, "%lf", &a[i]) != 1) { std::printf("bad state file %s\n", path.c_str()); std::exit(2); }
    for (int i = 0; i < 8; ++i) cs.v[c][i] = a[i];
    xmin = std::min(xmin, a[0]); xmax = std::max(xmax, a[0]);
    ymin = std::min(ymin, a[1]); ymax = std::max(ymax, a[1]);
  }
  std::fclose(f);
  cs.nx = 720; cs.ny = 72;
  cs.x0 = xmin; cs.y0 = ymin;
  cs.hx = (xmax - xmin) / cs.nx * (1.0 + 1e-12) + 1e-300;
  cs.hy = (ymax - ymin) / cs.ny * (1.0 + 1e-12) + 1e-300;
  cs.bucket.resize(static_cast<std::size_t>(cs.nx) * cs.ny);
  for (long c = 0; c < n; ++c) {
    const int i = std::min(cs.nx - 1, static_cast<int>((cs.v[c][0] - cs.x0) / cs.hx));
    const int j = std::min(cs.ny - 1, static_cast<int>((cs.v[c][1] - cs.y0) / cs.hy));
    cs.bucket[static_cast<std::size_t>(j) * cs.nx + i].push_back(c);
  }
  return cs;
}

// The nearest coarse centre: the point's bucket and rings around it until a
// ring beyond the best distance so far has been searched.
Index nearestCoarse(const CoarseState& cs, Real x, Real y) {
  const int i0 = std::clamp(static_cast<int>((x - cs.x0) / cs.hx), 0, cs.nx - 1);
  const int j0 = std::clamp(static_cast<int>((y - cs.y0) / cs.hy), 0, cs.ny - 1);
  Index best = -1;
  Real bd = 1e300;
  for (int r = 0; r < std::max(cs.nx, cs.ny); ++r) {
    for (int j = j0 - r; j <= j0 + r; ++j)
      for (int i = i0 - r; i <= i0 + r; ++i) {
        if (i < 0 || j < 0 || i >= cs.nx || j >= cs.ny) continue;
        if (std::max(std::abs(i - i0), std::abs(j - j0)) != r) continue;
        for (Index c : cs.bucket[static_cast<std::size_t>(j) * cs.nx + i]) {
          const Real dx = cs.v[c][0] - x, dy = cs.v[c][1] - y;
          const Real d = dx * dx + dy * dy;
          if (d < bd) { bd = d; best = c; }
        }
      }
    // Every point of ring r+1 is at least r * min(hx, hy) away.
    const Real reach = r * std::min(cs.hx, cs.hy);
    if (best >= 0 && reach * reach > bd) break;
  }
  return best;
}

// The cell columns around a station: the largest cell-centre x below it and
// the smallest above, each column's cells sorted by y.
struct Columns { std::vector<Index> left, right; Real xl = 0, xr = 0; };

Columns columnsAt(const PolyMesh& mesh, Real at) {
  auto cc = host(mesh.cellCentre());
  Real xl = -1e300, xr = 1e300;
  for (Index c = 0; c < mesh.nCells(); ++c) {
    const Real x = cc(c, 0);
    if (x < at && x > xl) xl = x;
    if (x > at && x < xr) xr = x;
  }
  Columns col; col.xl = xl; col.xr = xr;
  for (Index c = 0; c < mesh.nCells(); ++c) {
    if (std::abs(cc(c,0) - xl) < 1e-9) col.left.push_back(c);
    if (std::abs(cc(c,0) - xr) < 1e-9) col.right.push_back(c);
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
      std::printf("usage: backstep <mesh.hex> <out prefix> [dt] [max steps at the full dt] [U_in]\n");
      Kokkos::finalize();
      return 2;
    }
    const std::string meshPath = argv[1], out = argv[2];
    const Real dt = argc > 3 ? std::atof(argv[3]) : 0.08;
    const int maxFull = argc > 4 ? std::atoi(argv[4]) : 5000;
    const Real uIn = argc > 5 ? std::atof(argv[5]) : 1.0;
    const auto t0 = std::chrono::steady_clock::now();

    const PolyMesh mesh = PolyMesh::fromHexFile(meshPath);
    const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();
    auto bcen = host(mesh.boundaryCentre()); auto ba = host(mesh.boundaryArea());
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
    Bottom bottom;
    int count[5] = {0, 0, 0, 0, 0};
    Real volume = 0.0;
    for (Index f = 0; f < nb; ++f) {
      const Real x = bcen(f,0), y = bcen(f,1);
      const Real mag = std::sqrt(ba(f,0)*ba(f,0) + ba(f,1)*ba(f,1) + ba(f,2)*ba(f,2));
      Patch p;
      bool isBottom = false;
      if (std::abs(ba(f,2)) > 0.9 * mag) p = SPAN;
      else if (std::abs(x - X_IN) < 1e-9) p = INLET;
      else if (std::abs(x - X_OUT) < 1e-9) p = OUTLET;
      else if (std::abs(y - Y_TOP) < 1e-9) p = x < X_SLIP ? SLIP : WALL;
      else if (std::abs(y - 1.0) < 1e-9 && x < 0.0) { p = x < X_SLIP ? SLIP : WALL; isBottom = true; }
      else if (std::abs(y) < 1e-12 && x > 0.0) { p = WALL; isBottom = true; }
      else if (std::abs(x) < 1e-12 && y < 1.0) p = WALL;             // the step's face
      else { std::printf("unclassified boundary face at (%.6f, %.6f)\n", x, y); std::exit(2); }
      ++count[p];
      hUB(f,0) = hUB(f,1) = hUB(f,2) = 0.0; hFB(f) = 0.0;
      hKB(f) = 0.0; hWB(f) = 0.0; hW(f) = 0;
      hPT(f) = static_cast<int>(PressureBC::FixedFlux);
      hK(f) = static_cast<int>(TurbulenceBC::ZeroGradient);
      switch (p) {
        case INLET:
          hUT(f) = static_cast<int>(VelocityBC::Dirichlet);
          hUB(f,0) = uIn; hFB(f) = uIn * ba(f,0);
          hK(f) = static_cast<int>(TurbulenceBC::Dirichlet); hKB(f) = K_IN; hWB(f) = W_IN;
          break;
        case OUTLET:
          hUT(f) = static_cast<int>(VelocityBC::ZeroGradient);
          hPT(f) = static_cast<int>(PressureBC::FixedValue);
          break;
        case SLIP:
          hUT(f) = static_cast<int>(VelocityBC::Slip);
          break;
        case WALL:
          hUT(f) = static_cast<int>(VelocityBC::Dirichlet);
          hK(f) = static_cast<int>(TurbulenceBC::Wall); hW(f) = 1;
          if (isBottom) bottom.face.push_back(f);
          break;
        case SPAN:
          hUT(f) = static_cast<int>(VelocityBC::ZeroGradient);
          break;
      }
    }
    std::sort(bottom.face.begin(), bottom.face.end(),
              [&](Index a, Index b) { return bcen(a,0) < bcen(b,0); });
    for (Index f : bottom.face) bottom.x.push_back(bcen(f,0));
    {
      auto vol = host(mesh.cellVolume());
      for (Index c = 0; c < nc; ++c) volume += vol(c);
    }
    Real zmin = 1e300, zmax = -1e300;
    for (const auto& q : mesh.points()) { zmin = std::min(zmin, q.z); zmax = std::max(zmax, q.z); }
    std::printf("backward-facing step, SST-1994, %s: %d cells, volume %.15g (area %.15g), "
                "dt %g, U_in %.6f\n", meshPath.c_str(), static_cast<int>(nc), volume,
                volume / (zmax - zmin), dt, uIn);
    std::printf("  patches: inlet %d  outlet %d  slip %d  wall %d  span %d  (bottom wall %zu)\n",
                count[INLET], count[OUTLET], count[SLIP], count[WALL], count[SPAN],
                bottom.face.size());
    Kokkos::deep_copy(uType, hUT); Kokkos::deep_copy(pType, hPT); Kokkos::deep_copy(kind, hK);
    Kokkos::deep_copy(wallMask, hW); Kokkos::deep_copy(ub, hUB); Kokkos::deep_copy(fb, hFB);
    Kokkos::deep_copy(kB, hKB); Kokkos::deep_copy(wB, hWB);

    PisoControls ctl;
    ctl.outer = 1;
    // Four correctors, as the flat plate's (ADR-043, ADR-045).
    ctl.correctors = 4;
    if (const char* e = std::getenv("VIBEFLOW_CORRECTORS")) ctl.correctors = std::atoi(e);
    ctl.convection = ConvectionScheme::LinearUpwind;
    // Exploration only: the face value (linear, or upwind for first order)
    // and the explicit non-orthogonal diffusion.
    if (const char* e = std::getenv("VIBEFLOW_CONVECTION")) {
      const std::string c(e);
      if (c == "linear") ctl.convection = ConvectionScheme::Linear;
      else if (c == "upwind") ctl.deferredCorrection = false;
    }
    if (std::getenv("VIBEFLOW_NO_NONORTH")) ctl.diffusionNonOrth = false;
    Real dtNow = dt / 64.0;
    if (const char* e = std::getenv("VIBEFLOW_BS_RAMP")) dtNow = dt / std::max(1.0, std::atof(e));
    PisoSolver solver(mesh, NU, dtNow, ctl);
    solver.setBoundaryTypes(uType);
    solver.setPressureBoundary(pType, pval);

    const char* initPath = std::getenv("VIBEFLOW_BS_INIT");
    CoarseState coarse;
    std::vector<Index> fromCoarse;
    auto ccH = host(mesh.cellCentre());
    if (initPath) {
      coarse = readState(initPath);
      fromCoarse.resize(nt);
      for (Index c = 0; c < nt; ++c) fromCoarse[c] = nearestCoarse(coarse, ccH(c,0), ccH(c,1));
      std::printf("  initial state: %s (%zu coarse cells)\n", initPath, coarse.v.size());
    }
    {
      VectorField u0("u0", nt, 3);
      ScalarField p0("p0", nt), F0("F0", mesh.nInternalFaces());
      auto hu = host(u0); auto hp = host(p0);
      for (Index c = 0; c < nt; ++c) {
        if (initPath) {
          const auto& q = coarse.v[fromCoarse[c]];
          hu(c,0) = q[2]; hu(c,1) = q[3]; hu(c,2) = q[4]; hp(c) = q[5];
        } else {
          hu(c,0) = uIn; hu(c,1) = 0.0; hu(c,2) = 0.0; hp(c) = 0.0;
        }
      }
      Kokkos::deep_copy(u0, hu); Kokkos::deep_copy(p0, hp);
      auto fa = mesh.faceArea(); auto fc = mesh.faceCentre(); auto ccd = mesh.cellCentre();
      auto own = mesh.owner(); auto nei = mesh.neighbour();
      Kokkos::parallel_for("F0", Kokkos::RangePolicy<ExecSpace>(0, mesh.nInternalFaces()),
        KOKKOS_LAMBDA(const Index f) {
          Real lo = 0.0, ln = 0.0;
          for (int i = 0; i < 3; ++i) {
            const Real ro = fc(f, i) - ccd(own(f), i), rn = fc(f, i) - ccd(nei(f), i);
            lo += ro * ro; ln += rn * rn;
          }
          const Real w = Kokkos::sqrt(ln) / (Kokkos::sqrt(lo) + Kokkos::sqrt(ln));
          Real sum = 0.0;
          for (int i = 0; i < 3; ++i) sum += (w * u0(own(f), i) + (1.0 - w) * u0(nei(f), i)) * fa(f, i);
          F0(f) = sum;
        });
      Kokkos::fence();
      solver.setState(u0, p0, F0);
    }
    TurbulenceModel tm;
    tm.variant = SstVariant::Menter1994;
    if (std::getenv("VIBEFLOW_BS_FROZEN")) tm.frozen = true;      // exploration only
    solver.enableTurbulence(tm, wallMask);
    solver.setTurbulenceBoundary(kind, kB, wB);
    {
      ScalarField k0("k0", nt), w0("w0", nt);
      Kokkos::deep_copy(k0, K_IN); Kokkos::deep_copy(w0, W_IN);
      if (initPath) {
        auto hk = host(k0); auto hw = host(w0);
        for (Index c = 0; c < nt; ++c) {
          hk(c) = coarse.v[fromCoarse[c]][6]; hw(c) = coarse.v[fromCoarse[c]][7];
        }
        Kokkos::deep_copy(k0, hk); Kokkos::deep_copy(w0, hw);
      }
      solver.setTurbulence(k0, w0);
    }

    std::unique_ptr<LinearSolver> momentumPtr;
    {
      const char* e = std::getenv("VIBEFLOW_MOMENTUM");
#ifdef VIBEFLOW_HAVE_PETSC
      const std::string cfg = e ? e : "bicgstab+ilu";
#else
      const std::string cfg = e ? e : "native";
#endif
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

    // The channel's centre velocity at x = -4: the two columns around it at
    // the row nearest y = 5.
    const Columns c4 = columnsAt(mesh, X_CF4);
    auto centreU = [&]() {
      auto u = host(solver.velocity());
      auto near5 = [&](const std::vector<Index>& col) {
        Index best = col[0];
        for (Index c : col) if (std::abs(ccH(c,1) - 5.0) < std::abs(ccH(best,1) - 5.0)) best = c;
        return best;
      };
      const Index a = near5(c4.left), b = near5(c4.right);
      const Real t = (X_CF4 - c4.xl) / (c4.xr - c4.xl);
      return (1.0 - t) * u(a, 0) + t * u(b, 0);
    };

    std::vector<Real> uPrev, kPrev, wPrev, uN, kN, wN;
    auto snapshot = [&](std::vector<Real>& uu, std::vector<Real>& kk, std::vector<Real>& ww) {
      auto u = host(solver.velocity()); auto k = host(solver.turbulentKineticEnergy());
      auto w = host(solver.specificDissipation());
      uu.resize(nc * 3); kk.resize(nc); ww.resize(nc);
      for (Index c = 0; c < nc; ++c) {
        for (int d = 0; d < 3; ++d) uu[c*3 + d] = u(c, d);
        kk[c] = k(c); ww[c] = w(c);
      }
    };
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
    snapshot(uPrev, kPrev, wPrev);

    // The state for a finer grid's start: x y u v w p k omega per cell.
    auto saveState = [&](const std::string& path) {
      auto u = host(solver.velocity()); auto p = host(solver.pressure());
      auto k = host(solver.turbulentKineticEnergy()); auto w = host(solver.specificDissipation());
      std::FILE* fs = std::fopen(path.c_str(), "w");
      std::fprintf(fs, "%ld\n", static_cast<long>(nc));
      for (Index c = 0; c < nc; ++c)
        std::fprintf(fs, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", ccH(c,0), ccH(c,1),
                     u(c,0), u(c,1), u(c,2), p(c), k(c), w(c));
      std::fclose(fs);
    };
    // Exploration only: VIBEFLOW_BS_SAVE_AT=<step>[,<step>...] also saves
    // the state after those steps, to <out>.state.<step>; VIBEFLOW_BS_STOP
    // ends the march after the step it names.
    std::vector<int> saveAt;
    if (const char* e = std::getenv("VIBEFLOW_BS_SAVE_AT")) {
      std::string list(e);
      for (std::size_t a = 0; a < list.size();) {
        const std::size_t b = std::min(list.find(',', a), list.size());
        saveAt.push_back(std::atoi(list.substr(a, b - a).c_str()));
        a = b + 1;
      }
    }
    const int stopAt = std::getenv("VIBEFLOW_BS_STOP") ? std::atoi(std::getenv("VIBEFLOW_BS_STOP")) : -1;

    // The ramp follows the flow's largest cell Courant number (ADR-045's
    // third revision): C = dt sum_f |F_f| / 2V. Every 100 steps dt doubles,
    // never past the level's dt, if C at the doubled dt is at most C_UP;
    // whenever C exceeds C_DOWN it halves and holds for 100 steps.
    Real C_UP = 8.0, C_DOWN = 12.0;
    // Exploration only: other thresholds (VIBEFLOW_BS_CUP, VIBEFLOW_BS_CDOWN).
    if (const char* e = std::getenv("VIBEFLOW_BS_CUP")) C_UP = std::atof(e);
    if (const char* e = std::getenv("VIBEFLOW_BS_CDOWN")) C_DOWN = std::atof(e);
    const int MAX_STEPS = 20000;
    auto ownH = host(mesh.owner()); auto neiH = host(mesh.neighbour());
    auto bclH = host(mesh.boundaryCell()); auto volH = host(mesh.cellVolume());
    const Index nif = mesh.nInternalFaces();
    std::vector<Real> fsum(nt);
    struct Cfl { Real c = 0.0; Index cell = 0; };
    auto courant = [&](Real dtTest) {
      auto F = host(solver.faceFlux()); auto Fb = host(solver.boundaryFlux());
      std::fill(fsum.begin(), fsum.end(), 0.0);
      for (Index f = 0; f < nif; ++f) {
        const Real a = std::abs(F(f));
        fsum[ownH(f)] += a; fsum[neiH(f)] += a;
      }
      for (Index f = 0; f < nb; ++f) fsum[bclH(f)] += std::abs(Fb(f));
      Cfl r;
      for (Index c = 0; c < nc; ++c) {
        const Real v = fsum[c] / (2.0 * volH(c));
        if (v > r.c) { r.c = v; r.cell = c; }
      }
      r.c *= dtTest;
      return r;
    };

    VectorField src("src", nt, 3);
    Real time = 0.0;
    int step = 0, sinceChange = 0, holdUntil = 0;
    bool refused = false, steady = false, quasi = false;
    Cfl cNow;
    // Every 50 steps: the reattachment point and Cf at x = -4, with the step
    // they were taken at, for the quasi-steady test.
    std::vector<int> histStep;
    std::vector<Real> histXr, histCf4;
    Separation sep;
    Real cf4 = 0.0, uc = 0.0;
    for (step = 1; ; ++step) {
      if (step > 1 && (step - 1) % 100 == 0 && dtNow < dt && step > holdUntil) {
        const Real d2 = std::min(2.0 * dtNow, dt);
        const Cfl c2 = courant(d2);
        refused = c2.c > C_UP;
        if (!refused) { dtNow = d2; solver.setTimeStep(dtNow); sinceChange = 0; }
        std::printf("  step %6d  ramp: C %.2f at dt %.3e, at (%.4f, %.3e): %s\n", step, c2.c, d2,
                    ccH(c2.cell,0), ccH(c2.cell,1), refused ? "holds" : "doubles");
      }
      solver.advance(ub, fb, src, momentum, *pressure);
      time += dtNow;
      ++sinceChange;
      cNow = courant(dtNow);
      if (std::isfinite(cNow.c) && cNow.c > C_DOWN) {
        std::printf("  step %6d  ramp: C %.2f at dt %.3e, at (%.4f, %.3e): halves\n", step, cNow.c,
                    dtNow, ccH(cNow.cell,0), ccH(cNow.cell,1));
        dtNow *= 0.5; solver.setTimeStep(dtNow);
        sinceChange = 0; holdUntil = step + 100; refused = false;
      }
      // Steps at the dt the march settles at: the level's, or the one the
      // cap last kept it at.
      const bool atFinal = dtNow == dt || refused;
      const int atDt = atFinal ? sinceChange : 0;
      if (std::find(saveAt.begin(), saveAt.end(), step) != saveAt.end())
        saveState(out + ".state." + std::to_string(step));
      if (std::getenv("VIBEFLOW_BS_DEBUG")) {
        // Exploration only: where the fields go, step by step.
        auto u = host(solver.velocity()); auto k = host(solver.turbulentKineticEnergy());
        auto w = host(solver.specificDissipation()); auto nut = host(solver.eddyViscosity());
        Index iu = 0, ik = 0, iw = 0, in = 0;
        for (Index c = 0; c < nc; ++c) {
          if (std::abs(u(c,0)) + std::abs(u(c,1)) > std::abs(u(iu,0)) + std::abs(u(iu,1))) iu = c;
          if (k(c) < k(ik)) ik = c;
          if (w(c) > w(iw)) iw = c;
          if (nut(c) > nut(in)) in = c;
        }
        std::printf("    dbg %4d  u (%.3e, %.3e) |u| max %.3e at (%.3f, %.3e)  k min %.3e at (%.3f, %.3e)  "
                    "w max %.3e at (%.3f, %.3e)  nut/nu max %.3e at (%.3f, %.3e)  bounded %lld\n",
                    step, u(iu,0), u(iu,1), std::abs(u(iu,0)) + std::abs(u(iu,1)), ccH(iu,0), ccH(iu,1), k(ik),
                    ccH(ik,0), ccH(ik,1), w(iw), ccH(iw,0), ccH(iw,1), nut(in) / NU, ccH(in,0),
                    ccH(in,1), solver.boundedCells());
        std::fflush(stdout);
      }
      if (step % 50 == 0 || atDt >= maxFull || step >= MAX_STEPS) {
        const auto tau = wallStress(mesh, solver, bottom);
        std::vector<Real> cf(tau.size());
        for (std::size_t i = 0; i < tau.size(); ++i) cf[i] = 2.0 * tau[i];
        sep = separation(bottom.x, cf);
        cf4 = interpAt(bottom.x, cf, X_CF4);
        uc = centreU();
        if (!std::isfinite(cf4) || !std::isfinite(uc)) { std::printf("  step %d: not finite\n", step); break; }
        histStep.push_back(step); histXr.push_back(sep.xr); histCf4.push_back(cf4);
        snapshot(uN, kN, wN);
        const Change cu = relChange(uN, uPrev, 3), ck = relChange(kN, kPrev, 1),
                     cw = relChange(wN, wPrev, 1);
        const Real change = std::max({cu.rel, ck.rel, cw.rel});
        uPrev = uN; kPrev = kN; wPrev = wN;
        // Over the last 20% of the march: how far x_r and Cf(-4) moved.
        Real dxr = 0.0, dcf = 0.0;
        const int from = step - step / 5;
        for (std::size_t i = 0; i < histStep.size(); ++i) {
          if (histStep[i] < from) continue;
          dxr = std::max(dxr, std::abs(histXr[i] - sep.xr) / std::max(std::abs(sep.xr), 1e-300));
          dcf = std::max(dcf, std::abs(histCf4[i] - cf4) / std::abs(cf4));
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  step %6d  t %9.3f  dt %.2e  C %.2f  xr %.5f  Cf(-4) %.6e  uc(-4) %.6f  "
                    "bubble %.3f %.3f  drift %.1e %.1e  change/50 steps %.1e  bounded %lld  (%.0f s)\n",
                    step, time, dtNow, cNow.c, sep.xr, cf4, uc, sep.c0, sep.c1, dxr, dcf, change,
                    solver.boundedCells(), secs);
        std::printf("      change: u %.1e at (%.4f, %.3e)  k %.1e at (%.4f, %.3e)  "
                    "omega %.1e at (%.4f, %.3e)\n",
                    cu.rel, ccH(cu.cell,0), ccH(cu.cell,1), ck.rel, ccH(ck.cell,0), ccH(ck.cell,1),
                    cw.rel, ccH(cw.cell,0), ccH(cw.cell,1));
        std::fflush(stdout);
        if (atDt >= 1000 && change < 1e-6) { steady = true; break; }
        if (atDt >= 1000 && sep.xr > 0.0 && dxr <= 2e-3 && dcf <= 2e-3) { quasi = true; break; }
        if (atDt >= maxFull || step >= MAX_STEPS) break;
      }
      if (step == stopAt) break;
    }

    {
      const auto tau = wallStress(mesh, solver, bottom);
      std::FILE* fc = std::fopen((out + ".cf").c_str(), "w");
      std::fprintf(fc, "# x  Cf on the bottom wall (backstep, SST-1994)\n");
      for (std::size_t i = 0; i < tau.size(); ++i)
        std::fprintf(fc, "%.10e %.10e\n", bottom.x[i], 2.0 * tau[i]);
      std::fclose(fc);
      // Cp from the wall cells' pressure, shifted to zero at x = 40.
      auto p = host(solver.pressure()); auto bcell = host(mesh.boundaryCell());
      std::vector<Real> pw(bottom.face.size());
      for (std::size_t i = 0; i < pw.size(); ++i) pw[i] = p(bcell(bottom.face[i]));
      const Real p40 = interpAt(bottom.x, pw, 40.0);
      std::FILE* fp = std::fopen((out + ".cp").c_str(), "w");
      std::fprintf(fp, "# x  Cp on the bottom wall, 0 at x = 40\n");
      for (std::size_t i = 0; i < pw.size(); ++i)
        std::fprintf(fp, "%.10e %.10e\n", bottom.x[i], 2.0 * (pw[i] - p40));
      std::fclose(fp);
      // The profiles at the stations.
      auto u = host(solver.velocity());
      std::FILE* fq = std::fopen((out + ".prof").c_str(), "w");
      std::fprintf(fq, "# x_station  y  u  (the two columns around it, interpolated)\n");
      for (Real xs : STATIONS) {
        const Columns cs = columnsAt(mesh, xs);
        const Real t = (xs - cs.xl) / (cs.xr - cs.xl);
        const std::size_t n = std::min(cs.left.size(), cs.right.size());
        for (std::size_t j = 0; j < n; ++j) {
          const Index a = cs.left[j], b = cs.right[j];
          std::fprintf(fq, "%.4f %.10e %.10e\n", xs, (1.0 - t) * ccH(a,1) + t * ccH(b,1),
                       (1.0 - t) * u(a,0) + t * u(b,0));
        }
      }
      std::fclose(fq);
      if (std::getenv("VIBEFLOW_BS_SAVE")) saveState(out + ".state");
      const PisoTimings& ts = solver.timings();
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("timings: total %.0f s  pressure solve %.0f  momentum solve %.0f  assembly %.0f\n",
                  ts.total, ts.pressureSolve, ts.momentumSolve, ts.assemble + ts.pressureAssembly);
      std::printf("FINAL xr %.6f cf4 %.8e uc4 %.8f bubble %.5f %.5f steps %d full %d dt %.6g time %.3f "
                  "settled %s bounded %lld seconds %.0f\n",
                  sep.xr, cf4, uc, sep.c0, sep.c1, step, (dtNow == dt || refused) ? sinceChange : 0,
                  dtNow, time,
                  steady ? "steady" : quasi ? "quasi" : "no", solver.boundedCells(), secs);
    }
    rc = (steady || quasi) ? 0 : 1;
  }
  Kokkos::finalize();
#ifdef VIBEFLOW_HAVE_PETSC
  PetscFinalize();
#endif
  return rc;
}
