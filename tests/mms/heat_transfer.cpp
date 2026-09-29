// v2a gates 1-3, C++ side (ADR-038): the energy equation and Boussinesq
// buoyancy.
//
// The port of prototype/boussinesq.py, one mesh finer -- 8/16/32 where the
// Python gate runs 6/12/24 -- judged by the same criteria as revised before
// any C++ code (ADR-038, "Revisions before the result"):
//
//   1. Temperature in an exact flow. The Ethier-Steinman velocity carries
//          T = exp(-t) sin(pi x) sin(pi y) sin(pi z)
//      with the source that makes it exact, and no buoyancy. Spatial order in
//      [1.85, 2.15] on the orthogonal family and in [1.6, 2.3], approaching 2,
//      on the smooth distortion (dt = 2e-4, two steps); BDF2 order in time in
//      [1.8, 2.6] against a dt/64 reference.
//   2. Boussinesq, steady, manufactured. The steady solenoidal velocity of
//      steady_dt, p = cos(pi x) cos(pi y) cos(pi z) and
//      T = 1 + sin(pi x) sin(pi y) sin(pi z) / 2, buoyancy on (betaG =
//      (0, 0, -1), T_ref = 0), sources that make all three exact. Orders of u
//      and T in the same bands, and the same steady state at dt = 0.2 and 2.0
//      to 1e-6 of the discretisation error, on the n = 8 skewed fixture mesh
//      -- the Python gate's warped mesh, written out.
//   3. A stratified fluid at rest stays at rest. T = 1 - z between a hot
//      bottom and a cold top, adiabatic sides, the reference stratification
//      equal to it: after 50 steps max|u| <= 1e-12 and max|T - (1 - z)| <=
//      1e-12, on a Cartesian and a distorted mesh. With a constant reference
//      instead, the spurious velocity is reported, not gated.
//
// Two harness settings differ from the Python gate. Both were fixed before
// this program first ran, and ADR-038 records them:
//   * gate 2's order runs march with dt = 0.1, not 0.2. On 32^3 a step of 0.2
//     is a Courant number a third above the one Python's finest mesh (24^3)
//     converged at, and 0.1 keeps it below. The steady state does not depend
//     on dt -- the gate's own third check says so -- only whether the march
//     reaches it does.
//   * "steady" is the largest change of u, p or T over a step below 1e-12, not
//     1e-13: iterative linear solvers have a noise floor that the Python
//     gate's direct factorisations do not. steady_dt made the same choice.
//     REVISED after the first run (ADR-038): on the 16^3 distortion the
//     change stalled at 1.9e-9. That is the non-orthogonal loop's tolerance,
//     measured: the warm-started loop ends a step's sweeps as soon as one
//     moves the correction by less than it, and the state wobbles from step
//     to step at about 190 times it (1e-11 -> 1.9e-9, 1e-13 -> 1.9e-11,
//     1e-14 -> 1.9e-12; 1e-14 costs 55 s a step on 32^3). So the loop
//     converges to 1e-12, the order runs call a state steady below 1e-8 --
//     far below the errors they compare, 1.4e-4 and up -- and the dt check,
//     which compares two states to 1e-6 of a 2e-2 error, keeps 1e-12 on its
//     8^3 mesh, as steady_dt does.
//
// ADR-041 adds two gates for the balanced buoyancy form, and every gate here
// runs in that form (VIBEFLOW_BUOYANCY=cell selects ADR-038's cell force, the
// baseline):
//   4. at rest without a matched reference: gate 3 with T_ref = 0.5;
//   5. at rest in a curved stratification: a uniformly heated layer, T = 1 -
//      z^2 with source 2 kappa, T_ref = 0.5. u stays at zero and T
//      horizontally uniform, to 1e-12.
// The same resting fluid on the randomly perturbed fixture mesh, which has no
// layers, is reported in both forms and not gated.
//
// VIBEFLOW_PH=<PETSc configuration> solves the hydrostatic pressure p_h on
// that backend instead of the native CG (ADR-044's check).
//
// Run:  heat_transfer <fixtures> [gates, default 12345] [grids, default 8 16 32]

#include "mesh/HexMesh.hpp"
#include "discretization/FaceFlux.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#ifdef VIBEFLOW_HAVE_PETSC
#include "linalg/PetscSolver.hpp"
#include <petscsys.h>
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {

constexpr Real PI = 3.14159265358979323846;
constexpr Real A = PI / 4.0;
constexpr Real D = PI / 2.0;

using ScalarFn = std::function<Real(const Vec3&)>;
using VecFn = std::function<Vec3(const Vec3&)>;

// ------------------------------------------------------------ exact fields
Vec3 esVelocity(const Vec3& p, Real t, Real nu) {
  const Real e = std::exp(-D * D * nu * t);
  return {-A * (std::exp(A*p.x) * std::sin(A*p.y + D*p.z)
                + std::exp(A*p.z) * std::cos(A*p.x + D*p.y)) * e,
          -A * (std::exp(A*p.y) * std::sin(A*p.z + D*p.x)
                + std::exp(A*p.x) * std::cos(A*p.y + D*p.z)) * e,
          -A * (std::exp(A*p.z) * std::sin(A*p.x + D*p.y)
                + std::exp(A*p.y) * std::cos(A*p.z + D*p.x)) * e};
}

Real esPressure(const Vec3& p, Real t, Real nu) {
  return -A * A / 2.0 * (
      std::exp(2*A*p.x) + std::exp(2*A*p.y) + std::exp(2*A*p.z)
      + 2*std::sin(A*p.x + D*p.y)*std::cos(A*p.z + D*p.x)*std::exp(A*(p.y + p.z))
      + 2*std::sin(A*p.y + D*p.z)*std::cos(A*p.x + D*p.y)*std::exp(A*(p.z + p.x))
      + 2*std::sin(A*p.z + D*p.x)*std::cos(A*p.y + D*p.z)*std::exp(A*(p.x + p.y)))
      * std::exp(-2.0 * D * D * nu * t);
}

Real tExact(const Vec3& q, Real t) {
  return std::exp(-t) * std::sin(PI*q.x) * std::sin(PI*q.y) * std::sin(PI*q.z);
}

// dT/dt + u.grad T - kappa laplacian T for the manufactured temperature.
Real tSource(const Vec3& q, Real t, Real nu, Real kappa) {
  const Real e = std::exp(-t);
  const Real sx = std::sin(PI*q.x), sy = std::sin(PI*q.y), sz = std::sin(PI*q.z);
  const Real cx = std::cos(PI*q.x), cy = std::cos(PI*q.y), cz = std::cos(PI*q.z);
  const Real T = e * sx * sy * sz;
  const Vec3 gT{e*PI*cx*sy*sz, e*PI*sx*cy*sz, e*PI*sx*sy*cz};
  return -T + esVelocity(q, t, nu).dot(gT) + 3.0 * PI * PI * kappa * T;
}

Vec3 sfVelocity(const Vec3& q) {
  const Real sx = std::sin(PI*q.x), sy = std::sin(PI*q.y), sz = std::sin(PI*q.z);
  const Real cx = std::cos(PI*q.x), cy = std::cos(PI*q.y), cz = std::cos(PI*q.z);
  return {sx * (cy - cz), sy * (cz - cx), sz * (cx - cy)};
}

// g[i][j] = d u_i / d x_j of sfVelocity.
void sfGradient(const Vec3& q, Real g[3][3]) {
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

Real bTemperature(const Vec3& q) {
  return 1.0 + 0.5 * std::sin(PI*q.x) * std::sin(PI*q.y) * std::sin(PI*q.z);
}

Real bPressure(const Vec3& q) {
  return std::cos(PI*q.x) * std::cos(PI*q.y) * std::cos(PI*q.z);
}

// Momentum source of gate 2: (u.grad)u - nu laplacian(u) + grad p - f with
// the buoyancy f = -(T - 0) betaG = (0, 0, T), laplacian(u) = -2 pi^2 u.
Vec3 bMomentumSource(const Vec3& q, Real nu) {
  const Vec3 u = sfVelocity(q);
  const Real uu[3] = {u.x, u.y, u.z};
  Real g[3][3];
  sfGradient(q, g);
  Real s[3];
  for (int i = 0; i < 3; ++i) {
    s[i] = 2.0 * PI * PI * nu * uu[i];
    for (int j = 0; j < 3; ++j) s[i] += uu[j] * g[i][j];
  }
  const Real sx = std::sin(PI*q.x), sy = std::sin(PI*q.y), sz = std::sin(PI*q.z);
  const Real cx = std::cos(PI*q.x), cy = std::cos(PI*q.y), cz = std::cos(PI*q.z);
  s[0] -= PI * sx * cy * cz;
  s[1] -= PI * cx * sy * cz;
  s[2] -= PI * cx * cy * sz;
  s[2] -= bTemperature(q);
  return {s[0], s[1], s[2]};
}

// u.grad T - kappa laplacian T for gate 2's temperature.
Real bTemperatureSource(const Vec3& q, Real kappa) {
  const Real sx = std::sin(PI*q.x), sy = std::sin(PI*q.y), sz = std::sin(PI*q.z);
  const Real cx = std::cos(PI*q.x), cy = std::cos(PI*q.y), cz = std::cos(PI*q.z);
  const Vec3 gT{0.5*PI*cx*sy*sz, 0.5*PI*sx*cy*sz, 0.5*PI*sx*sy*cz};
  return sfVelocity(q).dot(gT) + kappa * 1.5 * PI * PI * sx * sy * sz;
}

// ------------------------------------------------------------------ fields
template <class V> auto host(const V& v) {
  return Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
}

template <class V> Vec3 centre(const V& cc, Index c) { return {cc(c,0), cc(c,1), cc(c,2)}; }

ScalarField cellScalar(const Mesh& m, const ScalarFn& fn) {
  auto cc = host(m.cellCentre());
  ScalarField s("cellScalar", m.nTotal());
  auto h = Kokkos::create_mirror_view(s);
  for (Index c = 0; c < m.nTotal(); ++c) h(c) = fn(centre(cc, c));
  Kokkos::deep_copy(s, h);
  return s;
}

VectorField cellVector(const Mesh& m, const VecFn& fn) {
  auto cc = host(m.cellCentre());
  VectorField v("cellVector", m.nTotal(), 3);
  auto h = Kokkos::create_mirror_view(v);
  for (Index c = 0; c < m.nTotal(); ++c) {
    const Vec3 x = fn(centre(cc, c));
    h(c,0) = x.x; h(c,1) = x.y; h(c,2) = x.z;
  }
  Kokkos::deep_copy(v, h);
  return v;
}

// Face average of a scalar, with the quadrature the velocity uses.
ScalarField faceAverage(const HexMesh& m, const ScalarFn& fn) {
  VectorField v;
  averageBoundaryValue(m, [&](const Vec3& q) { return Vec3{fn(q), 0.0, 0.0}; }, v);
  ScalarField s("faceAverage", m.nBoundaryFaces());
  Kokkos::parallel_for("fa", Kokkos::RangePolicy<ExecSpace>(0, m.nBoundaryFaces()),
    KOKKOS_LAMBDA(const Index f) { s(f) = v(f, 0); });
  Kokkos::fence();
  return s;
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

// Face flux of a cell field, weighted by each face's interpolation weight --
// the initial flux the Python gates build.
ScalarField initialFlux(const Mesh& mesh, const VectorField& u0) {
  ScalarField F0("F0", mesh.nInternalFaces());
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
  return F0;
}

// Volume-weighted RMS over cells of a host array with `ncomp` values per cell.
Real l2(const Mesh& m, const std::vector<Real>& e, int ncomp) {
  auto vol = host(m.cellVolume());
  Real num = 0.0, den = 0.0;
  for (Index c = 0; c < m.nCells(); ++c) {
    Real s = 0.0;
    for (int d = 0; d < ncomp; ++d) s += e[c*ncomp + d] * e[c*ncomp + d];
    num += s * vol(c);
    den += vol(c);
  }
  return std::sqrt(num / den);
}

std::vector<Real> toHost(const ScalarField& s, Index n) {
  auto h = host(s);
  std::vector<Real> out(n);
  for (Index c = 0; c < n; ++c) out[c] = h(c);
  return out;
}

std::vector<Real> toHost(const VectorField& v, Index n) {
  auto h = host(v);
  std::vector<Real> out(n * 3);
  for (Index c = 0; c < n; ++c)
    for (int d = 0; d < 3; ++d) out[c*3 + d] = h(c, d);
  return out;
}

std::vector<Real> orders(const std::vector<Real>& e, const std::vector<Real>& h) {
  std::vector<Real> o;
  for (std::size_t i = 1; i < e.size(); ++i)
    o.push_back(std::log(e[i-1] / e[i]) / std::log(h[i-1] / h[i]));
  return o;
}

// Order in [lo, hi]; with `approaching`, also no further from 2 than the order
// before it, within 0.02 (ADR-038's revision).
bool verdict(const std::string& label, const std::vector<Real>& o, Real lo, Real hi,
             bool approaching) {
  const Real last = o.back();
  bool ok = last >= lo && last <= hi;
  std::printf("  -> %s: order %.3f in [%g, %g]", label.c_str(), last, lo, hi);
  if (approaching && o.size() > 1) {
    bool r = true;
    for (std::size_t i = 1; i < o.size(); ++i)
      r &= std::abs(o[i] - 2.0) <= std::abs(o[i-1] - 2.0) + 0.02;
    ok &= r;
    std::printf(", %s", r ? "approaching 2" : "MOVING AWAY FROM 2");
  }
  std::printf(": %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

HexMesh family(Index n, Real skew) {
  return HexMesh::generate(n, skew, skew == 0.0 ? "none" : "smooth");
}

// The buoyancy form every gate here runs (ADR-041).
BuoyancyForm gateForm() {
  const char* e = std::getenv("VIBEFLOW_BUOYANCY");
  return (e && std::string(e) == "cell") ? BuoyancyForm::Cell : BuoyancyForm::Balanced;
}

// The hydrostatic pressure's solver (ADR-044): the native Jacobi CG, which is
// also every gate's pressure solver here, unless VIBEFLOW_PH names a PETSc
// configuration -- that ADR's check of a p_h on another backend.
void hydrostaticSolver(PisoSolver& solver, const Mesh& mesh) {
  const char* e = std::getenv("VIBEFLOW_PH");
  if (!e || std::string(e) == "native") return;
#ifdef VIBEFLOW_HAVE_PETSC
  auto s = std::make_unique<PetscSolver>(mesh, Comm(), std::string(e));
  s->setConstantNullSpace(true);
  s->setUnpreconditionedNorm(true);
  s->setSymmetricAMG();
  solver.setHydrostaticSolver(std::move(s));
#else
  (void)solver; (void)mesh;
  throw std::runtime_error(std::string("built without PETSc; VIBEFLOW_PH=") + e + " unavailable");
#endif
}

// ------------------------------------------ 1. temperature in an exact flow
std::vector<Real> energyExactFlow(const HexMesh& mesh, Real dt, int nsteps, Real nu,
                                  Real kappa, int outer) {
  const Index nb = mesh.nBoundaryFaces();
  PisoControls ctl;
  ctl.outer = outer;
  ctl.correctors = 2;
  PisoSolver solver(mesh, nu, dt, ctl);
  EnergyModel em;
  em.kappa = kappa;
  em.form = gateForm();
  solver.enableEnergy(em);
  hydrostaticSolver(solver, mesh);

  VectorField u0 = cellVector(mesh, [nu](const Vec3& q) { return esVelocity(q, 0.0, nu); });
  ScalarField p0 = cellScalar(mesh, [nu](const Vec3& q) { return esPressure(q, 0.0, nu); });
  {
    // Plain mean over the cells, as numpy's p0.mean() in the Python gate.
    auto hp = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), p0);
    Real mean = 0.0;
    for (Index c = 0; c < mesh.nCells(); ++c) mean += hp(c);
    mean /= mesh.nCells();
    for (Index c = 0; c < mesh.nTotal(); ++c) hp(c) -= mean;
    Kokkos::deep_copy(p0, hp);
  }
  solver.setState(u0, p0, initialFlux(mesh, u0));
  solver.setTemperature(cellScalar(mesh, [](const Vec3& q) { return tExact(q, 0.0); }));
  View1<int> tType("tType", nb);                 // all FixedValue

  VectorField src("src", mesh.nTotal(), 3);
  NativeBiCGStab momentum(mesh);
  NativeCG pressure(mesh, Comm(), true);
  for (int k = 0; k < nsteps; ++k) {
    const Real t = (k + 1) * dt;
    auto uFn = [t, nu](const Vec3& q) { return esVelocity(q, t, nu); };
    VectorField ub; ScalarField fb;
    averageBoundaryValue(mesh, uFn, ub);
    integrateBoundaryFlux(mesh, uFn, fb);
    adjustBoundaryFlux(mesh, fb);
    solver.setTemperatureBoundary(tType, faceAverage(mesh, [t](const Vec3& q) { return tExact(q, t); }));
    solver.setTemperatureSource(cellScalar(mesh,
        [t, nu, kappa](const Vec3& q) { return tSource(q, t, nu, kappa); }));
    solver.advance(ub, fb, src, momentum, pressure);
  }
  return toHost(solver.temperature(), mesh.nCells());
}

bool gateEnergyExactFlow(const std::vector<Index>& grids) {
  std::printf("\n1. temperature in the exact Ethier-Steinman flow (beta = 0)\n");
  const Real nu = 0.05, kappa = 0.05, dt = 2e-4;
  const int steps = 2;
  bool ok = true;
  struct Fam { Real skew, lo, hi; bool approaching; const char* tag; };
  for (const Fam& fm : {Fam{0.0, 1.85, 2.15, false, "orthogonal"},
                        Fam{0.25, 1.6, 2.3, true, "smooth distortion"}}) {
    std::vector<Real> errs, hs;
    for (Index n : grids) {
      const HexMesh mesh = family(n, fm.skew);
      const auto T = energyExactFlow(mesh, dt, steps, nu, kappa, 6);
      auto cc = host(mesh.cellCentre());
      std::vector<Real> e(mesh.nCells());
      for (Index c = 0; c < mesh.nCells(); ++c) e[c] = T[c] - tExact(centre(cc, c), steps * dt);
      errs.push_back(l2(mesh, e, 1));
      hs.push_back(1.0 / n);
      std::printf("  %-18s n=%-3d L2(T) %.10e\n", fm.tag, n, errs.back());
    }
    if (grids.size() > 1)
      ok &= verdict(std::string("spatial, ") + fm.tag, orders(errs, hs), fm.lo, fm.hi,
                    fm.approaching);
  }
  // BDF2 in time, against dt/64 on the same mesh, as the Python gate.
  const HexMesh mesh = family(8, 0.0);
  const Real tEnd = 0.8;
  const auto ref = energyExactFlow(mesh, tEnd / 64, 64, 1.0, 1.0, 200);
  std::vector<Real> errs, dts;
  for (int s : {2, 4, 8}) {
    const auto T = energyExactFlow(mesh, tEnd / s, s, 1.0, 1.0, 200);
    std::vector<Real> e(mesh.nCells());
    for (Index c = 0; c < mesh.nCells(); ++c) e[c] = T[c] - ref[c];
    errs.push_back(l2(mesh, e, 1));
    dts.push_back(tEnd / s);
    std::printf("  temporal   dt=%-6g L2(T - T_ref) %.6e\n", tEnd / s, errs.back());
  }
  ok &= verdict("temporal, BDF2", orders(errs, dts), 1.8, 2.6, false);
  return ok;
}

// ---------------------------------------------- 2. steady Boussinesq MMS
struct Steady { std::vector<Real> u, p, T; int steps = 0; Real change = 1e300; };

Steady steadyBoussinesq(const HexMesh& mesh, Real dt, Real tol, Real nu = 0.1,
                        Real kappa = 0.1, Real maxTime = 400.0) {
  const Index nc = mesh.nCells(), nb = mesh.nBoundaryFaces();
  PisoControls ctl;
  ctl.outer = 3;
  ctl.correctors = 2;
  ctl.nonOrthTol = 1e-12;         // see the header: the state wobbles at about
  ctl.pressureSolveTol = 1e-13;   // 190 times this from step to step
  // Exploration only, as in ethier_steinman.
  if (const char* e = std::getenv("VIBEFLOW_NONORTH_TOL")) ctl.nonOrthTol = std::atof(e);
  PisoSolver solver(mesh, nu, dt, ctl);
  EnergyModel em;
  em.kappa = kappa;
  em.betaG = {0.0, 0.0, -1.0};
  em.tRef = 0.0;
  em.form = gateForm();
  solver.enableEnergy(em);
  hydrostaticSolver(solver, mesh);

  VectorField src = cellVector(mesh, [nu](const Vec3& q) { return bMomentumSource(q, nu); });
  VectorField ub; ScalarField fb;
  averageBoundaryValue(mesh, sfVelocity, ub);
  integrateBoundaryFlux(mesh, sfVelocity, fb);
  adjustBoundaryFlux(mesh, fb);
  solver.setTemperature(cellScalar(mesh, [](const Vec3&) { return 1.0; }));
  solver.setTemperatureBoundary(View1<int>("tType", nb), faceAverage(mesh, bTemperature));
  solver.setTemperatureSource(cellScalar(mesh, [kappa](const Vec3& q) {
    return bTemperatureSource(q, kappa); }));

  NativeBiCGStab momentum(mesh);
  NativeCG pressure(mesh, Comm(), true);
  const bool verbose = std::getenv("VIBEFLOW_VERBOSE") != nullptr;
  Steady st;
  std::vector<Real> pu, pp, pT;
  const int maxSteps = static_cast<int>(std::lround(maxTime / dt));
  for (int k = 0; k < maxSteps; ++k) {
    solver.advance(ub, fb, src, momentum, pressure);
    st.steps = k + 1;
    auto u = toHost(solver.velocity(), nc);
    auto p = toHost(solver.pressure(), nc);
    auto T = toHost(solver.temperature(), nc);
    if (!pu.empty()) {
      Real ch = 0.0;
      for (std::size_t i = 0; i < u.size(); ++i) ch = std::max(ch, std::abs(u[i] - pu[i]));
      for (Index c = 0; c < nc; ++c) {
        ch = std::max(ch, std::abs(p[c] - pp[c]));
        ch = std::max(ch, std::abs(T[c] - pT[c]));
      }
      st.change = ch;
      pu = u; pp = p; pT = T;
      if (verbose && (k % 10 == 0 || ch < tol))
        std::printf("    step %5d  change %.3e\n", k + 1, ch);
      if (ch < tol) break;
    } else {
      pu = u; pp = p; pT = T;
    }
    if (!std::isfinite(pu[0])) break;
  }
  st.u = pu; st.p = pp; st.T = pT;
  return st;
}

std::pair<Real, Real> steadyErrors(const HexMesh& mesh, const Steady& s) {
  auto cc = host(mesh.cellCentre());
  const Index nc = mesh.nCells();
  std::vector<Real> eu(nc * 3), eT(nc);
  for (Index c = 0; c < nc; ++c) {
    const Vec3 ue = sfVelocity(centre(cc, c));
    eu[c*3] = s.u[c*3] - ue.x; eu[c*3+1] = s.u[c*3+1] - ue.y; eu[c*3+2] = s.u[c*3+2] - ue.z;
    eT[c] = s.T[c] - bTemperature(centre(cc, c));
  }
  return {l2(mesh, eu, 3), l2(mesh, eT, 1)};
}

bool gateBoussinesqMms(const std::vector<Index>& grids, const std::string& fixtures,
                       Real orderDt) {
  std::printf("\n2. steady manufactured Boussinesq flow (beta g = (0, 0, -1))\n");
  Real tolOrder = 1e-8;                        // see the header
  const Real tol = 1e-12;
  // The cross-check compares steady states to 1e-6 and so needs them
  // converged further than the order runs do; on its 6^3 mesh the wobble
  // allows it.
  if (const char* e = std::getenv("VIBEFLOW_STEADY_TOL")) tolOrder = std::atof(e);
  bool ok = true;
  struct Fam { Real skew, lo, hi; bool approaching; const char* tag; };
  for (const Fam& fm : {Fam{0.0, 1.85, 2.15, false, "orthogonal"},
                        Fam{0.25, 1.6, 2.3, true, "smooth distortion"}}) {
    std::vector<Real> eu, eT, hs;
    for (Index n : grids) {
      const HexMesh mesh = family(n, fm.skew);
      const Steady s = steadyBoussinesq(mesh, orderDt, tolOrder);
      const auto e = steadyErrors(mesh, s);
      eu.push_back(e.first); eT.push_back(e.second); hs.push_back(1.0 / n);
      std::printf("  %-18s n=%-3d steps %-4d L2(u) %.10e  L2(T) %.10e  last change %.0e\n",
                  fm.tag, n, s.steps, e.first, e.second, s.change);
      ok &= s.change < tolOrder;
    }
    if (grids.size() > 1) {
      ok &= verdict(std::string("u, ") + fm.tag, orders(eu, hs), fm.lo, fm.hi, fm.approaching);
      ok &= verdict(std::string("T, ") + fm.tag, orders(eT, hs), fm.lo, fm.hi, fm.approaching);
    }
  }
  // The buoyancy coupling must not bring dt back into the steady state.
  const HexMesh mesh = HexMesh::fromVertexFile(8, fixtures + "/vertices_n8_s25.txt");
  const Steady a = steadyBoussinesq(mesh, 0.2, tol), b = steadyBoussinesq(mesh, 2.0, tol);
  const auto eb = steadyErrors(mesh, b);
  const Index nc = mesh.nCells();
  std::vector<Real> du(nc * 3), dT(nc);
  for (std::size_t i = 0; i < du.size(); ++i) du[i] = a.u[i] - b.u[i];
  for (Index c = 0; c < nc; ++c) dT[c] = a.T[c] - b.T[c];
  const Real su = l2(mesh, du, 3) / eb.first, sT = l2(mesh, dT, 1) / eb.second;
  const bool passed = su <= 1e-6 && sT <= 1e-6 && a.change < tol && b.change < tol;
  ok &= passed;
  std::printf("  dt = 0.2 vs 2.0, skewed n=8 (steps %d/%d): spread u %.1e, T %.1e "
              "(bound 1e-6): %s\n", a.steps, b.steps, su, sT, passed ? "PASS" : "FAIL");
  return ok;
}

// ------------------------------------------------- 3-5. fluids at rest
struct Rest { Real du, dT, spread; };

// 50 steps from a resting stratified fluid between a hot bottom and a cold
// top, adiabatic sides, walls everywhere: max|u|, the largest departure of T
// from the profile, and the largest spread of T within a horizontal layer.
Rest restState(const HexMesh& mesh, Real tRef, Vec3 tRefGrad, const ScalarFn& profile,
               Real source, BuoyancyForm form) {
  const Index nc = mesh.nCells(), nb = mesh.nBoundaryFaces();
  PisoControls ctl;
  ctl.outer = 3;
  ctl.correctors = 2;
  PisoSolver solver(mesh, 1.0, 0.01, ctl);
  EnergyModel em;
  em.kappa = 1.0;
  em.betaG = {0.0, 0.0, -1700.0};
  em.tRef = tRef;
  em.tRefGrad = tRefGrad;
  em.form = form;
  solver.enableEnergy(em);
  hydrostaticSolver(solver, mesh);
  solver.setTemperature(cellScalar(mesh, profile));
  solver.setTemperatureSource(cellScalar(mesh, [source](const Vec3&) { return source; }));

  auto ba = host(mesh.boundaryArea());
  auto bcen = host(mesh.boundaryCentre());
  View1<int> tType("tType", nb);
  ScalarField tValue("tValue", nb);
  auto ht = Kokkos::create_mirror_view(tType);
  auto hv = Kokkos::create_mirror_view(tValue);
  for (Index f = 0; f < nb; ++f) {
    const Real mag = std::sqrt(ba(f,0)*ba(f,0) + ba(f,1)*ba(f,1) + ba(f,2)*ba(f,2));
    const bool horizontal = std::abs(ba(f,2)) / mag > 0.5;
    ht(f) = static_cast<int>(horizontal ? TemperatureBC::FixedValue : TemperatureBC::FixedFlux);
    hv(f) = horizontal ? profile({bcen(f,0), bcen(f,1), bcen(f,2)}) : 0.0;
  }
  Kokkos::deep_copy(tType, ht);
  Kokkos::deep_copy(tValue, hv);
  solver.setTemperatureBoundary(tType, tValue);

  VectorField ub("ub", nb, 3), src("src", mesh.nTotal(), 3);
  ScalarField fb("fb", nb);
  NativeBiCGStab momentum(mesh);
  NativeCG pressure(mesh, Comm(), true);
  for (int k = 0; k < 50; ++k) solver.advance(ub, fb, src, momentum, pressure);

  const auto u = toHost(solver.velocity(), nc);
  const auto T = toHost(solver.temperature(), nc);
  auto cc = host(mesh.cellCentre());
  Rest r{0.0, 0.0, 0.0};
  std::vector<std::pair<long long, Real>> layer(nc);
  for (Index c = 0; c < nc; ++c) {
    for (int d = 0; d < 3; ++d) r.du = std::max(r.du, std::abs(u[c*3 + d]));
    r.dT = std::max(r.dT, std::abs(T[c] - profile(centre(cc, c))));
    layer[c] = {std::llround(cc(c, 2) * 1e6), T[c]};
  }
  std::sort(layer.begin(), layer.end());
  for (std::size_t i = 0, j = 0; i < layer.size(); i = j) {
    Real lo = layer[i].second, hi = layer[i].second;
    for (j = i; j < layer.size() && layer[j].first == layer[i].first; ++j) {
      lo = std::min(lo, layer[j].second);
      hi = std::max(hi, layer[j].second);
    }
    r.spread = std::max(r.spread, hi - lo);
  }
  return r;
}

const ScalarFn LINEAR = [](const Vec3& q) { return 1.0 - q.z; };
const ScalarFn CURVED = [](const Vec3& q) { return 1.0 - q.z * q.z; };
const Vec3 MATCHED{0.0, 0.0, -1.0}, NONE{0.0, 0.0, 0.0};

bool gateRestState() {
  std::printf("\n3. a fluid resting in its reference stratification stays at rest\n");
  bool ok = true;
  for (Real skew : {0.0, 0.25}) {
    const Rest r = restState(family(8, skew), 1.0, MATCHED, LINEAR, 0.0, gateForm());
    const bool passed = r.du <= 1e-12 && r.dT <= 1e-12;
    ok &= passed;
    std::printf("  %-10s max|u| %.1e   max|T - (1 - z)| %.1e  %s\n",
                skew == 0.0 ? "Cartesian" : "distorted", r.du, r.dT, passed ? "PASS" : "FAIL");
  }
  return ok;
}

bool gateRestConstantReference(const std::string& fixtures) {
  std::printf("\n4. at rest without a matched reference: T = 1 - z, T_ref = 0.5 (ADR-041)\n");
  bool ok = true;
  for (Real skew : {0.0, 0.25}) {
    const Rest r = restState(family(8, skew), 0.5, NONE, LINEAR, 0.0, gateForm());
    const bool passed = r.du <= 1e-12 && r.dT <= 1e-12;
    ok &= passed;
    std::printf("  %-10s max|u| %.1e   max|T - (1 - z)| %.1e  %s\n",
                skew == 0.0 ? "Cartesian" : "distorted", r.du, r.dT, passed ? "PASS" : "FAIL");
  }
  // Reported, not gated: a randomly perturbed mesh has no layers.
  const HexMesh perturbed = HexMesh::fromVertexFile(8, fixtures + "/vertices_n8_s25.txt");
  for (BuoyancyForm form : {BuoyancyForm::Cell, gateForm()}) {
    const Rest r = restState(perturbed, 0.5, NONE, LINEAR, 0.0, form);
    std::printf("  perturbed  %-9s max|u| %.1e  (reported, not gated)\n",
                form == BuoyancyForm::Cell ? "cell" : "balanced", r.du);
  }
  return ok;
}

bool gateRestCurved() {
  std::printf("\n5. at rest in a curved stratification: T = 1 - z^2, source 2 kappa, "
              "T_ref = 0.5 (ADR-041)\n");
  bool ok = true;
  for (Real skew : {0.0, 0.25}) {
    const Rest r = restState(family(8, skew), 0.5, NONE, CURVED, 2.0, gateForm());
    const bool passed = r.du <= 1e-12 && r.spread <= 1e-12;
    ok &= passed;
    std::printf("  %-10s max|u| %.1e   T spread within a layer %.1e  %s\n",
                skew == 0.0 ? "Cartesian" : "distorted", r.du, r.spread,
                passed ? "PASS" : "FAIL");
  }
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
    const std::string fixtures = argc > 1 ? argv[1] : "tests/fixtures";
    const std::string gates = argc > 2 ? argv[2] : "12345";
    std::vector<Index> grids;
    for (int i = 3; i < argc; ++i) grids.push_back(std::stoi(argv[i]));
    if (grids.empty()) grids = {8, 16, 32};
    // Gate 2's marching step for the order runs (see the header). Exposed
    // only so that the cross-check can reproduce the Python harness exactly.
    Real orderDt = 0.1;
    if (const char* e = std::getenv("VIBEFLOW_BOUSSINESQ_DT")) orderDt = std::atof(e);

    bool ok = true;
    std::printf("buoyancy form: %s\n", gateForm() == BuoyancyForm::Cell ? "cell" : "balanced");
    if (const char* e = std::getenv("VIBEFLOW_PH")) std::printf("p_h solver: %s\n", e);
    auto runGate = [&](char g, const auto& fn) {
      if (gates.find(g) == std::string::npos) return;
      try {
        ok &= fn();
      } catch (const std::exception& e) {
        std::printf("  FAIL: %s\n", e.what());
        ok = false;
      }
    };
    runGate('1', [&] { return gateEnergyExactFlow(grids); });
    runGate('2', [&] { return gateBoussinesqMms(grids, fixtures, orderDt); });
    runGate('3', [&] { return gateRestState(); });
    runGate('4', [&] { return gateRestConstantReference(fixtures); });
    runGate('5', [&] { return gateRestCurved(); });
    std::printf("\nv2a heat-transfer GATE (C++): %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
#ifdef VIBEFLOW_HAVE_PETSC
  PetscFinalize();
#endif
  return rc;
}
