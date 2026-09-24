// Verification of the OPEN-domain path: an inlet with prescribed flux and an
// outlet where the pressure is prescribed and the mass flux is solved for.
//
// Every earlier gate runs a closed box. A closed box exercises none of this:
// the pressure operator is singular, every boundary flux is an input, and the
// discrete continuity equation is satisfied by construction because the
// boundary contribution to the right-hand side and the boundary contribution
// to the corrected flux are literally the same array. Open it up and those
// become two different arrays, which is exactly where the defect lived: the
// solver ran, produced a plausible wake, and reported a continuity residual
// equal to the entire outlet mass flow from the first step to the last.
//
// Three checks, in increasing order of what they permit:
//
//   A. Uniform flow through a box is an exact DISCRETE fixed point, not just
//      an exact solution of the differential equations. Convection of a
//      constant field carries a factor of the cell's net flux, which is zero;
//      diffusion of a constant field is zero on every face, including the
//      Dirichlet ones where the boundary value equals the cell value; the
//      least-squares gradient of a constant is identically zero; and the
//      Rhie-Chow flux collapses to u.S because both pressure terms vanish.
//      So starting AT the uniform field nothing should move, on a distorted
//      mesh as well as a Cartesian one, and the gate can demand round-off.
//
//      Starting from rest instead would measure a transient, not the scheme.
//      The flow needs a convective time to wash through, and how close it has
//      got by step four says nothing about whether the outlet conserves mass.
//
//   B. The same flow with the outlet pressure prescribed at 7 instead of 0.
//      Incompressible flow only feels pressure differences, so the velocity
//      must be bit-comparable to case A and the pressure must be exactly 7
//      everywhere. This is the only check that notices whether the prescribed
//      value reaches the right-hand side at all: with p_out = 0 a solver that
//      silently drops the term gives the right answer for the wrong reason.
//
//   C. A non-uniform inflow profile, run as a transient on the distorted
//      mesh. There is no exact solution here, so the gate checks the property
//      that does not need one: the discrete divergence must vanish in every
//      cell, and what flows in must flow out. Conservation is a theorem about
//      the scheme, not about the flow, so it holds every step from the first.
//
// A and B are steady and take one step. C is where a wrong outlet treatment
// shows up as a drift that no norm of the velocity would reveal.

#include "mesh/HexMesh.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#ifdef NSFLOW_HAVE_PETSC
#include "linalg/PetscSolver.hpp"
#include <petscsys.h>
#endif
#include <cmath>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace nsflow;

namespace {

constexpr Real U_IN = 1.0;

// Same backend selection the benchmarks use. The native CG is correct here
// but spends thousands of iterations per non-orthogonality sweep, which turns
// a gate that should take seconds into one nobody runs.
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

struct Result {
  Real uErr, pErr, divMax, imbalance, inflow, pbErr;
  // Where the drift lives. "How much" alone does not say which boundary
  // treatment is wrong, and on this case every candidate defect shows up as
  // the same number in a global norm.
  Real dInlet, dOutlet, dSide, dInterior;
  std::vector<Real> u, p;        // kept so two runs can be differenced
};

// Inflow profile. Uniform for cases A and B, a smooth hump for case C. It
// vanishes at the edges of the inlet so the prescribed flux is compatible
// with the no-through-flow side walls at the corner faces.
Real profile(Real y, Real z, bool uniform) {
  if (uniform) return U_IN;
  return U_IN * (0.5 + 1.5 * std::sin(M_PI * y) * std::sin(M_PI * y));
  (void)z;
}

Result run(Index n, Real skew, Real pOut, bool uniform, int steps, Real dt) {
  HexMesh mesh = HexMesh::generate(n, skew, skew > 0.0 ? "smooth" : "none");
  const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();
  const Real nu = 0.05;

  auto side = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                  mesh.boundarySide());
  auto bcen = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                  mesh.boundaryCentre());
  auto barea = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                   mesh.boundaryArea());

  View1<int> uType("uType", nb), pType("pType", nb);
  VectorField ub("ub", nb, 3);
  ScalarField fb("fb", nb), pval("pval", nb);
  auto hUT = Kokkos::create_mirror_view(uType);
  auto hPT = Kokkos::create_mirror_view(pType);
  auto hUB = Kokkos::create_mirror_view(ub);
  auto hFB = Kokkos::create_mirror_view(fb);
  auto hPV = Kokkos::create_mirror_view(pval);

  Real inflow = 0.0;
  for (Index f = 0; f < nb; ++f) {
    hUB(f, 0) = hUB(f, 1) = hUB(f, 2) = 0.0;
    hFB(f) = 0.0;
    hPV(f) = pOut;
    if (side(f) == 0) {                      // x- : inlet, flux prescribed
      const Real u = profile(bcen(f, 1), bcen(f, 2), uniform);
      hUT(f) = static_cast<int>(VelocityBC::Dirichlet);
      hPT(f) = static_cast<int>(PressureBC::FixedFlux);
      hUB(f, 0) = u;
      hFB(f) = u * barea(f, 0);              // area points OUT, so this is < 0
      inflow += -hFB(f);
    } else if (side(f) == 1) {               // x+ : outlet, pressure prescribed
      hUT(f) = static_cast<int>(VelocityBC::ZeroGradient);
      hPT(f) = static_cast<int>(PressureBC::FixedValue);
    } else {                                 // slip walls, no through-flow
      hUT(f) = static_cast<int>(VelocityBC::ZeroGradient);
      hPT(f) = static_cast<int>(PressureBC::FixedFlux);
    }
  }
  Kokkos::deep_copy(uType, hUT); Kokkos::deep_copy(pType, hPT);
  Kokkos::deep_copy(ub, hUB); Kokkos::deep_copy(fb, hFB);
  Kokkos::deep_copy(pval, hPV);

  PisoControls ctl;
  ctl.outer = 3;
  ctl.outerTol = 1e-12;
  // TIGHTER than the shipped defaults, deliberately. The checks below assert
  // exactness to round-off, and a check tighter than the linear solver's own
  // convergence tolerance measures that solver's noise floor rather than the
  // property it names. At the production defaults (ADR-025) the invariance
  // check lands at 2.8e-10, which is the solver converging as promised, not
  // the scheme losing the property -- confirmed by running both.
  //
  // A gate exists to measure the discretisation, so it runs the solver where
  // the discretisation is what is left. NSFLOW_NONORTH_TOL and
  // NSFLOW_PSOLVE_TOL override these to check the shipped defaults instead.
  ctl.nonOrthTol = 1e-12;
  ctl.pressureSolveTol = 1e-14;
  if (const char* e = std::getenv("NSFLOW_NONORTH_TOL")) ctl.nonOrthTol = std::atof(e);
  if (const char* e = std::getenv("NSFLOW_PSOLVE_TOL")) ctl.pressureSolveTol = std::atof(e);
  PisoSolver solver(mesh, nu, dt, ctl);
  solver.setBoundaryTypes(uType);
  solver.setPressureBoundary(pType, pval);

  VectorField u0("u0", nt, 3);
  ScalarField p0("p0", nt), F0("F0", mesh.nInternalFaces());
  {
    // Every case starts with the pressure at the outlet level. Starting the
    // p_out = 7 run at p = 0 would make it a different initial-value problem
    // from the p_out = 0 run -- the first step would see a jump of 7 across
    // the outlet -- and the invariance check would be comparing two things
    // that were never meant to agree.
    const Real pv0 = pOut;
    Kokkos::parallel_for("p0", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { p0(c) = pv0; });
    Kokkos::fence();
  }
  if (uniform) {
    // Start exactly at the fixed point: u = (1,0,0), p = p_out, and the face
    // fluxes that field produces. Any drift from here belongs to the scheme.
    auto fa = mesh.faceArea();
    Kokkos::parallel_for("u0", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const Index c) { u0(c, 0) = U_IN; });
    Kokkos::parallel_for("F0", Kokkos::RangePolicy<ExecSpace>(0, mesh.nInternalFaces()),
      KOKKOS_LAMBDA(const Index f) { F0(f) = U_IN * fa(f, 0); });
    Kokkos::fence();
  }
  // Otherwise start from rest: the profiled case has no exact solution to
  // start at, and conservation is a property of every step regardless.
  solver.setState(u0, p0, F0);

  VectorField src("src", nt, 3);
  NativeBiCGStab momentum(mesh);
  auto pressure = makePressureSolver(mesh);

  Real divMax = 0.0;
  for (int k = 0; k < steps; ++k) {
    const auto rep = solver.advance(ub, fb, src, momentum, *pressure);
    divMax = std::max(divMax, rep.continuityError);
  }

  // Global mass balance: every boundary flux summed. The interior cancels
  // identically, so this is the only place a scheme can leak.
  Real imbalance = 0.0;
  {
    auto Fb = solver.boundaryFlux();
    Kokkos::parallel_reduce("imb", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f, Real& a) { a += Fb(f); }, imbalance);
  }

  // The pressure the solver itself reports on the outlet faces. The pressure
  // equation imposes p = p_out weakly, through the flux term; the gradient
  // operator and the force integral read this array instead. If the two
  // disagree the solver is applying two different outlet conditions.
  Real pbErr = 0.0;
  {
    auto pb = solver.boundaryPressure();
    auto pt = pType;
    const Real target = pOut;
    Kokkos::parallel_reduce("pberr", Kokkos::RangePolicy<ExecSpace>(0, nb),
      KOKKOS_LAMBDA(const Index f, Real& a) {
        if (pt(f) == static_cast<int>(PressureBC::FixedValue))
          a = Kokkos::max(a, Kokkos::abs(pb(f) - target));
      }, Kokkos::Max<Real>(pbErr));
  }

  std::vector<Real> uOut(static_cast<std::size_t>(nc) * 3), pOutF(nc);
  {
    auto hu = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                  solver.velocity());
    auto hp = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                  solver.pressure());
    for (Index c = 0; c < nc; ++c) {
      for (int d = 0; d < 3; ++d) uOut[static_cast<std::size_t>(c) * 3 + d] = hu(c, d);
      pOutF[c] = hp(c);
    }
  }

  // Drift broken down by which boundary the cell touches.
  Real dIn = 0.0, dOut = 0.0, dSide = 0.0, dInt = 0.0;
  if (uniform) {
    std::vector<int> touch(nc, 0);          // bit 1 inlet, 2 outlet, 4 side
    auto hbc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                   mesh.boundaryCell());
    for (Index f = 0; f < nb; ++f) {
      const Index c = hbc(f);
      if (c < nc) touch[c] |= (side(f) == 0) ? 1 : (side(f) == 1) ? 2 : 4;
    }
    auto hu = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                  solver.velocity());
    for (Index c = 0; c < nc; ++c) {
      Real e = std::abs(hu(c, 0) - U_IN);
      e = std::max(e, std::abs(hu(c, 1)));
      e = std::max(e, std::abs(hu(c, 2)));
      if (touch[c] & 1) dIn = std::max(dIn, e);
      if (touch[c] & 2) dOut = std::max(dOut, e);
      if (touch[c] & 4) dSide = std::max(dSide, e);
      if (touch[c] == 0) dInt = std::max(dInt, e);
    }
  }

  Real uErr = 0.0, pErr = 0.0;
  if (uniform) {
    auto u = solver.velocity(); auto p = solver.pressure();
    Kokkos::parallel_reduce("uerr", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& a) {
        a = Kokkos::max(a, Kokkos::abs(u(c, 0) - U_IN));
        a = Kokkos::max(a, Kokkos::abs(u(c, 1)));
        a = Kokkos::max(a, Kokkos::abs(u(c, 2)));
      }, Kokkos::Max<Real>(uErr));
    Kokkos::parallel_reduce("perr", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& a) {
        a = Kokkos::max(a, Kokkos::abs(p(c) - pOut));
      }, Kokkos::Max<Real>(pErr));
  }
  return {uErr, pErr, divMax, std::abs(imbalance), inflow, pbErr,
          dIn, dOut, dSide, dInt, std::move(uOut), std::move(pOutF)};
}

bool check(const char* what, Real value, Real tol) {
  const bool ok = value <= tol && !std::isnan(value);
  std::printf("    %-34s %9.2e  (tol %7.1e)  %s\n", what, value, tol,
              ok ? "ok" : "FAIL");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef NSFLOW_HAVE_PETSC
  PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif
  Kokkos::initialize(argc, argv);
  bool ok = true;
  {
    const Index n = (argc > 1) ? std::atoi(argv[1]) : 16;
    std::printf("open-domain gate, %d^3 cells\n", static_cast<int>(n));

    for (const Real skew : {0.0, 0.3}) {
      const char* kind = skew > 0.0 ? "distorted" : "Cartesian";

      // A. Uniform flow, outlet pressure 0. Exact to machine precision.
      std::printf("  A  uniform flow is a fixed point  [%s]\n", kind);
      Result a = run(n, skew, 0.0, true, 3, 0.05);
      ok &= check("max |u - (1,0,0)|", a.uErr, 1e-10);
      ok &= check("max |p - 0|", a.pErr, 1e-10);
      std::printf("       drift by location: inlet %.2e  outlet %.2e  "
                  "side %.2e  interior %.2e\n",
                  a.dInlet, a.dOutlet, a.dSide, a.dInterior);
      ok &= check("max cell |div|", a.divMax, 1e-11);
      ok &= check("global mass imbalance", a.imbalance, 1e-11 * a.inflow);

      // B. Same flow, outlet pressure 7. Only the level moves.
      std::printf("  B  same, with p_out = 7           [%s]\n", kind);
      Result b = run(n, skew, 7.0, true, 3, 0.05);
      ok &= check("max |u - (1,0,0)|", b.uErr, 1e-10);
      ok &= check("max |p - 7|", b.pErr, 1e-10);
      ok &= check("max cell |div|", b.divMax, 1e-11);
      ok &= check("global mass imbalance", b.imbalance, 1e-11 * b.inflow);

      // C. Non-uniform inflow, transient. Conservation without an exact
      //    solution to lean on.
      std::printf("  C  profiled inflow, transient     [%s]\n", kind);
      Result c = run(n, skew, 0.0, false, 8, 0.01);
      ok &= check("max cell |div| over all steps", c.divMax, 1e-11);
      ok &= check("global mass imbalance", c.imbalance, 1e-11 * c.inflow);
      // D. Under a real pressure gradient the outlet face pressure must still
      //    be the prescribed one. A and B cannot see this: their pressure is
      //    constant, so an extrapolation lands on the right answer by luck.
      ok &= check("outlet face p vs prescribed", c.pbErr, 1e-11);

      // E. Incompressible flow feels pressure differences only. Raising the
      //    outlet level by 7 must move the whole pressure field by 7 and
      //    leave the velocity untouched -- to round-off, not to plotting
      //    accuracy.
      std::printf("  E  outlet level invariance        [%s]\n", kind);
      Result e = run(n, skew, 7.0, false, 8, 0.01);
      Real du = 0.0, dp = 0.0;
      for (std::size_t i = 0; i < c.u.size(); ++i)
        du = std::max(du, std::abs(e.u[i] - c.u[i]));
      for (std::size_t i = 0; i < c.p.size(); ++i)
        dp = std::max(dp, std::abs((e.p[i] - c.p[i]) - 7.0));
      ok &= check("velocity unchanged by p_out", du, 1e-11);
      ok &= check("pressure shifted by exactly 7", dp, 1e-11);
    }
    std::printf("\nopen-domain GATE: %s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
#ifdef NSFLOW_HAVE_PETSC
  PetscFinalize();
#endif
  return ok ? 0 : 1;
}
