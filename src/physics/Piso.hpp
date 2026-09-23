#pragma once
// Incompressible Navier-Stokes: SIMPLE/PISO with Rhie-Chow, BDF2 in time.
//
//     du/dt + div(u u) = -grad(p) + nu laplacian(u) + s,   div(u) = 0
//
// Collocated variables (ADR-010). The face mass flux comes from Rhie-Chow
// interpolation, not from interpolating u, because interpolating u leaves the
// pressure decoupled on the odd-even mode.
//
// Three things here are easy to get subtly wrong, and the Python reference
// got all three wrong first. Each one still produced a running solver and a
// plausible field:
//
//  1. The pressure equation is A p = -div(F*). Solving A p = +div doubles the
//     divergence instead of removing it.
//  2. The second PISO corrector only does something if H is re-evaluated from
//     the updated velocity. Writing HbyA = u + grad(p) V/aP right after
//     u = HbyA - grad(p) V/aP is an identity.
//  3. The pressure boundary value is NOT zero-gradient. The wall normal
//     pressure gradient is whatever the momentum equation requires; assuming
//     zero makes the near-wall cells first order and the global order 1.5.

#include "core/Parallel.hpp"
#include "core/Types.hpp"
#include "discretization/Diffusion.hpp"
#include "discretization/Gradient.hpp"

namespace nsflow {

class Mesh;
class LinearSolver;

// Velocity boundary condition per face.
//   Dirichlet    - the face value is prescribed (walls, inlets, a moving lid)
//   ZeroGradient - the face value follows the cell (slip planes, outlets)
// A zero-gradient face contributes NOTHING to the diffusive matrix or source:
// treating it as Dirichlet with the cell's own value would leave the diagonal
// term in place and quietly over-damp the near-boundary cells.
enum class VelocityBC : int { Dirichlet = 0, ZeroGradient = 1 };

// Pressure boundary condition per face.
//   FixedFlux  - the mass flux through the face is prescribed, so the pressure
//                takes whatever normal gradient satisfies it. Walls, inlets,
//                slip planes. This is the only kind a closed domain has, and
//                it leaves the pressure operator singular.
//   FixedValue - the pressure is prescribed and the flux is part of the
//                solution. An outlet. One such face makes the operator
//                non-singular, which is why the null-space projection is
//                switched off when any is present.
enum class PressureBC : int { FixedFlux = 0, FixedValue = 1 };

struct PisoControls {
  int correctors = 2;        // PISO pressure correctors
  int nonOrthCorrectors = 40;  // iterated to convergence, not a fixed count
  // Convergence threshold for that loop, relative to the largest face flux.
  // Each sweep is a full pressure solve, so this is the single biggest cost
  // knob in the solver, and the default is deliberately far tighter than the
  // discretisation error: a case may loosen it, but only against a measured
  // comparison with this value.
  Real nonOrthTol = 1e-12;
  int outer = 1;             // PIMPLE outer iterations
  Real outerTol = 1e-10;
  bool consistentRhieChow = true;
};

// Coarse phase timings. ADR-016 was written because a sweep count was
// mistaken for a profile; the same mistake was available again here, where
// the two linear solvers together account for barely half the wall time.
struct PisoTimings {
  Real assemble{}, gradient{}, boundaryP{}, hbya{}, rhieChow{},
       pressureAssembly{}, momentumSolve{}, pressureSolve{}, total{};
  int  gradCalls{}, boundaryPCalls{};
};

struct StepReport {
  Real continuityError{};
  int outerUsed{};
  int nonOrthSweeps{};
};

class PisoSolver {
 public:
  PisoSolver(const Mesh& mesh, Real nu, Real dt, PisoControls controls = {},
             Comm comm = Comm());

  // bcType is one VelocityBC per boundary face; empty means all Dirichlet.
  void setBoundaryTypes(const View1<int>& bcType) { bcType_ = bcType; }

  // pType is one PressureBC per boundary face, pValue the prescribed pressure
  // on the FixedValue ones. Calling this switches the solver from the closed
  // -domain path to the open one.
  void setPressureBoundary(const View1<int>& pType, const ScalarField& pValue);

  // Boundary mass fluxes as the solver last computed them. On FixedFlux faces
  // these are what the caller supplied; on FixedValue faces they are solved
  // for, which is the whole point of an outlet.
  ScalarField boundaryFlux() const { return Fb_; }

  // Extrapolated boundary pressure, as the solver itself uses it.
  ScalarField boundaryPressure() const { return pressureBoundary(p_); }

  // Net force the fluid exerts on the faces where mask is non-zero, computed
  // from the SAME discrete operators the momentum equation uses. Recomputing
  // the wall stress with an independent formula would measure a different
  // equation than the one being solved.
  Vec3 boundaryForce(const View1<int>& mask) const;

  StepReport advance(const VectorField& uBoundary, const ScalarField& fBoundary,
                     const VectorField& source, LinearSolver& momentumSolver,
                     LinearSolver& pressureSolver);

  VectorField velocity() const { return u_; }
  ScalarField pressure() const { return p_; }
  ScalarField faceFlux() const { return F_; }
  void setState(const VectorField& u, const ScalarField& p, const ScalarField& F);

  Real continuityError(const ScalarField& F, const ScalarField& Fb) const;

  const PisoTimings& timings() const { return t_; }
  void resetTimings() { t_ = PisoTimings{}; }

 private:
  void bdf(Real& aP, Real& a1, Real& a2) const;
  void assembleMomentum(const VectorField& uB, const VectorField& src);
  void computeHbyA();
  void rhieChow();
  void solvePressure(LinearSolver& solver);
  ScalarField pressureBoundary(const ScalarField& p) const;
  // Ghost cells of a cell field, refreshed from the rank that owns them.
  // No-ops in serial. Every field read at nei(f) or at a ghost index must
  // pass through one of these first; the list of such fields is short and is
  // enumerated at each call site below, because a missing exchange does not
  // crash -- it converges to a different answer.
  void sync(const ScalarField& f) const;
  void sync(const VectorField& f) const;
  void gradP(const ScalarField& p, VectorField& g) const;

  const Mesh& m_;
  Real nu_, dt_;
  PisoControls ctl_;
  Comm comm_;
  Index step_{0};

  DiffusionOperator diff_, pdiff_;
  LeastSquaresGradient grad_;

  ScalarField w_, aP_, Df_, Fstar_;
  VectorField skew_;

  VectorField u_, uOld_, uOld2_, HbyA_, gp_;
  VectorField gH0_, gH1_, gH2_;   // gradients of H/aP, for the
                                  // skewness correction on its face value
  ScalarField p_, F_, FOld_, Fb_;
  VectorField bSrc_;           // pressure-free momentum right-hand sides
  ScalarField diag_, upper_, lower_;
  View1<int> bcType_;
  VectorField uBnd_;           // last prescribed boundary velocity, so the
                               // force integral can rebuild the same gradient
                               // field the momentum equation used
  View1<int> pType_;
  ScalarField pValue_, FbStar_;
  bool openDomain_{false};
  mutable int lastNonOrth_{0};
  mutable PisoTimings t_;
};

}  // namespace nsflow
