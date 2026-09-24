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
  // Each sweep is a full pressure solve plus a boundary-pressure
  // extrapolation, so this is the single biggest cost knob in the solver.
  //
  // 1e-8, measured rather than assumed (ADR-025). On the Ethier-Steinman
  // order study it moves the L2 error by 3e-7 relative -- four orders below
  // the discretisation error being measured -- and halves the pressure time.
  // The old 1e-12 bought nothing: it drove the residual six orders below the
  // point where the answer stopped changing.
  Real nonOrthTol = 1e-8;
  // Relative tolerance of each pressure linear solve. Every sweep of the loop
  // above pays it, and the sweeps are a fixed-point iteration -- solving an
  // intermediate sweep to fourteen digits is fourteen digits of an answer
  // that the next sweep changes.
  //
  // 1e-10: two orders tighter than nonOrthTol, which is the rule that matters.
  // Looser than the loop's own threshold and the loop starts chasing solver
  // noise (the mistake behind ADR-016); much tighter than it and every sweep
  // buys digits the next sweep discards. Measured: 1e-14 costs 12,511 linear
  // iterations where 1e-10 costs 7,980, for the same drag to four decimals.
  // A case that loosens nonOrthTol should loosen this with it.
  Real pressureSolveTol = 1e-10;
  int outer = 1;             // PIMPLE outer iterations
  Real outerTol = 1e-10;
  bool consistentRhieChow = true;
  // Second-order convection is carried as a deferred correction on the
  // right-hand side, which is explicit. Turning it off leaves first-order
  // upwind: wrong, but unconditionally stable in the convective term. It is
  // here as a diagnostic -- when a run grows without bound, this says in one
  // experiment whether the explicit correction is what is growing.
  bool deferredCorrection = true;
  // The non-orthogonal part of the momentum DIFFUSION flux, also carried
  // explicitly on the right-hand side -- and, unlike the pressure equation's
  // non-orthogonal loop, not iterated at all. Its stability limit scales like
  // nu*dt/h^2, not like the Courant number, which is why refining a mesh can
  // destabilise a run whose Courant number looks unremarkable. A diagnostic
  // switch, same as above: off means the diffusion operator keeps only its
  // orthogonal part.
  bool diffusionNonOrth = true;
  // Extrapolate the wall pressure from the interior gradient (ADR: assuming
  // zero normal gradient makes the near-wall cells first order). The
  // extrapolation is a fixed-point iteration whose least-squares weights go
  // like 1/d^2, so as the near-wall cell shrinks the boundary value comes to
  // dominate its own gradient. Switchable to test exactly that.
  bool pressureExtrapolation = true;
  int  pressureExtrapSweeps = 3;
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
  // Convective Courant number, max over cells: 0.5 dt sum|F_f| / V. The
  // deferred correction that carries the scheme's second order in convection
  // is EXPLICIT, so the solver has a time-step limit despite being implicit
  // elsewhere, and the limit tightens with the smallest cell rather than the
  // average one. Refining a mesh at fixed dt is how you find this: the
  // cylinder wake ran happily at Courant 1.7 and diverged inside three steps
  // at 3.3 on a finer mesh of better quality.
  Real courant{};
  // Where the fastest cell is, and how fast. A growing norm says a run is
  // going wrong; it does not say whether the trouble starts at the wall, in
  // the wake or at an outlet, and those have nothing in common to fix.
  Real uMax{};
  Real uMaxAt[3]{};
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
  Real courant() const;

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
  // The non-orthogonal correction, kept ACROSS calls. It is the fixed point of
  // a deferred-correction loop whose answer moves only a little from one
  // pressure solve to the next, so starting from the last one costs nothing
  // and saves nearly all the sweeps. Starting from zero each time re-derived
  // the same field from scratch a hundred times a step, and every sweep is a
  // full pressure solve plus a boundary-pressure extrapolation.
  ScalarField nonorth_;
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
