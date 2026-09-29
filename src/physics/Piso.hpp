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
#include <memory>
#include <vector>
#include "discretization/Diffusion.hpp"
#include "discretization/Gradient.hpp"

namespace vibeflow {

class Mesh;
class LinearSolver;
class LinearSystem;

// Velocity boundary condition per face.
//   Dirichlet    - the face value is prescribed (walls, inlets, a moving lid)
//   ZeroGradient - the face value follows the cell (outlets, and the two faces
//                  of a one-cell-thick slab, where nothing drives the normal
//                  component anyway)
//   Slip         - zero normal velocity and zero tangential stress: a
//                  symmetry plane, or a wall a roll may slide along (ADR-039).
//                  Treated as a Dirichlet face whose value is the tangential
//                  part of the adjacent cell's velocity, taken from the latest
//                  iterate, so the implicit diagonal balances the lagged value
//                  and at convergence the viscous flux through the face acts
//                  on the normal component alone. The face carries no mass:
//                  the caller passes zero boundary flux there.
// A zero-gradient face contributes NOTHING to the diffusive matrix or source:
// treating it as Dirichlet with the cell's own value would leave the diagonal
// term in place and quietly over-damp the near-boundary cells.
enum class VelocityBC : int { Dirichlet = 0, ZeroGradient = 1, Slip = 2 };

// Temperature boundary condition per face (ADR-038).
//   FixedValue - the face temperature is prescribed.
//   FixedFlux  - the diffusive heat flux INTO the domain per unit area,
//                kappa dT/dn with n pointing inwards, is prescribed; zero is
//                an adiabatic wall. Outflow through such a face carries the
//                cell's own temperature, implicitly.
enum class TemperatureBC : int { FixedValue = 0, FixedFlux = 1 };

// The energy equation and Boussinesq buoyancy (ADR-038):
//
//     dT/dt + div(u T) = div(kappa grad T) + S_T
//     f = -(T - T_ref(x)) betaG,    T_ref(x) = tRef + tRefGrad . x
//
// betaG is the expansion coefficient times the gravity vector; f is a force
// per unit mass added to the momentum source. T_ref is a reference
// stratification: its buoyancy is a gradient and is absorbed into the
// pressure analytically, so a fluid resting in exactly that stratification
// is an exact discrete fixed point on any mesh. A constant T_ref (tRefGrad
// zero) is the usual choice when no such state exists.
// How the buoyancy enters the momentum equation (ADR-041).
//   Cell     - ADR-038's cell force in the momentum source. A resting fluid in
//              a stratification T_ref does not match drifts: its hydrostatic
//              pressure is quadratic, and the least-squares gradient and the
//              boundary extrapolation that carry it are exact only for linear
//              fields (0.47 kappa/L within 50 steps at Ra = 1700 on 8^3).
//   Balanced - the force enters through the faces. Each outer iteration a
//              hydrostatic pressure p_h absorbs the gradient part of the face
//              force B_f = f(T_f).S_f: the residual r_f = B_f - [a_f dp_h +
//              k_f . grad p_h] is made divergence-free, with r = 0 on every
//              boundary face. The cell force is reconstructed from r alone,
//              g_P = M_P^-1 sum S_f r_f / |S_f|, and the predicted face flux
//              takes the compact r_f in place of the interpolated g. On a mesh
//              whose cells stand in layers normal to g, a resting fluid in any
//              T(z) has r = 0 exactly. The solved pressure is then the
//              dynamic part; p_h completes it.
enum class BuoyancyForm : int { Cell = 0, Balanced = 1 };

struct EnergyModel {
  Real kappa = 0.0;
  Vec3 betaG{0.0, 0.0, 0.0};
  Real tRef = 0.0;
  Vec3 tRefGrad{0.0, 0.0, 0.0};
  // Balanced by default since ADR-041's gates passed; Cell stays selectable
  // as the recorded baseline.
  BuoyancyForm form = BuoyancyForm::Balanced;
};

// The k-omega SST model, integrated to the wall (ADR-042):
//
//     dk/dt + div(u k) = P~ - beta* w k + div[(nu + sigma_k nu_t) grad k]
//     dw/dt + div(u w) = gamma P~(or P)/nu_t - beta w^2 + div[(nu + sigma_w nu_t) grad w]
//                        + 2 (1 - F1) sigma_w2 grad k . grad w / w
//
// with P = nu_t S^2 (the flow is incompressible, so that is exact) and the
// isotropic (2/3)k carried by the pressure. Two variants:
//   Menter2003 - nu_t = a1 k / max(a1 w, S F2); P~ = min(P, 10 beta* w k) in
//                both equations; CD_kw floor 1e-10; gamma_1, gamma_2 = 5/9,
//                0.44. The default.
//   Menter1994 - nu_t = a1 k / max(a1 w, Omega F2); P~ = min(P, 20 beta* w k)
//                in the k equation only; CD_kw floor 1e-20; gamma_i =
//                beta_i/beta* - sigma_wi kappa^2/sqrt(beta*). The variant the
//                NASA TMR benchmarks were run in.
enum class SstVariant : int { Menter2003 = 0, Menter1994 = 1 };

// k and omega boundary condition per face.
//   Dirichlet    - both prescribed (an inlet, a manufactured value)
//   ZeroGradient - an outlet, a slip face, a symmetry plane
//   Wall         - k = 0, omega = 10 * 6 nu / (beta_1 d_1^2), d_1 the wall
//                  distance of the adjacent cell centre (Menter 1994)
// The wall distance is measured from the faces flagged as walls when the
// model is enabled, which need not be the Wall faces: a manufactured solution
// measures d from y = 0 and prescribes the exact values there.
enum class TurbulenceBC : int { Dirichlet = 0, ZeroGradient = 1, Wall = 2 };

struct TurbulenceModel {
  SstVariant variant = SstVariant::Menter2003;
  // Advection of k and omega. First-order upwind by default, as TMR's CFL3D
  // and FUN3D run it: omega jumps by five orders of magnitude between the
  // free stream and the first cell on a wall, at a leading edge from one cell
  // to the next, and central differencing across that jump drives the cell
  // upstream of it negative (ADR-042's revision). true: upwind plus the
  // deferred correction to the skew-corrected face value, second order, as
  // the temperature -- what the manufactured gates verify.
  bool secondOrderAdvection = false;
  // After each solve k and omega are bounded below by these; the solver
  // counts the cells it touched (boundedCells), which the manufactured gates
  // require to stay zero.
  Real kFloor = 1e-20;
  Real wFloor = 1e-20;
  // Diagnostics, as PisoControls' switches are: hold k, omega and nu_t where
  // they are (no turbulence solve), or leave out the explicit transpose part
  // of the eddy-viscosity stress. Both change the answer; neither is a model.
  bool frozen = false;
  bool transposeStress = true;
};

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

// How the predicted face flux is built from H/aP.
//   Standard     - F* = (H/aP)_f.S, as in OpenFOAM's PISO. The only pressure
//                  term in the flux is the compact face gradient that the
//                  pressure solve applies. THE DEFAULT.
//   Interpolated - F* = (H/aP)_f.S + D (grad(p)_f.S - snGrad p_old). Both the
//                  Python reference and this solver were first written this
//                  way, and it is wrong for a solver that re-solves the FULL
//                  pressure on every corrector. The checkerboard part of the
//                  pressure equation becomes p_new = -p_old + forcing: an
//                  eigenvalue near -1, so a decoupled pressure-velocity mode
//                  flips sign on every solve, and grows once the velocity
//                  coupling pushes the magnitude past one (ADR-026). Every
//                  outer iteration adds two more flips, which is why iterating
//                  harder made the fine cylinder diverge sooner.
//                  Second order like the standard form -- the MMS gates could
//                  not tell them apart -- which is how it survived.
// Kept only so the defect can be demonstrated on demand.
enum class RhieChowForm : int { Interpolated = 0, Standard = 1 };

// How the old-flux term and the damping coefficient are built (ADR-037).
//   Exact - the steady equations contain no dt at all:
//     * the old residual R = F - I[u].S uses the SAME skew-corrected
//       interpolation I as the predicted flux;
//     * the predicted flux interpolates q = H/aP - (V/aP) grad p and adds
//       D_f L[grad p].S back, so the product is never interpolated as one;
//     * D_f = V_f / aP_f, volume and aP interpolated separately, so that
//       1/D_f - a0 = aPs_f / V_f exactly;
//     * an outlet face carries the old-flux term too, with the cell value in
//       place of the interpolation.
//     A steady state then has F = I[u].S + (V_f/aPs_f)(L[grad p].Delta -
//     a_f (p_N - p_P)): the damping with the spatial part of aP only.
//   V1 - R = F - L[u].S with plain linear L, D_f = interp(V/aP). On a skewed
//     mesh the steady residual carries (I - L)[u].S divided by 1 - D_f a0,
//     about (2/3) Co in a cell, which falls with dt: the steady state moved by
//     69% of its discretisation error across dt = 0.02 .. 2.0, and the
//     cylinder's far wake grew a spurious velocity at dt = 0.025 (ADR-031).
//     Kept so the gate can be shown to fail on it.
enum class OldFluxForm : int { Exact = 0, V1 = 1 };

// Momentum's convected face value, carried as the deferred correction to
// upwind (ADR-042).
//   Linear       - linear interpolation plus the skewness correction. Central:
//                  it does not see the odd-even mode, and where no diffusion
//                  damps that mode -- cells long in the stream direction at a
//                  large cell Peclet number -- a sharp change downstream sets
//                  it going upstream. The default.
//   LinearUpwind - the upwind cell's value extrapolated to the face centre by
//                  its least-squares gradient, u_U + grad(u)_U . (x_f - x_U).
//                  Exact for a linear field, so second order too; the
//                  odd-even mode it sees as upwinding does. The flat plate's,
//                  whose leading edge showed the mode on TMR's grids.
enum class ConvectionScheme : int { Linear = 0, LinearUpwind = 1 };

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
  // Momentum predictor sweeps per outer iteration. Convection is upwind in
  // the matrix plus a deferred correction evaluated from the velocity of the
  // previous sweep; with one sweep that correction lags a whole outer
  // iteration. More sweeps re-evaluate it and re-solve until the velocity
  // stops moving by convectionSweepTol (relative). The matrix -- and so aP,
  // which the Rhie-Chow flux reads -- stays upwind, so the converged answer
  // is the same one; only how fast the outer loop gets there changes. The
  // roadmap called this "making the deferred correction implicit"; a truly
  // implicit central matrix would change aP, could make it vanish, and so
  // would change the Rhie-Chow flux as well as the convergence (ADR-036).
  int  convectionSweeps = 1;
  Real convectionSweepTol = 0.0;
  bool consistentRhieChow = true;
  RhieChowForm rhieChowForm = RhieChowForm::Standard;
  OldFluxForm oldFlux = OldFluxForm::Exact;
  // Second-order convection is carried as a deferred correction on the
  // right-hand side, which is explicit. Turning it off leaves first-order
  // upwind: wrong, but unconditionally stable in the convective term. It is
  // here as a diagnostic -- when a run grows without bound, this says in one
  // experiment whether the explicit correction is what is growing.
  bool deferredCorrection = true;
  // Which second-order face value that correction goes to, for momentum only;
  // the energy equation and k and omega have their own (see the enum).
  ConvectionScheme convection = ConvectionScheme::Linear;
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
  // Start each extrapolation from the boundary values the last one produced,
  // and sweep once, instead of restarting from zero normal gradient. OFF, and
  // kept only as recorded evidence (ADR-034). Warm-started, the truncated
  // fixed-point iteration keeps going across calls and reaches the fixed
  // point three cold sweeps stop short of -- the same answer as ten cold
  // sweeps -- but it trails the pressure by a sweep, and the loops that call
  // it pay: 6 outer iterations instead of 4 on Ethier-Steinman, twice the
  // non-orthogonal sweeps and +52% wall time on the cylinder. Ten cold sweeps
  // move the cylinder drag by 0.0001, so three stand.
  bool pressureExtrapWarmStart = false;
  // Two diagnostic switches for ADR-043's ablations, off by default: BDF1 on
  // every step instead of only the first, and the momentum predictor's
  // relative tolerance (1e-13, as it always was).
  bool bdf1 = false;
  Real momentumSolveTol = 1e-13;
  // And one for its cause: the aP that the Rhie-Chow coefficient D_f (and
  // with it the pressure equation) reads leaves out the diffusion through the
  // faces normal to this axis (0, 1, 2; -1, the default, leaves nothing
  // out). The cell velocity's V/aP keeps it. It changes the discretisation
  // and exists only to show what that diffusion does to the correctors.
  int rhieChowAxisOff = -1;
};

// Coarse phase timings. ADR-016 was written because a sweep count was
// mistaken for a profile; the same mistake was available again here, where
// the two linear solvers together account for barely half the wall time.
struct PisoTimings {
  Real assemble{}, gradient{}, boundaryP{}, hbya{}, rhieChow{},
       pressureAssembly{}, momentumSolve{}, pressureSolve{}, total{};
  int  gradCalls{}, boundaryPCalls{};
  // Least-squares gradient passes spent inside the boundary-pressure
  // extrapolation: sweeps, summed over calls. Deterministic, unlike the
  // seconds beside it, so a change in the sweep logic can be costed from it.
  long long boundaryPSweeps{};
};

// Everything the solver knows about one cell and its faces, after a step.
//
// ADR-024: five explanations for the fine-mesh instability survived while the
// only diagnostics were norms over the whole field. A norm says a run is
// going wrong; it cannot say which term in which equation is doing it. This
// does: every part of the predicted flux, every part of the correction, the
// neighbour states and the boundary values, so a growing mode can be watched
// term by term in the one place it lives.
struct FaceProbe {
  Index face{-1};
  bool boundary{false};
  Index other{-1};            // neighbour cell, -1 on a boundary face
  Real F{}, Fstar{};          // OUTWARD from the probed cell
  Real HfS{}, Dgpf{}, DsnOld{}, choi{};   // parts of F*, outward
  Real DsnNew{}, nonorth{};   // the pressure correction, outward
  Real area{};
  Real uOther[3]{}, pOther{};
  Real pBnd{};                // boundary faces: the face pressure the solver uses
  int uBC{-1}, pBC{-1};
};
struct CellProbe {
  Index cell{-1};
  Real x[3]{}, vol{};
  Real u[3]{}, p{}, aP{}, VbyAP{};
  Real HbyA[3]{}, gp[3]{}, corr[3]{};   // corr = grad(p) V/aP, what the corrector removes
  Real divF{};
  std::vector<FaceProbe> faces;
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
  int convectionSweeps{};     // momentum predictor sweeps, summed over the outer iterations
  int nonOrthSweeps{};
};

class PisoSolver {
 public:
  PisoSolver(const Mesh& mesh, Real nu, Real dt, PisoControls controls = {},
             Comm comm = Comm());
  ~PisoSolver();

  // bcType is one VelocityBC per boundary face; empty means all Dirichlet.
  void setBoundaryTypes(const View1<int>& bcType);

  // The energy equation (ADR-038). Off until enabled; with it off the solver
  // is the v1 solver, operation for operation. Temperature is transported
  // like a velocity component -- BDF2, upwind plus the deferred correction to
  // the skew-corrected face value, diffusion with the non-orthogonal
  // correction -- and solved inside every outer iteration after the pressure
  // correctors, with the corrected face flux, so that a converged outer loop
  // carries no coupling lag. The outer loop's convergence test includes T.
  void enableEnergy(const EnergyModel& model);
  // The solver for the balanced form's hydrostatic pressure p_h (ADR-044):
  // before or after enableEnergy. Without a call, or with a null pointer,
  // the native Jacobi CG with its null-space projection (ADR-041). The solve
  // keeps its tolerances -- the residual's 2-norm below 1e-15 of the
  // right-hand side's -- so a backend must measure that norm: a PetscSolver
  // for p_h takes setConstantNullSpace and setUnpreconditionedNorm.
  void setHydrostaticSolver(std::unique_ptr<LinearSolver> solver);
  // type is one TemperatureBC per boundary face, value the prescribed
  // temperature or heat flux. Without a call every face is FixedValue at 0.
  void setTemperatureBoundary(const View1<int>& type, const ScalarField& value);
  // Volumetric source S_T per cell, in temperature per unit time.
  void setTemperatureSource(const ScalarField& source);
  // Sets the current and both old time levels, as setState does for u.
  void setTemperature(const ScalarField& T);
  ScalarField temperature() const { return T_; }
  // Diffusive heat flux INTO the domain through each boundary face,
  // integrated over the face, from the same discrete operator the energy
  // equation applies there -- the Nusselt number is read from this.
  ScalarField boundaryHeatFlux() const;

  // The k-omega SST model (ADR-042). Off until enabled; with it off every
  // operation is the v2a one. k and omega are transported like the
  // temperature and solved in every outer iteration after the pressure
  // correctors and the energy equation, then nu_t is updated; the momentum
  // equation takes nu + nu_t on every face and the explicit part
  // div(nu_t grad(u)^T) of the stress. wallMask flags, per boundary face, the
  // faces the wall distance is measured from.
  void enableTurbulence(const TurbulenceModel& model, const View1<int>& wallMask);
  // kind is one TurbulenceBC per boundary face; kValue and wValue the
  // prescribed values on the Dirichlet ones. Without a call every face is
  // ZeroGradient.
  void setTurbulenceBoundary(const View1<int>& kind, const ScalarField& kValue,
                             const ScalarField& wValue);
  // Imposed volumetric sources, per unit time (a manufactured solution's).
  // Their negative parts enter by Patankar's rule, on the diagonal.
  void setTurbulenceSource(const ScalarField& kSource, const ScalarField& wSource);
  // Sets the current and both old time levels, and nu_t from them.
  void setTurbulence(const ScalarField& k, const ScalarField& w);
  ScalarField turbulentKineticEnergy() const { return k_; }
  ScalarField specificDissipation() const { return w_t_; }
  ScalarField eddyViscosity() const { return nut_; }
  ScalarField wallDistance() const { return dWall_; }
  ScalarField boundaryWallDistance() const { return dWallB_; }
  // Cells bounded below since the model was enabled.
  long long boundedCells() const { return bounded_; }
  // One BDF step of k and omega alone, in the velocity and face flux the
  // solver holds (setState) and the given boundary values: Eca's "frozen
  // velocity" exercise, gate 2a.
  void advanceTurbulenceFrozen(const VectorField& uBoundary, const ScalarField& fBoundary,
                               LinearSolver& solver);

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
  // Diagnostic (ADR-034): restart the boundary-pressure extrapolation cold on
  // the current pressure and sweep it n times. Entry k-1 is the largest change
  // any boundary value made in sweep k, relative to the largest |p| in the
  // domain, so the list shows how far a truncated sweep count stops short of
  // the fixed point. Leaves the solver's state untouched.
  std::vector<Real> extrapolationHistory(int n) const;

  // Net force the fluid exerts on the faces where mask is non-zero, computed
  // from the SAME discrete operators the momentum equation uses. Recomputing
  // the wall stress with an independent formula would measure a different
  // equation than the one being solved.
  Vec3 boundaryForce(const View1<int>& mask) const;

  // A new time step size. The next step restarts BDF at first order, so the
  // old two-level history is never combined with the new step. A steady state
  // does not depend on the step (ADR-037), so a march may ramp it.
  void setTimeStep(Real dt) { dt_ = dt; step_ = 0; turbSteps_ = 0; }

  StepReport advance(const VectorField& uBoundary, const ScalarField& fBoundary,
                     const VectorField& source, LinearSolver& momentumSolver,
                     LinearSolver& pressureSolver);

  VectorField velocity() const { return u_; }
  ScalarField pressure() const { return p_; }
  ScalarField faceFlux() const { return F_; }
  void setState(const VectorField& u, const ScalarField& p, const ScalarField& F);

  Real continuityError(const ScalarField& F, const ScalarField& Fb) const;
  Real courant() const;

  // Probe one cell. Costs one extra predictable branch in the Rhie-Chow
  // kernel while enabled and nothing at all otherwise.
  void enableProbe(Index cell);
  CellProbe probe() const;

  const PisoTimings& timings() const { return t_; }
  void resetTimings() { t_ = PisoTimings{}; }

 private:
  void bdf(Real& aP, Real& a1, Real& a2) const;
  void assembleMomentum(const VectorField& uB, const VectorField& src);
  // Boundary velocity with every slip face replaced by the tangential part of
  // its cell's current velocity; the caller's value everywhere else.
  void slipBoundaryVelocity(const VectorField& uB);
  // src plus the buoyancy of the current temperature, into srcTotal_.
  void addBuoyancy(const VectorField& src);
  // The balanced form (ADR-041): the face force, its hydrostatic pressure,
  // the face residual and the cell force reconstructed from it; srcTotal_ =
  // src + that cell force.
  void balancedBuoyancy(const VectorField& src);
  void solveEnergy(LinearSolver& solver);
  // Gradient of T with the boundary values the energy equation uses.
  void gradT(VectorField& g) const;
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
  ScalarField aPrc_;           // the aP D_f reads: aP_ itself unless rhieChowAxisOff
  // The non-orthogonal correction, kept ACROSS calls. It is the fixed point of
  // a deferred-correction loop whose answer moves only a little from one
  // pressure solve to the next, so starting from the last one costs nothing
  // and saves nearly all the sweeps. Starting from zero each time re-derived
  // the same field from scratch a hundred times a step, and every sweep is a
  // full pressure solve plus a boundary-pressure extrapolation.
  ScalarField nonorth_;
  VectorField skew_;

  VectorField u_, uOld_, uOld2_, HbyA_, gp_;
  VectorField gH0_, gH1_, gH2_;   // gradients of the interpolated cell
                                  // vector (H/aP, or q in the exact form),
                                  // for the skewness correction on its face
                                  // value
  VectorField q_;              // H/aP - (V/aP) grad p, the exact form's
                               // interpolated vector (ADR-037)
  ScalarField rOld_, rOldB_;   // old Rhie-Chow residual per internal face and
                               // per outlet face, once per step (ADR-037)
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
  bool probing_{false};
  Index probeCell_{-1};
  View2<Real> rcComp_;        // per internal face: HfS, D gpf, D snGrad, Choi
  // Last boundary pressure from pressureBoundary, for the warm start. Only
  // p_ is ever extrapolated, so one buffer is enough; setState invalidates it.
  mutable ScalarField pbWarm_;
  mutable bool pbWarmValid_{false};
  // gp_ holds grad(p_) for the p_ now stored. The last non-orthogonal sweep
  // of solvePressure computes exactly that gradient, and p_ does not change
  // again until the next pressure solve, so the corrector and the next outer
  // iteration can use it instead of recomputing it: the same numbers, nine
  // fewer gradient-plus-extrapolation passes per step (ADR-034).
  bool gpValid_{false};

  bool exactOldFlux() const {
    return ctl_.oldFlux == OldFluxForm::Exact &&
           ctl_.rhieChowForm == RhieChowForm::Standard;
  }
  void computeOldResidual();

  // Slip faces (ADR-039). hasSlip_ skips all of it when there are none, so a
  // v1 case runs exactly the operations it ran before.
  bool hasSlip_{false};
  VectorField uBEff_;          // the boundary velocity the momentum equation
                               // actually used, slip faces included

  // Energy equation (ADR-038).
  bool energy_{false};
  EnergyModel em_;
  ScalarField T_, TOld_, TOld2_, TSrc_;
  View1<int> tType_;           // TemperatureBC per boundary face
  ScalarField tValue_;         // prescribed temperature or heat flux
  VectorField srcTotal_;       // caller's source plus buoyancy

  // The SST model (ADR-042).
  bool turb_{false};
  TurbulenceModel tm_;
  ScalarField k_, kOld_, kOld2_, w_t_, wOld_, wOld2_;   // w_t_: omega (w_ is taken)
  ScalarField nut_, nutB_;     // cells, boundary faces
  ScalarField dWall_, dWallB_; // wall distance: cells, boundary face centres
  View1<int> kwType_;          // TurbulenceBC per boundary face
  ScalarField kValue_, wValue_, kSrc_, wSrc_;
  long long bounded_{0};
  Index turbSteps_{0};         // advanceTurbulenceFrozen's own BDF counter
  bool nutValid_{false};       // nu_t computed from the current k, omega and u
  // k and omega, then nu_t: one call per outer iteration, with the BDF
  // coefficients of the step.
  void solveTurbulence(LinearSolver& solver, const VectorField& uB, Real aPt, Real a1, Real a2);
  void updateEddyViscosity(const VectorField& uB);
  // Face values of k and omega: prescribed, the wall's, or the cell's.
  void turbulenceBoundaryValues(ScalarField& kb, ScalarField& wb) const;
  // S = sqrt(2 S_ij S_ij) and Omega = sqrt(2 W_ij W_ij) from the velocity
  // gradient with the momentum equation's boundary values.
  void velocityInvariants(const VectorField& uB, ScalarField& S, ScalarField& Om) const;
  void transportSolve(ScalarField& phi, const ScalarField& old, const ScalarField& old2,
                      const ScalarField& phib, const VectorField& gphi,
                      const ScalarField& Gc, const ScalarField& Gb,
                      const ScalarField& diagV, const ScalarField& rhsV,
                      Real aPt, Real a1, Real a2, LinearSolver& solver);

  // Balanced buoyancy (ADR-041).
  bool balanced_{false};
  ScalarField pH_;             // hydrostatic pressure of the face force
  ScalarField rFace_;          // face residual, interior faces
  ScalarField kgH_;            // p_h's non-orthogonal part, kept across calls
  VectorField gCell_;          // cell force reconstructed from the residual
  View2<Real> reconMinv_;      // per cell, (sum S S^T / |S|)^-1, row-major
  std::unique_ptr<LinearSystem> phSys_;   // p_h's Laplacian: constant
  std::unique_ptr<LinearSolver> phSolver_;
};

}  // namespace vibeflow
