#pragma once
// Convection-diffusion operator.
//
// Convection is implicit first-order upwind plus a deferred correction to the
// second-order face value. Upwind alone keeps the matrix an M-matrix at any
// cell Peclet number; the correction recovers second order without giving
// that up.
//
// The second-order face value carries a skewness correction,
//     phi_f = [w phi_P + (1-w) phi_N] + grad(phi)_f . (x_f - x_lin)
// where x_lin is the point linear interpolation actually refers to. Dropping
// that term costs a full order on a distorted mesh.

#include "core/Types.hpp"
#include "discretization/Diffusion.hpp"
#include "discretization/Gradient.hpp"

namespace nsflow {

class Mesh;
class LinearSystem;
class LinearSolver;

class ConvectionDiffusion {
 public:
  ConvectionDiffusion(const Mesh& mesh, Real gamma,
                      const ScalarField& fInternal, const ScalarField& fBoundary);

  void assembleMatrix(LinearSystem& sys) const;
  void assembleSource(LinearSystem& sys, const ScalarField& volSource,
                      const ScalarField& phiB, const ScalarField& phiPrev) const;

  // Deferred-correction loop. Reports whether it converged: a truncated loop
  // still returns a plausible field, so the caller must be told rather than
  // left to infer it from the iteration count.
  int solve(LinearSystem& sys, LinearSolver& solver, const ScalarField& volSource,
            const ScalarField& phiB, ScalarField& phi, bool& converged,
            int maxSweeps = 400, Real tol = 1e-12, Real relax = 1.0) const;

  Real maxPeclet() const;

 private:
  const Mesh& m_;
  Real gamma_;
  ScalarField F_, Fb_;
  ScalarField aInt_, aBnd_, wOwner_;
  VectorField kInt_, kBnd_, skew_;
  LeastSquaresGradient grad_;
};

}  // namespace nsflow
