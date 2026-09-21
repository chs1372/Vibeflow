#pragma once
// Laplacian with over-relaxed non-orthogonal correction.
//
//   Sf = Delta + k,   Delta = d |Sf|^2 / (d.Sf)
//
// The Delta part is implicit (keeps the full face area in the matrix, so the
// system stays well conditioned on skewed meshes); k is deferred-corrected
// against the least-squares gradient. Mirrors prototype/fvm.py.

#include "core/Parallel.hpp"
#include "core/Types.hpp"
#include "discretization/Gradient.hpp"

namespace nsflow {

class Mesh;
class LinearSystem;
class LinearSolver;

class DiffusionOperator {
 public:
  DiffusionOperator(const Mesh& mesh, Real gamma, Comm comm = Comm());

  // Assemble the constant implicit coefficients. Call once.
  void assembleMatrix(LinearSystem& sys) const;

  // Rebuild the explicit side for the current iterate.
  void assembleSource(LinearSystem& sys, const ScalarField& volSource,
                      const ScalarField& phiB, const ScalarField& phiPrev) const;

  // Deferred-correction loop. Returns the number of sweeps used.
  int solve(LinearSystem& sys, LinearSolver& solver, const ScalarField& volSource,
            const ScalarField& phiB, ScalarField& phi,
            int maxSweeps = 40, Real tol = 1e-12) const;

 private:
  const Mesh& m_;
  Real gamma_;
  Comm comm_;
  ScalarField aInt_, aBnd_, wOwner_;
  VectorField kInt_, kBnd_;
  LeastSquaresGradient grad_;
};

}  // namespace nsflow
