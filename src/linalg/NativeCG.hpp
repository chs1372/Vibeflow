#pragma once
// Jacobi-preconditioned conjugate gradient in face-based storage.
//
// This is NOT the production solver -- PetscSolver (hypre BoomerAMG / AmgX) is.
// It exists so the verification gates run in CI without a PETSc build, and so
// every PETSc result has a dependency-free reference to be compared against.
// Valid only for symmetric systems, which pure diffusion is.

#include "linalg/LinearSolver.hpp"
#include "core/Parallel.hpp"
#include "core/Types.hpp"

namespace vibeflow {

class Mesh;

class NativeCG final : public LinearSolver {
 public:
  // singularNullSpace: the operator has the constants in its null space, as
  // the pure-Neumann pressure equation does. CG then needs the constant
  // component projected out of the residual at every iteration, not just out
  // of the right-hand side once.
  explicit NativeCG(const Mesh& mesh, Comm comm = Comm(),
                    bool singularNullSpace = false);
  SolveReport solve(LinearSystem& sys, ScalarField& x,
                    Real relTol, Real absTol, int maxIter) override;
  std::string backendName() const override { return "native-cg(jacobi)"; }

  // y = A x, face-based.
  void apply(LinearSystem& sys, const ScalarField& x, ScalarField& y) const;

 private:
  const Mesh& m_;
  Comm comm_;
  bool nullSpace_;
  ScalarField r_, z_, p_, q_;
  void projectOut(ScalarField& v) const;
};

}  // namespace vibeflow
