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

namespace nsflow {

class Mesh;

class NativeCG final : public LinearSolver {
 public:
  explicit NativeCG(const Mesh& mesh, Comm comm = Comm());
  SolveReport solve(LinearSystem& sys, ScalarField& x,
                    Real relTol, Real absTol, int maxIter) override;
  std::string backendName() const override { return "native-cg(jacobi)"; }

  // y = A x, face-based.
  void apply(LinearSystem& sys, const ScalarField& x, ScalarField& y) const;

 private:
  const Mesh& m_;
  Comm comm_;
  ScalarField r_, z_, p_, q_;
};

}  // namespace nsflow
