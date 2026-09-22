#pragma once
// Jacobi-preconditioned BiCGStab, for the NON-SYMMETRIC systems convection
// produces. CG is only valid for the symmetric diffusion-only case, so using
// it once convection is switched on would converge to the wrong thing while
// still reporting a small residual.
//
// Like NativeCG this is not the production solver -- PETSc is -- but it keeps
// the gates runnable without a PETSc build and gives every PETSc result an
// independent reference.

#include "core/Parallel.hpp"
#include "core/Types.hpp"
#include "linalg/LinearSolver.hpp"

namespace nsflow {

class Mesh;

class NativeBiCGStab final : public LinearSolver {
 public:
  explicit NativeBiCGStab(const Mesh& mesh, Comm comm = Comm());
  SolveReport solve(LinearSystem& sys, ScalarField& x,
                    Real relTol, Real absTol, int maxIter) override;
  std::string backendName() const override { return "native-bicgstab(jacobi)"; }

 private:
  void apply(LinearSystem& sys, const ScalarField& x, ScalarField& y) const;
  const Mesh& m_;
  Comm comm_;
  ScalarField r_, r0_, p_, v_, s_, t_, ph_, sh_;
};

}  // namespace nsflow
