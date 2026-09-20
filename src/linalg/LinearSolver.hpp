#pragma once
// Backend-agnostic linear solve. Concrete backends: PetscSolver (hypre
// BoomerAMG / AmgX / Ginkgo selected at runtime from the case dictionary).
//
// The pressure Poisson solve is 60-80% of incompressible runtime, so every
// backend must report its own timing and iteration count for profiling.

#include "core/Types.hpp"
#include <string>

namespace nsflow {

class LinearSystem;

struct SolveReport {
  int  iterations{};
  Real initialResidual{};
  Real finalResidual{};
  Real wallSeconds{};
  bool converged{};
};

class LinearSolver {
 public:
  virtual ~LinearSolver() = default;
  virtual SolveReport solve(LinearSystem& sys, ScalarField& x,
                            Real relTol, Real absTol, int maxIter) = 0;
  virtual std::string backendName() const = 0;
};

}  // namespace nsflow
