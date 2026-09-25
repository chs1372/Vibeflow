#pragma once
// Backend-agnostic linear solve. Concrete backends: PetscSolver (hypre
// BoomerAMG / AmgX / Ginkgo selected at runtime from the case dictionary).
//
// The pressure Poisson solve is 60-80% of incompressible runtime, so every
// backend must report its own timing and iteration count for profiling.

#include "core/Types.hpp"
#include <string>

namespace vibeflow {

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

  // CONTRACT: on return, x has a valid halo. Callers read x at ghost cells
  // immediately -- a gradient, a face flux, a Rhie-Chow interpolation -- and
  // a stale ghost there produces a plausible wrong answer rather than a
  // crash. The iterative solvers exchange inside the matrix-vector product,
  // which leaves the halo one update behind at exit, so each must exchange
  // once more before returning.

  // Called when the coefficients change. A backend that builds something
  // expensive from the matrix -- a factorisation, an AMG hierarchy -- rebuilds
  // it only after this. The non-orthogonal pressure corrector solves the SAME
  // matrix 20-40 times in a row with a different right-hand side each time
  // (ADR-016), so rebuilding per solve would throw away most of what AMG buys.
  virtual void notifyMatrixChanged() {}

  // Cumulative wall time and iteration count, for profiling.
  Real totalSeconds() const { return totalSeconds_; }
  int totalIterations() const { return totalIterations_; }
  int solveCount() const { return solveCount_; }
  void resetCounters() { totalSeconds_ = 0.0; totalIterations_ = 0; solveCount_ = 0; }

 protected:
  void record(const SolveReport& r) {
    totalSeconds_ += r.wallSeconds;
    totalIterations_ += r.iterations;
    ++solveCount_;
  }

 private:
  Real totalSeconds_{0.0};
  int totalIterations_{0}, solveCount_{0};

 public:
};

}  // namespace vibeflow
