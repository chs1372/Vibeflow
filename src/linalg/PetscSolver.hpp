#pragma once
// Production linear-solver backend: PETSc, with hypre BoomerAMG available as a
// preconditioner.
//
// The pressure Poisson solve is 60-80% of incompressible runtime, so this is
// where solver performance is won or lost. The backend is selected at runtime
// from a string ("cg+jacobi", "cg+hypre", "gmres+hypre", ...) so a case can
// change preconditioner without a rebuild, and so the GPU path later is a
// configuration change rather than a code change.
//
// Assembly stays in face-based storage; this class translates once per matrix
// change. Ghost columns are mapped to their owning rank's global rows, so a
// distributed mesh assembles into a real parallel PETSc matrix.

#include "core/Parallel.hpp"
#include "core/Types.hpp"
#include "linalg/LinearSolver.hpp"
#include <memory>
#include <string>
#include <vector>

namespace nsflow {

class Mesh;

class PetscSolver final : public LinearSolver {
 public:
  // globalRowOf maps each LOCAL cell (owned, then ghost) to its global row.
  // Pass an empty vector for a serial mesh.
  // pcRebuildInterval: rebuild the preconditioner only every Nth time the
  // matrix changes, reusing it in between. A preconditioner does not have to
  // match the matrix it preconditions -- it only has to be close enough that
  // the Krylov method converges quickly. For BoomerAMG the setup is most of
  // the cost, and the pressure matrix changes only slightly from one PISO
  // corrector to the next, so reusing it is nearly free accuracy-wise and
  // large wall-clock-wise.
  PetscSolver(const Mesh& mesh, Comm comm, std::string config = "cg+jacobi",
              std::vector<Index> globalRowOf = {}, int pcRebuildInterval = 1);
  ~PetscSolver() override;

  SolveReport solve(LinearSystem& sys, ScalarField& x,
                    Real relTol, Real absTol, int maxIter) override;
  void notifyMatrixChanged() override;
  std::string backendName() const override { return "petsc(" + config_ + ")"; }

  // True when this PETSc build actually has hypre; a case asking for it on a
  // build without it should fail loudly rather than silently use something else.
  static bool hasHypre();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string config_;
};

}  // namespace nsflow
