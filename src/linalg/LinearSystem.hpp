#pragma once
// Matrix + right-hand side in face-based (owner/neighbour) storage.
//
// Assembly never touches PETSc. LinearSolver translates this into whatever the
// backend wants, so switching hypre <-> AmgX <-> Ginkgo changes no solver code.

#include "core/Types.hpp"

namespace vibeflow {

class Mesh;

class LinearSystem {
 public:
  explicit LinearSystem(const Mesh& mesh);

  ScalarField diag()        { return diag_; }   // (nCells)
  ScalarField upper()       { return upper_; }  // (nInternalFaces) owner -> neighbour
  ScalarField lower()       { return lower_; }  // (nInternalFaces) neighbour -> owner
  ScalarField source()      { return source_; } // (nCells)

  void zero();
  Real residualNorm(const ScalarField& x) const;

 private:
  ScalarField diag_, upper_, lower_, source_;
};

}  // namespace vibeflow
