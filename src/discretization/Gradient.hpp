#pragma once
// Weighted least-squares cell gradient, including boundary faces.
// Mirrors prototype/fvm.py::LeastSquaresGradient. The 3x3 normal matrix is
// inverted once at construction; each evaluation is one pass over the faces.

#include "core/Types.hpp"

namespace nsflow {

class Mesh;

class LeastSquaresGradient {
 public:
  explicit LeastSquaresGradient(const Mesh& mesh);
  // grad <- gradient of phi, with boundary face values phiB.
  void operator()(const ScalarField& phi, const ScalarField& phiB,
                  VectorField& grad) const;

 private:
  const Mesh& m_;
  Kokkos::View<Real***, MemSpace> Ainv_;   // (nCells, 3, 3)
  VectorField dInt_, dBnd_;
  ScalarField wInt_, wBnd_;
};

}  // namespace nsflow
