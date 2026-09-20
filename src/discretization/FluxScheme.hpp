#pragma once
// THE BRANCH POINT between the pressure-based (incompressible, low-Mach) and
// density-based (compressible) solver paths.
//
// Everything above this interface -- transport equations, turbulence models,
// time integration -- is written against FluxScheme and does not know which
// path is active. Keeping this boundary clean is what makes v3 an addition
// rather than a rewrite.

#include "core/Types.hpp"
#include "linalg/LinearSystem.hpp"

namespace nsflow {

class Mesh;

class FluxScheme {
 public:
  virtual ~FluxScheme() = default;

  // Assemble the implicit convection + diffusion contribution of one transported
  // variable into `sys`. Non-orthogonal and higher-order terms go to the
  // explicit side of `sys` by deferred correction.
  virtual void assemble(const Mesh& mesh,
                        const ScalarField& phi,
                        const ScalarField& massFlux,   // (nInternalFaces)
                        Real gamma,
                        LinearSystem& sys) = 0;

  // Number of deferred-correction sweeps this scheme needs to reach its formal
  // order. Orthogonal meshes converge in 1-2; heavily skewed meshes need more.
  virtual int correctorSweeps() const = 0;
};

}  // namespace nsflow
