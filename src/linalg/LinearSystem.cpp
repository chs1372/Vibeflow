#include "linalg/LinearSystem.hpp"
#include "mesh/Mesh.hpp"

namespace nsflow {

LinearSystem::LinearSystem(const Mesh& mesh)
    : diag_("diag", mesh.nCells()),
      upper_("upper", mesh.nInternalFaces()),
      lower_("lower", mesh.nInternalFaces()),
      source_("source", mesh.nCells()) {}

void LinearSystem::zero() {
  Kokkos::deep_copy(diag_, 0.0);
  Kokkos::deep_copy(upper_, 0.0);
  Kokkos::deep_copy(lower_, 0.0);
  Kokkos::deep_copy(source_, 0.0);
}

}  // namespace nsflow
