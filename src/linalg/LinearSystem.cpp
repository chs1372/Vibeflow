#include "linalg/LinearSystem.hpp"
#include "mesh/Mesh.hpp"

namespace vibeflow {

LinearSystem::LinearSystem(const Mesh& mesh)
    // Sized over TOTAL cells: face loops write into ghost rows. Those rows are
    // never read back, but they must exist.
    : diag_("diag", mesh.nTotal()),
      upper_("upper", mesh.nInternalFaces()),
      lower_("lower", mesh.nInternalFaces()),
      source_("source", mesh.nTotal()) {}

void LinearSystem::zero() {
  Kokkos::deep_copy(diag_, 0.0);
  Kokkos::deep_copy(upper_, 0.0);
  Kokkos::deep_copy(lower_, 0.0);
  Kokkos::deep_copy(source_, 0.0);
}

}  // namespace vibeflow
