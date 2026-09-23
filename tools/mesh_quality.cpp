// Report the mesh quality measures the solver actually reacts to, without
// running the solver. Iterating on a mesh generator by launching a flow
// solve is slow enough that it discourages iterating at all.
#include "mesh/PolyMesh.hpp"
#include <cstdio>

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: mesh_quality <file.hex>\n"); return 2; }
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    auto mesh = nsflow::PolyMesh::fromHexFile(argv[1]);
    std::printf("%s\n  cells %d  faces %d/%d  non-orth %.1f deg  skew %.2f\n",
                argv[1], static_cast<int>(mesh.nCells()),
                static_cast<int>(mesh.nInternalFaces()),
                static_cast<int>(mesh.nBoundaryFaces()),
                mesh.maxNonOrthogonality(), mesh.maxSkewness());
  }
  Kokkos::finalize();
  return rc;
}
