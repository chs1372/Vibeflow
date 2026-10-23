// ADR-047: which rank owns each cell of a .hex mesh at P ranks under RCB.
// It calls the function the harness's DistributedMesh calls,
// DistributedMesh::partition(raw, P, RCB), on one process. The gate script
// reads it to decide, before any fault run, whether rank 0 owns the cell that
// sets C or a relChange scale at a 50-step line (the F5 and F6 exception).
//
// Run:  backstep_owners <mesh.hex> <P>
// Prints one line per cell in global (file) order: "<global id> <rank>".
#include "mesh/DistributedMesh.hpp"
#include "mesh/RawMesh.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vibeflow;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::printf("usage: backstep_owners <mesh.hex> <P>\n");
    return 2;
  }
  const int P = std::atoi(argv[2]);
  if (P < 1) {
    std::printf("P must be at least 1\n");
    return 2;
  }
  const RawMesh raw = RawMesh::fromHexFile(argv[1]);
  const std::vector<int> part = DistributedMesh::partition(raw, P, PartitionMethod::RCB);
  for (std::size_t g = 0; g < part.size(); ++g) std::printf("%zu %d\n", g, part[g]);
  return 0;
}
