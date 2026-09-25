// Memory-scaling gate: the mesh must cost less per rank as ranks are added.
//
// Every other parallel gate asks whether the ANSWER depends on the rank count.
// None of them would notice a partitioner that gets the right answer by
// building the entire global mesh on every rank and throwing away the parts it
// does not own -- which is exactly what this one did (ADR-006), and why a
// mesh that fits on one workstation was the largest mesh the solver could run
// no matter how many ranks it was given.
//
// So this gate measures the build, not the result. Two numbers per rank:
//
//   * cells and faces CONSTRUCTED -- deterministic, allocator-independent, and
//     counted before the faces that turn out to belong to another rank are
//     discarded, because they were still built.
//   * peak resident set, as the kernel reports it. Reported, not gated: it
//     includes Kokkos, MPI and PETSc start-up, it is allocator-dependent, and
//     on an oversubscribed box it is noisy. It is here because it is the thing
//     anyone actually cares about, and a count that falls while the RSS does
//     not would be worth knowing.
//
// The bar: on P ranks, the most any single rank builds must be at most
// (1/P + 0.35) of the serial build. The slack is for the ghost ring, which is
// real work that does not shrink as fast as the interior. On 4 ranks that is
// 0.60 -- a partitioner that replicates the mesh scores 1.00 and fails.
//
// A trivial way to pass would be to build nothing, so the gate also checks
// that the owned cells still add up to the global mesh and that no rank built
// fewer cells than it owns.
//
// Run:  mpirun -n <N> mms_parallel_mem [serial faces built]

#include "core/Parallel.hpp"
#include "mesh/RawMesh.hpp"
#include "mesh/DistributedMesh.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

using namespace vibeflow;

namespace {

// VmHWM: the high-water mark of the resident set, so it captures the transient
// peak during face discovery rather than only what survives it.
long peakRssKb() {
  std::ifstream in("/proc/self/status");
  std::string key;
  long value = -1;
  while (in >> key) {
    if (key == "VmHWM:") { in >> value; return value; }
    std::getline(in, key);
  }
  return -1;
}

}  // namespace

int main(int argc, char** argv) {
  ParallelScope mpi(argc, argv);
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const Comm comm = Comm::world();
    const Real ref = argc > 1 ? std::atof(argv[1]) : -1.0;
    const Index n = argc > 2 ? std::atoi(argv[2]) : 40;

    // Large enough that the mesh dominates the process, small enough to run
    // four of them on a laptop.
    RawMesh raw = RawMesh::generate(n, 0.3, "smooth");
    const Index globalCells = raw.nCells();

    const long rssBefore = peakRssKb();
    DistributedMesh mesh(raw, comm, PartitionMethod::RCB);
    const long rssAfter = peakRssKb();

    const Index built = mesh.nFacesBuilt();
    const Index cellsBuilt = mesh.nCellsBuilt();
    const Index owned = mesh.nCells();

    // Worst rank, not the average: the peak is what has to fit in memory.
    const Real maxBuilt = comm.max(static_cast<Real>(built));
    const Real maxRss = comm.max(static_cast<Real>(rssAfter));
    const Index totalOwned = comm.sum(owned);

    if (cellsBuilt < owned) {
      std::printf("  rank %d built %d cells but owns %d\n", comm.rank(),
                  static_cast<int>(cellsBuilt), static_cast<int>(owned));
      rc = 1;
    }

    if (comm.rank() == 0) {
      std::printf("  ranks %2d   global cells %6d   worst rank: built %6d cells "
                  "%7d faces   peak RSS %ld MB (+%ld)\n",
                  comm.size(), static_cast<int>(globalCells),
                  static_cast<int>(cellsBuilt), static_cast<int>(maxBuilt),
                  static_cast<long>(maxRss) / 1024, (rssAfter - rssBefore) / 1024);
      if (ref > 0.0) {
        const Real allowed = ref * (1.0 / comm.size() + 0.35);
        const Real ratio = maxBuilt / ref;
        const bool ok = maxBuilt <= allowed;
        std::printf("            built/serial %.3f, allowed %.3f  %s\n",
                    ratio, allowed / ref, ok ? "PASS" : "FAIL");
        if (!ok) rc = 1;
      }
      if (totalOwned != globalCells) {
        std::printf("  FAIL: owned cells sum to %d, mesh has %d\n",
                    static_cast<int>(totalOwned), static_cast<int>(globalCells));
        rc = 1;
      }
    }
#ifdef VIBEFLOW_HAVE_MPI
    int g = rc; MPI_Allreduce(&g, &rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  }
  Kokkos::finalize();
  return rc;
}
