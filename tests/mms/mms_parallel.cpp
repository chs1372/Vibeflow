// Parallel-consistency gate: the answer must not depend on how many ranks
// solved it.
//
// This is the one gate that catches a wrong halo exchange. A missing or
// mis-ordered exchange still converges and still looks like a plausible
// solution -- it just quietly gives a different answer on 4 ranks than on 1.
// So the comparison is not "close to the exact solution" (which would pass
// anyway at this mesh size) but "bitwise close to the serial run".
//
// Run:  mpirun -n <N> mms_parallel <fixtures> [reference L2]
// With a reference value it exits non-zero on mismatch, so a script can sweep
// rank counts.

#include "core/Parallel.hpp"
#include "mesh/HexMesh.hpp"
#include "mesh/DistributedMesh.hpp"
#include "discretization/Diffusion.hpp"
#include "linalg/LinearSystem.hpp"
#include "linalg/NativeCG.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace nsflow;

namespace {

constexpr Real PI = 3.14159265358979323846;
Real uExact(Real x, Real y, Real z) { return std::exp(x + 0.5*y) * std::sin(PI*z + 0.3); }
Real fSource(Real x, Real y, Real z) { return (PI*PI - 1.25) * uExact(x, y, z); }

}  // namespace

int main(int argc, char** argv) {
  ParallelScope mpi(argc, argv);
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const Comm comm = Comm::world();
    const std::string dir = argc > 1 ? argv[1] : "tests/fixtures";
    const Real ref = argc > 2 ? std::atof(argv[2]) : -1.0;
    const Index n = 16;

    // v0 partitioner: every rank reads the whole mesh and keeps its slice.
    HexMesh global(HexMesh::fromVertexFile(n, dir + "/vertices_n" + std::to_string(n) + "_s25.txt"));
    DistributedMesh mesh(global, comm, PartitionMethod::RCB);

    const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();
    ScalarField src("src", nt), phiB("phiB", nb), phi("phi", nt), exact("exact", nt);
    auto h_cc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellCentre());
    auto h_bc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.boundaryCentre());
    auto h_s = Kokkos::create_mirror_view(src);
    auto h_e = Kokkos::create_mirror_view(exact);
    auto h_p = Kokkos::create_mirror_view(phiB);
    for (Index i = 0; i < nt; ++i) {
      h_s(i) = fSource(h_cc(i,0), h_cc(i,1), h_cc(i,2));
      h_e(i) = uExact(h_cc(i,0), h_cc(i,1), h_cc(i,2));
    }
    for (Index i = 0; i < nb; ++i) h_p(i) = uExact(h_bc(i,0), h_bc(i,1), h_bc(i,2));
    Kokkos::deep_copy(src, h_s); Kokkos::deep_copy(exact, h_e); Kokkos::deep_copy(phiB, h_p);

    DiffusionOperator op(mesh, 1.0, comm);
    LinearSystem sys(mesh);
    NativeCG cg(mesh, comm);
    const int sweeps = op.solve(sys, cg, src, phiB, phi);

    auto vol = mesh.cellVolume();
    Real num = 0.0, den = 0.0;
    Kokkos::parallel_reduce("l2", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index i, Real& a) {
        const Real e = phi(i) - exact(i); a += e * e * vol(i);
      }, num);
    Kokkos::parallel_reduce("vol", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index i, Real& a) { a += vol(i); }, den);
    const Real l2 = std::sqrt(comm.sum(num) / comm.sum(den));
    const Index gcells = comm.sum(nc);

    if (comm.rank() == 0) {
      std::printf("  ranks %2d   cells %6d   ghosts(r0) %5d   sweeps %2d   L2 %.14e",
                  comm.size(), static_cast<int>(gcells), static_cast<int>(mesh.nGhost()),
                  sweeps, l2);
      if (ref > 0.0) {
        const Real rel = std::abs(l2 - ref) / ref;
        std::printf("   rel %.2e  %s", rel, rel < 1e-10 ? "PASS" : "FAIL");
        rc = rel < 1e-10 ? 0 : 1;
      }
      std::printf("\n");
    }
    if (gcells != global.nCells()) {
      if (comm.rank() == 0)
        std::printf("  FAIL: partition lost cells (%d of %d)\n",
                    static_cast<int>(gcells), static_cast<int>(global.nCells()));
      rc = 1;
    }
#ifdef NSFLOW_HAVE_MPI
    int g = rc; MPI_Allreduce(&g, &rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  }
  Kokkos::finalize();
  return rc;
}
