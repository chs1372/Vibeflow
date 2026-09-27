// Distributed-read gate: each rank must read only its share of the mesh file,
// and the answer must not care.
//
// ADR-023 made each rank BUILD only its own subdomain, and said plainly what
// it did not fix: every rank still read the whole mesh description, so the
// largest mesh the solver could take was still the largest one a single
// rank could hold. This gate measures that directly. Each rank reports the
// peak bytes of description it held while building -- points at 24 bytes,
// cells at 64, as in the file -- and the worst rank must stay under
// (1/P + 0.35) of the file, the same bar the build-memory gate uses. Holding
// the whole file on every rank scores 1.00.
//
// A reader that skipped cells would pass that easily, so the same run solves
// the manufactured diffusion problem of mms_parallel on the mesh it read. The
// L2 error has to match the serial run to 1e-10: the file reader partitions by
// Morton order, not by RCB, so the partitions differ from every other gate's,
// and only the answer is held fixed.
//
// Run:  mpirun -n <N> mms_parallel_read <fixtures> <n> <read|replicated> [ref L2]
//   n = 16 uses the skewed 16^3 fixture of mms_parallel; any other n generates
//   the smooth n^3 family. "replicated" is the RawMesh path, kept so the gate
//   can be shown to fail on it.

#include "core/Parallel.hpp"
#include "mesh/RawMesh.hpp"
#include "mesh/DistributedMesh.hpp"
#include "discretization/Diffusion.hpp"
#include "linalg/LinearSystem.hpp"
#include "linalg/NativeCG.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

using namespace vibeflow;

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
    const Index n = argc > 2 ? std::atoi(argv[2]) : 16;
    const std::string mode = argc > 3 ? argv[3] : "read";
    const Real ref = argc > 4 ? std::atof(argv[4]) : -1.0;

    // The file lives next to the binary, inside the build tree.
    std::string self = argv[0];
    const auto slash = self.find_last_of('/');
    const std::string path = (slash == std::string::npos ? std::string(".") : self.substr(0, slash))
                           + "/mms_parallel_read_" + std::to_string(n) + ".vmesh";
    long long fileBytes = 0;
    if (comm.rank() == 0) {
      const RawMesh raw = n == 16
          ? RawMesh::fromVertexFile(n, dir + "/vertices_n16_s25.txt")
          : RawMesh::generate(n, 0.3, "smooth");
      vmesh::write(path, raw);
    }
    comm.barrier();
    {
      const auto h = vmesh::readHeader(path);
      fileBytes = h.nPoints * 24 + h.nCells * 64;
    }

    std::unique_ptr<DistributedMesh> meshPtr;
    if (mode == "replicated") {
      const RawMesh raw = RawMesh::fromBinary(path);
      meshPtr = std::make_unique<DistributedMesh>(raw, comm, PartitionMethod::RCB);
    } else {
      meshPtr = std::make_unique<DistributedMesh>(path, comm);
    }
    const DistributedMesh& mesh = *meshPtr;
    const Index gcells = comm.sum(mesh.nCells());

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
    op.solve(sys, cg, src, phiB, phi);

    auto vol = mesh.cellVolume();
    Real num = 0.0, den = 0.0;
    Kokkos::parallel_reduce("l2", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index i, Real& a) {
        const Real e = phi(i) - exact(i); a += e * e * vol(i);
      }, num);
    Kokkos::parallel_reduce("vol", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index i, Real& a) { a += vol(i); }, den);
    const Real volume = comm.sum(den);
    const Real l2 = std::sqrt(comm.sum(num) / volume);
    const long long worst = comm.max(mesh.descriptionBytes());

    if (comm.rank() == 0) {
      const Real ratio = static_cast<Real>(worst) / static_cast<Real>(fileBytes);
      // One rank holds everything by definition; the bar applies from two up.
      const Real allowed = 1.0 / comm.size() + 0.35;
      const bool memOk = !comm.parallel() || ratio <= allowed + 1e-12;
      std::printf("  ranks %2d   %-10s  cells %6d   description held %.3f of the file "
                  "(allowed %.3f) %s   L2 %.14e",
                  comm.size(), mode.c_str(), static_cast<int>(gcells), ratio, allowed,
                  memOk ? "ok" : "TOO MUCH", l2);
      if (!memOk) rc = 1;
      if (ref > 0.0) {
        const Real rel = std::abs(l2 - ref) / ref;
        std::printf("   rel %.2e", rel);
        if (rel >= 1e-10) rc = 1;
      }
      std::printf("   %s\n", rc == 0 ? "PASS" : "FAIL");
      if (std::abs(volume - 1.0) > 1e-12) {
        std::printf("  FAIL: cell volumes sum to %.15f, not 1\n", volume);
        rc = 1;
      }
    }
    if (gcells != (n == 16 ? 4096 : n * n * n)) {
      if (comm.rank() == 0)
        std::printf("  FAIL: %d cells read, the mesh has %d\n", static_cast<int>(gcells),
                    static_cast<int>(n * n * n));
      rc = 1;
    }
#ifdef VIBEFLOW_HAVE_MPI
    int g = rc; MPI_Allreduce(&g, &rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  }
  Kokkos::finalize();
  return rc;
}
