// ADR-047 gate 1: PETSc on more than one rank.
//
// Two systems on the 4,096-cell vertex-file mesh, partitioned by RCB as
// mms_parallel partitions it: the diffusion operator solved by its own
// non-orthogonal sweep loop, and a nonsymmetric convection-diffusion system
// (the same operator plus first-order upwind fluxes of a uniform velocity)
// solved once. The serial run (P = 1, a one-rank DistributedMesh) writes each
// configuration's solution in global id order; the runs under mpirun -n 2, 3, 4
// read those files and judge themselves, max |du| <= 1e-9 as test_backends.
//
//   (a) at one rank, bicgstab+bjilu against bicgstab+ilu;
//   (b) at P ranks, each configuration against its own serial solution, and
//       bicgstab+bjilu also against serial bicgstab+ilu;
//   (c) describe(), after a solve, shows P blocks, each preonly with ILU at
//       zero fill;
//   (d) on two ranks, a row map with permuted owned rows, and one with a wrong
//       ghost row, each make the constructor throw on every rank.
// Every solve must report converged. Reported: ilu on two ranks, the factor
// options, the iteration counts.
//
// Run:  test_backends_mpi <fixtures> <dir>              (P = 1: writes <dir>)
//       mpirun -n P test_backends_mpi <fixtures> <dir>  (P > 1: reads it)

#include <cstdio>
#ifndef VIBEFLOW_HAVE_PETSC
int main() {
  std::printf("built without PETSc: test_backends_mpi does not apply\n");
  return 77;
}
#else
#include "core/Parallel.hpp"
#include "mesh/RawMesh.hpp"
#include "mesh/DistributedMesh.hpp"
#include "discretization/Diffusion.hpp"
#include "linalg/LinearSystem.hpp"
#include "linalg/PetscSolver.hpp"
#include <petscsys.h>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {

constexpr Real PI = 3.14159265358979323846;
constexpr Real BOUND = 1e-9;
constexpr Real UX = 2.0, UY = 1.0, UZ = 0.5;     // the uniform velocity
Real uExact(Real x, Real y, Real z) { return std::exp(x + 0.5*y) * std::sin(PI*z + 0.3); }
Real fSource(Real x, Real y, Real z) { return (PI*PI - 1.25) * uExact(x, y, z); }
int failures = 0;

// Forwards every call and counts the solves that do not report converged:
// DiffusionOperator::solve discards its reports.
class Counted final : public LinearSolver {
 public:
  explicit Counted(LinearSolver& s) : s_(s) {}
  SolveReport solve(LinearSystem& sys, ScalarField& x, Real relTol, Real absTol,
                    int maxIter) override {
    const SolveReport r = s_.solve(sys, x, relTol, absTol, maxIter);
    if (!r.converged) ++unconverged;
    iterations += r.iterations;
    ++solves;
    return r;
  }
  void notifyMatrixChanged() override { s_.notifyMatrixChanged(); }
  std::string backendName() const override { return s_.backendName(); }
  int unconverged = 0, iterations = 0, solves = 0;

 private:
  LinearSolver& s_;
};

struct Result {
  bool ok = false;           // the solve ran (no exception)
  int unconverged = 0, iterations = 0;
  std::vector<Real> owned;   // the owned cells' values, local order
  std::string what, describe;
};

bool allRanks(const Comm& comm, bool mine) { return comm.min(mine ? 1.0 : 0.0) > 0.5; }

}  // namespace

int main(int argc, char** argv) {
  ParallelScope mpi(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  Kokkos::initialize(argc, argv);
  {
    const Comm comm = Comm::world();
    const int P = comm.size(), rank = comm.rank();
    if (argc < 3) {
      if (rank == 0) std::printf("usage: test_backends_mpi <fixtures> <dir>\n");
      failures = 1;
    } else {
    const std::string fix = argv[1], dir = argv[2];
    RawMesh raw = RawMesh::fromVertexFile(16, fix + "/vertices_n16_s25.txt");
    const Index nGlobal = raw.nCells();
    DistributedMesh mesh(raw, comm, PartitionMethod::RCB);
    const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();
    const Index nf = mesh.nInternalFaces();
    const std::vector<Index>& gid = mesh.globalCellId();

    ScalarField src("src", nt), phiB("phiB", nb);
    {
      auto cc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellCentre());
      auto bcen = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.boundaryCentre());
      auto hs = Kokkos::create_mirror_view(src);
      auto hp = Kokkos::create_mirror_view(phiB);
      for (Index i = 0; i < nt; ++i) hs(i) = fSource(cc(i,0), cc(i,1), cc(i,2));
      for (Index i = 0; i < nb; ++i) hp(i) = uExact(bcen(i,0), bcen(i,1), bcen(i,2));
      Kokkos::deep_copy(src, hs); Kokkos::deep_copy(phiB, hp);
    }
    DiffusionOperator op(mesh, 1.0, comm);

    // The convection-diffusion system: the operator's matrix and source (the
    // deferred correction taken about zero), plus upwind fluxes F = U . S.
    auto convection = [&](LinearSystem& sys) {
      op.assembleMatrix(sys);
      ScalarField zero("zero", nt);
      op.assembleSource(sys, src, phiB, zero);
      auto own = mesh.owner(); auto nei = mesh.neighbour(); auto sf = mesh.faceArea();
      auto bc = mesh.boundaryCell(); auto ba = mesh.boundaryArea();
      auto diag = sys.diag(); auto up = sys.upper(); auto lo = sys.lower(); auto b = sys.source();
      auto pb = phiB;
      Kokkos::parallel_for("upwindInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
        KOKKOS_LAMBDA(const Index f) {
          const Real F = UX*sf(f,0) + UY*sf(f,1) + UZ*sf(f,2);
          Kokkos::atomic_add(&diag(own(f)), F > 0.0 ? F : 0.0);
          Kokkos::atomic_add(&diag(nei(f)), F < 0.0 ? -F : 0.0);
          up(f) += F < 0.0 ? F : 0.0;
          lo(f) += F > 0.0 ? -F : 0.0;
        });
      Kokkos::parallel_for("upwindBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
        KOKKOS_LAMBDA(const Index f) {
          const Real F = UX*ba(f,0) + UY*ba(f,1) + UZ*ba(f,2);
          if (F > 0.0) Kokkos::atomic_add(&diag(bc(f)), F);
          else Kokkos::atomic_add(&b(bc(f)), -F * pb(f));
        });
      Kokkos::fence();
    };

    auto run = [&](bool diffusion, const std::string& cfg) {
      Result r;
      try {
        PetscSolver ps(mesh, comm, cfg);
        ScalarField x("x", nt);
        LinearSystem sys(mesh);
        if (diffusion) {
          Counted c(ps);
          op.solve(sys, c, src, phiB, x);
          r.unconverged = c.unconverged; r.iterations = c.iterations;
        } else {
          convection(sys);
          const SolveReport rep = ps.solve(sys, x, 1e-12, 1e-16, 5000);
          r.unconverged = rep.converged ? 0 : 1; r.iterations = rep.iterations;
        }
        auto h = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), x);
        r.owned.assign(h.data(), h.data() + nc);
        r.describe = ps.describe();
        r.ok = true;
      } catch (const std::exception& e) {
        r.what = e.what();
      }
      return r;
    };
    auto fileOf = [&](bool diffusion, const std::string& cfg) {
      return dir + "/" + (diffusion ? "diffusion_" : "convection_") + cfg + ".txt";
    };
    // max |du| over every rank's owned cells against a serial file.
    auto against = [&](const Result& r, const std::string& file, bool& readOk) {
      std::vector<Real> ref(nGlobal, 0.0);
      std::ifstream in(file);
      Index k = 0;
      for (Real v; k < nGlobal && (in >> v); ++k) ref[k] = v;
      readOk = allRanks(comm, k == nGlobal);
      Real d = 0.0;
      if (r.ok) for (Index c = 0; c < nc; ++c) d = std::max(d, std::abs(r.owned[c] - ref[gid[c]]));
      return comm.max(d);
    };
    auto judge = [&](const char* what, bool ok) {
      if (!ok) ++failures;
      if (rank == 0) std::printf("  %-64s %s\n", what, ok ? "PASS" : "FAIL");
    };

    struct Cfg { bool diffusion; const char* name; };
    const std::vector<Cfg> cfgs = {{true, "cg+jacobi"}, {true, "cg+hypre"},
                                   {false, "bicgstab+jacobi"}, {false, "bicgstab+bjilu"}};
    if (rank == 0) std::printf("test_backends_mpi: %d rank(s), %d cells\n", P, static_cast<int>(nGlobal));

    std::vector<Result> res;
    for (const auto& c : cfgs) {
      res.push_back(run(c.diffusion, c.name));
      const Result& r = res.back();
      const bool ran = allRanks(comm, r.ok);
      const int unconv = static_cast<int>(comm.sum(static_cast<Real>(r.unconverged)));
      if (rank == 0) {
        std::printf("  %-10s %-16s iterations %6d  unconverged %d%s%s\n",
                    c.diffusion ? "diffusion" : "convection", c.name, r.iterations, unconv,
                    ran ? "" : "  threw: ", ran ? "" : r.what.c_str());
      }
      judge((std::string(c.name) + ": every solve converged").c_str(), ran && unconv == 0);
    }

    if (P == 1) {
      // The serial reference, in global id order; then (a).
      for (std::size_t i = 0; i < cfgs.size(); ++i) {
        if (!res[i].ok) continue;
        std::vector<Real> g(nGlobal);
        for (Index c = 0; c < nc; ++c) g[gid[c]] = res[i].owned[c];
        std::FILE* f = std::fopen(fileOf(cfgs[i].diffusion, cfgs[i].name).c_str(), "w");
        for (Real v : g) std::fprintf(f, "%.17g\n", v);
        std::fclose(f);
      }
      const Result ilu = run(false, "bicgstab+ilu");
      if (ilu.ok) {
        std::vector<Real> g(nGlobal);
        for (Index c = 0; c < nc; ++c) g[gid[c]] = ilu.owned[c];
        std::FILE* f = std::fopen(fileOf(false, "bicgstab+ilu").c_str(), "w");
        for (Real v : g) std::fprintf(f, "%.17g\n", v);
        std::fclose(f);
      }
      const Result& bj = res[3];
      Real d = 0.0;
      bool bitwise = bj.ok && ilu.ok;
      if (bj.ok && ilu.ok)
        for (Index c = 0; c < nc; ++c) {
          d = std::max(d, std::abs(bj.owned[c] - ilu.owned[c]));
          bitwise &= bj.owned[c] == ilu.owned[c];
        }
      std::printf("  (a) bjilu against ilu at one rank: max |du| %.3e, bitwise %s, iterations %d and %d\n",
                  d, bitwise ? "yes" : "no", bj.iterations, ilu.iterations);
      judge("(a) bicgstab+bjilu within 1e-9 of bicgstab+ilu at one rank",
            bj.ok && ilu.ok && ilu.unconverged == 0 && d <= BOUND);
      std::printf("  ilu's factor options: %s\n", ilu.describe.c_str());
    } else {
      for (std::size_t i = 0; i < cfgs.size(); ++i) {
        bool readOk = false;
        const Real d = against(res[i], fileOf(cfgs[i].diffusion, cfgs[i].name), readOk);
        if (rank == 0) std::printf("  (b) %-16s against its serial solution: max |du| %.3e\n", cfgs[i].name, d);
        judge((std::string("(b) ") + cfgs[i].name + " within 1e-9 of serial").c_str(),
              readOk && allRanks(comm, res[i].ok) && d <= BOUND);
      }
      bool readOk = false;
      const Real d = against(res[3], fileOf(false, "bicgstab+ilu"), readOk);
      if (rank == 0) std::printf("  (b) bicgstab+bjilu against serial bicgstab+ilu: max |du| %.3e\n", d);
      judge("(b) bicgstab+bjilu within 1e-9 of serial bicgstab+ilu",
            readOk && allRanks(comm, res[3].ok) && d <= BOUND);
    }

    // (c) describe(): P blocks, each preonly with ILU at zero fill.
    {
      const std::string& s = res[3].describe;
      const bool blocks = s.find("blocks=" + std::to_string(P) + " ") != std::string::npos;
      bool sub = s.find("ksp=preonly pc=ilu levels=0") != std::string::npos;
      if (rank == 0) std::printf("  (c) describe() on rank 0: %s\n", s.c_str());
      judge("(c) describe(): P blocks, each preonly ILU at zero fill", allRanks(comm, blocks && sub));
    }

    // (d) on two ranks: a bad row map makes the constructor throw everywhere.
    if (P == 2) {
      const std::vector<Index> good = PetscSolver::rowMap(mesh, comm);
      const bool sized = allRanks(comm, good.size() == static_cast<std::size_t>(nt) && nc >= 2);
      auto throwsEverywhere = [&](std::vector<Index> map) {
        bool threw = false;
        try { PetscSolver ps(mesh, comm, "bicgstab+jacobi", map); }
        catch (const std::exception&) { threw = true; }
        return allRanks(comm, threw);
      };
      bool permuted = false, ghost = false;
      if (sized) {
        std::vector<Index> m = good;
        if (rank == 0) std::swap(m[0], m[1]);
        permuted = throwsEverywhere(m);
        m = good;
        if (rank == 0 && mesh.nGhost() > 0) m[nc] = (m[nc] + 1) % nGlobal;
        ghost = throwsEverywhere(m);
      }
      judge("(d) permuted owned rows: the constructor throws on every rank", sized && permuted);
      judge("(d) a wrong ghost row: the constructor throws on every rank", sized && ghost);

      // Reported: plain ilu on two ranks, expected to be refused or to fail.
      const Result ilu = run(false, "bicgstab+ilu");
      if (rank == 0)
        std::printf("  reported: bicgstab+ilu on two ranks: %s\n",
                    ilu.ok ? "ran" : ("refused: " + ilu.what).c_str());
    }

    if (rank == 0) std::printf("\nparallel backends gate (P = %d): %s\n", P, failures ? "FAIL" : "PASS");
    }
#ifdef VIBEFLOW_HAVE_MPI
    int g = failures; MPI_Allreduce(&g, &failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  }
  Kokkos::finalize();
  PetscFinalize();
  return failures ? 1 : 0;
}
#endif
