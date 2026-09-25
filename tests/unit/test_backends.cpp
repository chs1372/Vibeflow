// Backend-equivalence gate: every linear-solver backend must produce the same
// solution of the same system.
//
// A preconditioner is not supposed to change the answer, only how fast it is
// reached. If BoomerAMG and Jacobi-CG disagree beyond the requested tolerance,
// the matrix handed to PETSc is not the matrix the native path solves -- an
// assembly bug, not a solver one. That is exactly the failure this catches.

#include "core/Parallel.hpp"
#include "mesh/HexMesh.hpp"
#include "discretization/Diffusion.hpp"
#include "linalg/LinearSystem.hpp"
#include "linalg/NativeCG.hpp"
#ifdef VIBEFLOW_HAVE_PETSC
#include "linalg/PetscSolver.hpp"
#include <petscsys.h>
#endif
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {
constexpr Real PI = 3.14159265358979323846;
Real uExact(Real x, Real y, Real z) { return std::exp(x + 0.5*y) * std::sin(PI*z + 0.3); }
Real fSource(Real x, Real y, Real z) { return (PI*PI - 1.25) * uExact(x, y, z); }
int failures = 0;
}  // namespace

int main(int argc, char** argv) {
  ParallelScope mpi(argc, argv);
#ifdef VIBEFLOW_HAVE_PETSC
  PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif
  Kokkos::initialize(argc, argv);
  {
    const Comm comm = Comm::world();
    const std::string dir = argc > 1 ? argv[1] : "tests/fixtures";
    const Index n = 16;
    auto mesh = HexMesh::fromVertexFile(n, dir + "/vertices_n16_s25.txt");
    const Index nc = mesh.nCells(), nb = mesh.nBoundaryFaces();

    ScalarField src("src", nc), phiB("phiB", nb), exact("exact", nc);
    auto h_cc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellCentre());
    auto h_bc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.boundaryCentre());
    auto h_s = Kokkos::create_mirror_view(src);
    auto h_e = Kokkos::create_mirror_view(exact);
    auto h_p = Kokkos::create_mirror_view(phiB);
    for (Index i = 0; i < nc; ++i) {
      h_s(i) = fSource(h_cc(i,0), h_cc(i,1), h_cc(i,2));
      h_e(i) = uExact(h_cc(i,0), h_cc(i,1), h_cc(i,2));
    }
    for (Index i = 0; i < nb; ++i) h_p(i) = uExact(h_bc(i,0), h_bc(i,1), h_bc(i,2));
    Kokkos::deep_copy(src, h_s); Kokkos::deep_copy(exact, h_e); Kokkos::deep_copy(phiB, h_p);

    DiffusionOperator op(mesh, 1.0, comm);

    struct Run { std::string name; std::vector<Real> u; int iters; Real secs; Real l2; };
    std::vector<Run> runs;

    auto doRun = [&](const std::string& name, LinearSolver& s) {
      LinearSystem sys(mesh);
      ScalarField phi("phi", nc);
      op.solve(sys, s, src, phiB, phi);
      auto h = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), phi);
      auto vol = mesh.cellVolume();
      Real num = 0.0, den = 0.0;
      Kokkos::parallel_reduce("l2", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index i, Real& a) {
          const Real e = phi(i) - exact(i); a += e * e * vol(i);
        }, num);
      Kokkos::parallel_reduce("vol", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index i, Real& a) { a += vol(i); }, den);
      // Report the LAST linear solve's cost, which is representative.
      LinearSystem probe(mesh);
      op.assembleMatrix(probe);
      op.assembleSource(probe, src, phiB, phi);
      ScalarField y("y", nc);
      const auto rep = s.solve(probe, y, 1e-12, 1e-16, 5000);
      runs.push_back({name, std::vector<Real>(h.data(), h.data() + nc),
                      rep.iterations, rep.wallSeconds, std::sqrt(num / den)});
    };

    NativeCG native(mesh, comm);
    doRun(native.backendName(), native);

#ifdef VIBEFLOW_HAVE_PETSC
    std::printf("PETSc built with hypre: %s\n", PetscSolver::hasHypre() ? "yes" : "no");
    for (const char* cfg : {"cg+jacobi", "cg+ilu", "cg+hypre", "gmres+hypre"}) {
      if (std::string(cfg).find("hypre") != std::string::npos && !PetscSolver::hasHypre())
        continue;
      PetscSolver ps(mesh, comm, cfg);
      doRun(ps.backendName(), ps);
    }
#else
    std::printf("built without PETSc -- only the native backend is compared\n");
#endif

    std::printf("\n%d cells, same system solved by every backend\n", static_cast<int>(nc));
    std::printf("  %-24s %8s %10s %16s %14s\n", "backend", "iters", "sec", "L2 vs exact", "max|du| vs cg");
    const auto& ref = runs[0];
    for (const auto& r : runs) {
      Real d = 0.0;
      for (Index i = 0; i < nc; ++i) d = std::max(d, std::abs(r.u[i] - ref.u[i]));
      const bool ok = d < 1e-9;
      if (!ok) ++failures;
      std::printf("  %-24s %8d %10.4f %16.9e %14.2e %s\n",
                  r.name.c_str(), r.iters, r.secs, r.l2, d, ok ? "" : "FAIL");
    }
    std::printf("\nbackend equivalence gate: %s\n", failures ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
#ifdef VIBEFLOW_HAVE_PETSC
  PetscFinalize();
#endif
  return failures ? 1 : 0;
}
