// Parallel-consistency gate for the full Navier-Stokes solver.
//
// The existing parallel gate covers the diffusion operator only. That was the
// whole of v0; the PISO solver added six more fields that are read at ghost
// cells -- the velocity, the momentum diagonal, H/aP, the pressure, and the
// two gradients built from them -- and not one of them is checked by any gate
// in the suite.
//
// A missing halo exchange does not crash and does not diverge. It converges to
// a slightly different answer, which on one rank is the right answer and on
// four ranks is not, and nothing else in the suite can tell the difference:
// an order study run on four ranks would report a clean second order around
// the wrong solution. So the comparison here is not against the exact solution
// but against the serial run, to ten digits.
//
// Run:  mpirun -n <N> mms_parallel_ns [reference L2]

#include "core/Parallel.hpp"
#include "mesh/RawMesh.hpp"
#include "mesh/DistributedMesh.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace nsflow;

namespace {

constexpr Real PI = 3.14159265358979323846;
constexpr Real A = PI / 4.0;
constexpr Real D = PI / 2.0;

Vec3 velocity(const Vec3& p, Real t, Real nu) {
  const Real e = std::exp(-D * D * nu * t);
  return {-A * (std::exp(A*p.x) * std::sin(A*p.y + D*p.z)
                + std::exp(A*p.z) * std::cos(A*p.x + D*p.y)) * e,
          -A * (std::exp(A*p.y) * std::sin(A*p.z + D*p.x)
                + std::exp(A*p.x) * std::cos(A*p.y + D*p.z)) * e,
          -A * (std::exp(A*p.z) * std::sin(A*p.x + D*p.y)
                + std::exp(A*p.y) * std::cos(A*p.z + D*p.x)) * e};
}

// Spread the quadrature's boundary-flux imbalance over the boundary, so the
// pure-Neumann pressure problem is solvable. The serial version of this in
// ethier_steinman.cpp sums over one rank's faces; here the sum has to be
// global, or each rank adjusts by a different amount and the two runs stop
// solving the same problem -- which is itself the class of bug this gate is
// for.
void adjustBoundaryFlux(const Mesh& mesh, const Comm& comm, ScalarField& fb) {
  auto bar = mesh.boundaryArea();
  Real total = 0.0, areaSum = 0.0;
  Kokkos::parallel_reduce("fbSum", Kokkos::RangePolicy<ExecSpace>(0, mesh.nBoundaryFaces()),
    KOKKOS_LAMBDA(const Index f, Real& a) { a += fb(f); }, total);
  Kokkos::parallel_reduce("areaSum", Kokkos::RangePolicy<ExecSpace>(0, mesh.nBoundaryFaces()),
    KOKKOS_LAMBDA(const Index f, Real& a) {
      a += Kokkos::sqrt(bar(f,0)*bar(f,0) + bar(f,1)*bar(f,1) + bar(f,2)*bar(f,2));
    }, areaSum);
  total = comm.sum(total);
  areaSum = comm.sum(areaSum);
  Kokkos::parallel_for("fbAdj", Kokkos::RangePolicy<ExecSpace>(0, mesh.nBoundaryFaces()),
    KOKKOS_LAMBDA(const Index f) {
      const Real a = Kokkos::sqrt(bar(f,0)*bar(f,0) + bar(f,1)*bar(f,1) + bar(f,2)*bar(f,2));
      fb(f) -= total * a / areaSum;
    });
  Kokkos::fence();
}

}  // namespace

int main(int argc, char** argv) {
  ParallelScope mpi(argc, argv);
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const Comm comm = Comm::world();
    const Real ref = argc > 1 ? std::atof(argv[1]) : -1.0;
    const Index n = 8;
    const Real nu = 0.5, dt = 2e-3;
    const int steps = 2;

    // A distorted mesh, so the non-orthogonal and skewness corrections -- both
    // of which read gradients at ghost cells -- are actually exercised. On a
    // Cartesian mesh they are identically zero and a wrong ghost gradient
    // costs nothing.
    RawMesh raw = RawMesh::generate(n, 0.3, "smooth");
    const Index globalCells = raw.nCells();
    DistributedMesh mesh(raw, comm, PartitionMethod::RCB);

    const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();
    const Index nf = mesh.nInternalFaces();

    PisoControls ctl;
    ctl.outer = 2;
    ctl.outerTol = 0.0;              // never break early
    // A FIXED sweep count, not a converged one. A tolerance makes the two runs
    // take different numbers of sweeps whenever the residual lands either side
    // of it, and then a difference in the answer could be blamed on that
    // rather than on the halo. With the count fixed, the serial and parallel
    // runs perform literally the same sequence of operations, and any
    // difference is the thing this gate is looking for.
    ctl.nonOrthCorrectors = 4;
    ctl.nonOrthTol = 0.0;
    PisoSolver solver(mesh, nu, dt, ctl, comm);

    // Dirichlet velocity everywhere: a closed domain, so the pressure operator
    // is singular and the null-space projection runs -- another path that has
    // global reductions in it.
    View1<int> uType("uType", nb);
    VectorField ub("ub", nb, 3), u0("u0", nt, 3), src("src", nt, 3);
    ScalarField fb("fb", nb), p0("p0", nt), F0("F0", nf);

    auto hcc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellCentre());
    auto hbc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.boundaryCentre());
    auto hba = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.boundaryArea());
    auto hfc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.faceCentre());
    auto hfa = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.faceArea());
    auto hu0 = Kokkos::create_mirror_view(u0);
    auto hub = Kokkos::create_mirror_view(ub);
    auto hfb = Kokkos::create_mirror_view(fb);
    auto hF0 = Kokkos::create_mirror_view(F0);

    for (Index c = 0; c < nt; ++c) {
      const Vec3 v = velocity({hcc(c,0), hcc(c,1), hcc(c,2)}, 0.0, nu);
      hu0(c,0) = v.x; hu0(c,1) = v.y; hu0(c,2) = v.z;
    }
    for (Index f = 0; f < nb; ++f) {
      const Vec3 v = velocity({hbc(f,0), hbc(f,1), hbc(f,2)}, 0.0, nu);
      hub(f,0) = v.x; hub(f,1) = v.y; hub(f,2) = v.z;
      hfb(f) = v.x*hba(f,0) + v.y*hba(f,1) + v.z*hba(f,2);
    }
    for (Index f = 0; f < nf; ++f) {
      const Vec3 v = velocity({hfc(f,0), hfc(f,1), hfc(f,2)}, 0.0, nu);
      hF0(f) = v.x*hfa(f,0) + v.y*hfa(f,1) + v.z*hfa(f,2);
    }
    Kokkos::deep_copy(u0, hu0); Kokkos::deep_copy(ub, hub);
    Kokkos::deep_copy(fb, hfb); Kokkos::deep_copy(F0, hF0);
    adjustBoundaryFlux(mesh, comm, fb);

    solver.setBoundaryTypes(uType);          // all zero = all Dirichlet
    solver.setState(u0, p0, F0);

    NativeBiCGStab momentum(mesh, comm);
    NativeCG pressure(mesh, comm, true);     // singular: project the null space

    Real cont = 0.0;
    for (int k = 0; k < steps; ++k)
      cont = solver.advance(ub, fb, src, momentum, pressure).continuityError;

    // L2 velocity error against the exact solution, volume weighted, reduced
    // over OWNED cells and summed across ranks.
    VectorField ex("ex", nt, 3);
    auto hex = Kokkos::create_mirror_view(ex);
    for (Index c = 0; c < nt; ++c) {
      const Vec3 v = velocity({hcc(c,0), hcc(c,1), hcc(c,2)}, steps * dt, nu);
      hex(c,0) = v.x; hex(c,1) = v.y; hex(c,2) = v.z;
    }
    Kokkos::deep_copy(ex, hex);
    auto u = solver.velocity(); auto vol = mesh.cellVolume();
    Real num = 0.0, den = 0.0;
    Kokkos::parallel_reduce("l2", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& a) {
        Real s = 0.0;
        for (int d = 0; d < 3; ++d) { const Real e = u(c,d) - ex(c,d); s += e*e; }
        a += s * vol(c);
      }, num);
    Kokkos::parallel_reduce("vol", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c, Real& a) { a += vol(c); }, den);
    const Real l2 = std::sqrt(comm.sum(num) / comm.sum(den));
    const Index gcells = comm.sum(nc);

    if (comm.rank() == 0) {
      std::printf("  ranks %2d   cells %6d   ghosts(r0) %5d   div %.1e   L2 %.14e",
                  comm.size(), static_cast<int>(gcells),
                  static_cast<int>(mesh.nGhost()), cont, l2);
      if (ref > 0.0) {
        const Real rel = std::abs(l2 - ref) / ref;
        std::printf("   rel %.2e  %s", rel, rel < 1e-10 ? "PASS" : "FAIL");
        rc = rel < 1e-10 ? 0 : 1;
      }
      std::printf("\n");
    }
    if (gcells != globalCells) {
      if (comm.rank() == 0)
        std::printf("  FAIL: partition lost cells (%d of %d)\n",
                    static_cast<int>(gcells), static_cast<int>(globalCells));
      rc = 1;
    }
#ifdef NSFLOW_HAVE_MPI
    int g = rc; MPI_Allreduce(&g, &rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  }
  Kokkos::finalize();
  return rc;
}
