// Parallel-consistency gate for the k-omega SST model (ADR-042, gate 3).
//
// The model adds fields read at ghost cells -- k, omega, their gradients,
// nu_t and its face interpolation, the diffusivities -- and one global
// operation, the wall distance, which gathers every rank's wall faces. A
// missing exchange converges to a slightly different answer on more than one
// rank, so, as mms_parallel_ns and mms_parallel_heat do, the comparison is
// against the serial run, to ten digits: u, k and omega, each.
//
// The problem is gate 2b's coupled MS-A (SST-2003) on a distorted mesh: the
// wall distance measured from y = 0, slip on the x faces (the MS-A velocity
// meets the slip condition there), the exact velocity elsewhere, k and omega
// prescribed on every face. Two steps with fixed iteration counts, so the
// serial and parallel runs perform the same sequence of operations.
//
// Run:  mpirun -n <N> mms_parallel_sst [reference "L2u,L2k,L2w"]

#include "core/Parallel.hpp"
#include "mesh/RawMesh.hpp"
#include "mesh/DistributedMesh.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#include "sst_ms.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

using namespace vibeflow;
using namespace vibeflow::sst_ms;

int main(int argc, char** argv) {
  ParallelScope mpi(argc, argv);
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const Comm comm = Comm::world();
    std::vector<Real> ref;
    if (argc > 1) {
      std::string r(argv[1]);
      std::size_t pos = 0;
      while (pos <= r.size()) {
        const std::size_t comma = r.find(',', pos);
        ref.push_back(std::atof(r.substr(pos, comma - pos).c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
    }
    const Index n = 8;
    const Real nu = nu_A, dt = 0.5;
    const int steps = 2;

    RawMesh raw = RawMesh::generate(n, 0.3, "smooth");
    const Index globalCells = raw.nCells();
    DistributedMesh mesh(raw, comm, PartitionMethod::RCB);
    const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();
    const Index nf = mesh.nInternalFaces();

    PisoControls ctl;
    ctl.outer = 2;
    ctl.outerTol = 0.0;              // never break early
    ctl.nonOrthCorrectors = 4;       // fixed counts: see mms_parallel_ns
    ctl.nonOrthTol = 0.0;

    auto H = [](const auto& v) {
      return Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
    };
    auto hcc = H(mesh.cellCentre());
    auto hbc = H(mesh.boundaryCentre());
    auto hba = H(mesh.boundaryArea());
    auto hfc = H(mesh.faceCentre());
    auto hfa = H(mesh.faceArea());

    View1<int> uType("uType", nb), wall("wall", nb), kind("kind", nb);
    VectorField ub("ub", nb, 3), u0("u0", nt, 3), src("src", nt, 3);
    ScalarField fb("fb", nb), p0("p0", nt), F0("F0", nf);
    ScalarField k0("k0", nt), w0("w0", nt), kS("kS", nt), wS("wS", nt);
    ScalarField kB("kB", nb), wB("wB", nb);
    auto hut = Kokkos::create_mirror_view(uType);
    auto hwall = Kokkos::create_mirror_view(wall);
    auto hub = Kokkos::create_mirror_view(ub);
    auto hfb = Kokkos::create_mirror_view(fb);
    auto hu0 = Kokkos::create_mirror_view(u0);
    auto hsrc = Kokkos::create_mirror_view(src);
    auto hF0 = Kokkos::create_mirror_view(F0);
    auto hk0 = Kokkos::create_mirror_view(k0); auto hw0 = Kokkos::create_mirror_view(w0);
    auto hkS = Kokkos::create_mirror_view(kS); auto hwS = Kokkos::create_mirror_view(wS);
    auto hkB = Kokkos::create_mirror_view(kB); auto hwB = Kokkos::create_mirror_view(wB);

    for (Index c = 0; c < nt; ++c) {
      const Real x = hcc(c,0), y = hcc(c,1), z = hcc(c,2);
      hu0(c,0) = u0_A(x, y, z); hu0(c,1) = u1_A(x, y, z); hu0(c,2) = u2_A(x, y, z);
      hsrc(c,0) = srcu0_A_2003(x, y, z); hsrc(c,1) = srcu1_A_2003(x, y, z);
      hsrc(c,2) = srcu2_A_2003(x, y, z);
      hk0(c) = k_A(x, y, z); hw0(c) = w_A(x, y, z);
      hkS(c) = srck_A_2003(x, y, z); hwS(c) = srcw_A_2003(x, y, z);
    }
    Real fsum = 0.0, asum = 0.0;
    for (Index f = 0; f < nb; ++f) {
      const Real x = hbc(f,0), y = hbc(f,1), z = hbc(f,2);
      const Real mag = std::sqrt(hba(f,0)*hba(f,0) + hba(f,1)*hba(f,1) + hba(f,2)*hba(f,2));
      const bool xFace = std::abs(hba(f,0)) / mag > 0.99;
      const bool yLow = hba(f,1) / mag < -0.99;          // the wall, y = 0
      const Real v0 = u0_A(x, y, z), v1 = u1_A(x, y, z), v2 = u2_A(x, y, z);
      hut(f) = static_cast<int>(xFace ? VelocityBC::Slip : VelocityBC::Dirichlet);
      hub(f,0) = xFace ? 0.0 : v0; hub(f,1) = xFace ? 0.0 : v1; hub(f,2) = xFace ? 0.0 : v2;
      hfb(f) = xFace ? 0.0 : v0*hba(f,0) + v1*hba(f,1) + v2*hba(f,2);
      if (!xFace) { fsum += hfb(f); asum += mag; }
      hwall(f) = yLow ? 1 : 0;
      hkB(f) = k_A(x, y, z); hwB(f) = w_A(x, y, z);
    }
    fsum = comm.sum(fsum); asum = comm.sum(asum);
    for (Index f = 0; f < nb; ++f) {
      if (hut(f) == static_cast<int>(VelocityBC::Slip)) continue;
      const Real mag = std::sqrt(hba(f,0)*hba(f,0) + hba(f,1)*hba(f,1) + hba(f,2)*hba(f,2));
      hfb(f) -= fsum * mag / asum;
    }
    for (Index f = 0; f < nf; ++f) {
      const Real x = hfc(f,0), y = hfc(f,1), z = hfc(f,2);
      hF0(f) = u0_A(x, y, z)*hfa(f,0) + u1_A(x, y, z)*hfa(f,1) + u2_A(x, y, z)*hfa(f,2);
    }
    Kokkos::deep_copy(uType, hut); Kokkos::deep_copy(wall, hwall);
    Kokkos::deep_copy(ub, hub); Kokkos::deep_copy(fb, hfb);
    Kokkos::deep_copy(u0, hu0); Kokkos::deep_copy(src, hsrc); Kokkos::deep_copy(F0, hF0);
    Kokkos::deep_copy(k0, hk0); Kokkos::deep_copy(w0, hw0);
    Kokkos::deep_copy(kS, hkS); Kokkos::deep_copy(wS, hwS);
    Kokkos::deep_copy(kB, hkB); Kokkos::deep_copy(wB, hwB);

    Real l2[3] = {0.0, 0.0, 0.0}, cont = 0.0;
    try {
      PisoSolver solver(mesh, nu, dt, ctl, comm);
      solver.setBoundaryTypes(uType);
      solver.setState(u0, p0, F0);
      TurbulenceModel tm;
      tm.secondOrderAdvection = true;   // gate 2b's configuration
      solver.enableTurbulence(tm, wall);
      solver.setTurbulenceBoundary(kind, kB, wB);       // kind: all Dirichlet
      solver.setTurbulenceSource(kS, wS);
      solver.setTurbulence(k0, w0);

      NativeBiCGStab momentum(mesh, comm);
      NativeCG pressure(mesh, comm, true);
      for (int k = 0; k < steps; ++k)
        cont = solver.advance(ub, fb, src, momentum, pressure).continuityError;

      auto u = solver.velocity(); auto kk = solver.turbulentKineticEnergy();
      auto ww = solver.specificDissipation(); auto vol = mesh.cellVolume();
      auto ue = u0; auto ke = k0; auto we = w0;
      Real s[3] = {0.0, 0.0, 0.0}, den = 0.0;
      Kokkos::parallel_reduce("l2u", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c, Real& a) {
          Real e2 = 0.0;
          for (int d = 0; d < 3; ++d) { const Real e = u(c,d) - ue(c,d); e2 += e*e; }
          a += e2 * vol(c);
        }, s[0]);
      Kokkos::parallel_reduce("l2k", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c, Real& a) { const Real e = kk(c) - ke(c); a += e*e*vol(c); }, s[1]);
      Kokkos::parallel_reduce("l2w", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c, Real& a) { const Real e = ww(c) - we(c); a += e*e*vol(c); }, s[2]);
      Kokkos::parallel_reduce("vol", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c, Real& a) { a += vol(c); }, den);
      den = comm.sum(den);
      for (int i = 0; i < 3; ++i) l2[i] = std::sqrt(comm.sum(s[i]) / den);
    } catch (const std::exception& e) {
      if (comm.rank() == 0) std::printf("  FAIL: %s\n", e.what());
      rc = 1;
    }
    const Index gcells = comm.sum(nc);
    if (rc == 0 && comm.rank() == 0) {
      std::printf("  ranks %2d   cells %6d   ghosts(r0) %5d   div %.1e   L2 %.14e,%.14e,%.14e",
                  comm.size(), static_cast<int>(gcells), static_cast<int>(mesh.nGhost()),
                  cont, l2[0], l2[1], l2[2]);
      if (ref.size() == 3) {
        bool pass = true;
        std::printf("   rel");
        const char* names[3] = {"u", "k", "w"};
        for (int i = 0; i < 3; ++i) {
          const Real r = std::abs(l2[i] - ref[i]) / ref[i];
          pass &= r < 1e-10;
          std::printf(" %s %.2e", names[i], r);
        }
        std::printf("  %s", pass ? "PASS" : "FAIL");
        rc = pass ? 0 : 1;
      }
      std::printf("\n");
    }
    if (gcells != globalCells) {
      if (comm.rank() == 0)
        std::printf("  FAIL: partition lost cells (%d of %d)\n",
                    static_cast<int>(gcells), static_cast<int>(globalCells));
      rc = 1;
    }
#ifdef VIBEFLOW_HAVE_MPI
    int g = rc; MPI_Allreduce(&g, &rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  }
  Kokkos::finalize();
  return rc;
}
