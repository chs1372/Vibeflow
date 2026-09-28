// Parallel-consistency gate for the energy equation, the buoyancy coupling
// and slip walls (ADR-038, ADR-039).
//
// v2a adds fields that are read at ghost cells -- the temperature, its
// gradient, the energy matrix diagonal -- and a boundary type whose value is
// rebuilt from the cell velocity every outer iteration. A missing exchange of
// any of them converges to a slightly different answer on more than one rank,
// and an order study run on four ranks would report a clean second order
// around it. So, as mms_parallel_ns does for v1, the comparison is against the
// serial run, to ten digits: the velocity and the temperature, each.
//
// The problem is gate 2's steady manufactured Boussinesq flow on a distorted
// mesh, with a mixture of every boundary type: slip on the x faces (the
// velocity meets the slip condition there), prescribed velocity elsewhere; a
// prescribed heat flux on the x faces, a prescribed temperature elsewhere.
// Two steps with fixed iteration counts, so the serial and parallel runs
// perform the same sequence of operations.
//
// Runs in the balanced buoyancy form (ADR-041), whose hydrostatic pressure is
// one more field read at ghost cells; VIBEFLOW_BUOYANCY=cell selects the cell
// force.
//
// Run:  mpirun -n <N> mms_parallel_heat [reference "L2u,L2T"]

#include "core/Parallel.hpp"
#include "mesh/RawMesh.hpp"
#include "mesh/DistributedMesh.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

using namespace vibeflow;

namespace {

constexpr Real PI = 3.14159265358979323846;

Vec3 velocity(const Vec3& q) {
  const Real sx = std::sin(PI*q.x), sy = std::sin(PI*q.y), sz = std::sin(PI*q.z);
  const Real cx = std::cos(PI*q.x), cy = std::cos(PI*q.y), cz = std::cos(PI*q.z);
  return {sx * (cy - cz), sy * (cz - cx), sz * (cx - cy)};
}

Real temperature(const Vec3& q) {
  return 1.0 + 0.5 * std::sin(PI*q.x) * std::sin(PI*q.y) * std::sin(PI*q.z);
}

// Gate 2's sources (heat_transfer.cpp), betaG = (0, 0, -1), T_ref = 0.
Vec3 momentumSource(const Vec3& q, Real nu) {
  const Vec3 u = velocity(q);
  const Real uu[3] = {u.x, u.y, u.z};
  const Real x[3] = {q.x, q.y, q.z};
  Real g[3][3];
  for (int i = 0; i < 3; ++i) {
    const int j = (i + 1) % 3, k = (i + 2) % 3;
    const Real si = std::sin(PI*x[i]), ci = std::cos(PI*x[i]);
    const Real sj = std::sin(PI*x[j]), cj = std::cos(PI*x[j]);
    const Real sk = std::sin(PI*x[k]), ck = std::cos(PI*x[k]);
    g[i][i] = PI * ci * (cj - ck);
    g[i][j] = -PI * si * sj;
    g[i][k] = PI * si * sk;
  }
  Real s[3];
  for (int i = 0; i < 3; ++i) {
    s[i] = 2.0 * PI * PI * nu * uu[i];
    for (int j = 0; j < 3; ++j) s[i] += uu[j] * g[i][j];
  }
  const Real sx = std::sin(PI*q.x), sy = std::sin(PI*q.y), sz = std::sin(PI*q.z);
  const Real cx = std::cos(PI*q.x), cy = std::cos(PI*q.y), cz = std::cos(PI*q.z);
  s[0] -= PI * sx * cy * cz;
  s[1] -= PI * cx * sy * cz;
  s[2] -= PI * cx * cy * sz + temperature(q);
  return {s[0], s[1], s[2]};
}

// The buoyancy form the gate runs (ADR-041); VIBEFLOW_BUOYANCY=cell selects
// ADR-038's cell force, the recorded baseline.
inline BuoyancyForm gateForm() {
  const char* e = std::getenv("VIBEFLOW_BUOYANCY");
  return (e && std::string(e) == "cell") ? BuoyancyForm::Cell : BuoyancyForm::Balanced;
}

Real temperatureSource(const Vec3& q, Real kappa) {
  const Real sx = std::sin(PI*q.x), sy = std::sin(PI*q.y), sz = std::sin(PI*q.z);
  const Real cx = std::cos(PI*q.x), cy = std::cos(PI*q.y), cz = std::cos(PI*q.z);
  const Vec3 gT{0.5*PI*cx*sy*sz, 0.5*PI*sx*cy*sz, 0.5*PI*sx*sy*cz};
  return velocity(q).dot(gT) + kappa * 1.5 * PI * PI * sx * sy * sz;
}

}  // namespace

int main(int argc, char** argv) {
  ParallelScope mpi(argc, argv);
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const Comm comm = Comm::world();
    Real refU = -1.0, refT = -1.0;
    if (argc > 1) {
      const std::string r(argv[1]);
      const std::size_t comma = r.find(',');
      refU = std::atof(r.substr(0, comma).c_str());
      if (comma != std::string::npos) refT = std::atof(r.substr(comma + 1).c_str());
    }
    const Index n = 8;
    const Real nu = 0.1, kappa = 0.1, dt = 1e-2;
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

    View1<int> uType("uType", nb), tType("tType", nb);
    VectorField ub("ub", nb, 3), u0("u0", nt, 3), src("src", nt, 3);
    ScalarField fb("fb", nb), p0("p0", nt), F0("F0", nf);
    ScalarField T0("T0", nt), tValue("tValue", nb), tSrc("tSrc", nt);
    auto hut = Kokkos::create_mirror_view(uType);
    auto htt = Kokkos::create_mirror_view(tType);
    auto hub = Kokkos::create_mirror_view(ub);
    auto hfb = Kokkos::create_mirror_view(fb);
    auto htv = Kokkos::create_mirror_view(tValue);
    auto hu0 = Kokkos::create_mirror_view(u0);
    auto hsrc = Kokkos::create_mirror_view(src);
    auto hT0 = Kokkos::create_mirror_view(T0);
    auto hts = Kokkos::create_mirror_view(tSrc);
    auto hF0 = Kokkos::create_mirror_view(F0);

    for (Index c = 0; c < nt; ++c) {
      const Vec3 q{hcc(c,0), hcc(c,1), hcc(c,2)};
      const Vec3 v = velocity(q), s = momentumSource(q, nu);
      hu0(c,0) = v.x; hu0(c,1) = v.y; hu0(c,2) = v.z;
      hsrc(c,0) = s.x; hsrc(c,1) = s.y; hsrc(c,2) = s.z;
      hT0(c) = 1.0;
      hts(c) = temperatureSource(q, kappa);
    }
    // Slip and a prescribed heat flux on the x faces, whose outward normals
    // are +-x on this mesh (the distortion leaves the box's faces planar).
    // The flux INTO the domain there is kappa (pi/2) sin(pi y) sin(pi z) on
    // both faces.
    Real fsum = 0.0, asum = 0.0;
    for (Index f = 0; f < nb; ++f) {
      const Vec3 q{hbc(f,0), hbc(f,1), hbc(f,2)};
      const Real mag = std::sqrt(hba(f,0)*hba(f,0) + hba(f,1)*hba(f,1) + hba(f,2)*hba(f,2));
      const bool xFace = std::abs(hba(f,0)) / mag > 0.99;
      const Vec3 v = velocity(q);
      hut(f) = static_cast<int>(xFace ? VelocityBC::Slip : VelocityBC::Dirichlet);
      hub(f,0) = xFace ? 0.0 : v.x; hub(f,1) = xFace ? 0.0 : v.y; hub(f,2) = xFace ? 0.0 : v.z;
      hfb(f) = xFace ? 0.0 : v.x*hba(f,0) + v.y*hba(f,1) + v.z*hba(f,2);
      if (!xFace) { fsum += hfb(f); asum += mag; }
      htt(f) = static_cast<int>(xFace ? TemperatureBC::FixedFlux : TemperatureBC::FixedValue);
      htv(f) = xFace ? kappa * 0.5 * PI * std::sin(PI*q.y) * std::sin(PI*q.z) : temperature(q);
    }
    // Close the mass balance over the faces that carry flux -- globally, or
    // each rank would adjust by a different amount (mms_parallel_ns).
    fsum = comm.sum(fsum); asum = comm.sum(asum);
    for (Index f = 0; f < nb; ++f) {
      if (hut(f) == static_cast<int>(VelocityBC::Slip)) continue;
      const Real mag = std::sqrt(hba(f,0)*hba(f,0) + hba(f,1)*hba(f,1) + hba(f,2)*hba(f,2));
      hfb(f) -= fsum * mag / asum;
    }
    for (Index f = 0; f < nf; ++f) {
      const Vec3 v = velocity({hfc(f,0), hfc(f,1), hfc(f,2)});
      hF0(f) = v.x*hfa(f,0) + v.y*hfa(f,1) + v.z*hfa(f,2);
    }
    Kokkos::deep_copy(uType, hut); Kokkos::deep_copy(tType, htt);
    Kokkos::deep_copy(ub, hub); Kokkos::deep_copy(fb, hfb); Kokkos::deep_copy(tValue, htv);
    Kokkos::deep_copy(u0, hu0); Kokkos::deep_copy(src, hsrc);
    Kokkos::deep_copy(T0, hT0); Kokkos::deep_copy(tSrc, hts); Kokkos::deep_copy(F0, hF0);

    Real l2u = 0.0, l2T = 0.0, cont = 0.0;
    try {
      PisoSolver solver(mesh, nu, dt, ctl, comm);
      EnergyModel em;
      em.kappa = kappa;
      em.betaG = {0.0, 0.0, -1.0};
      em.form = gateForm();
      solver.enableEnergy(em);
      solver.setBoundaryTypes(uType);
      solver.setState(u0, p0, F0);
      solver.setTemperature(T0);
      solver.setTemperatureBoundary(tType, tValue);
      solver.setTemperatureSource(tSrc);

      NativeBiCGStab momentum(mesh, comm);
      NativeCG pressure(mesh, comm, true);
      for (int k = 0; k < steps; ++k)
        cont = solver.advance(ub, fb, src, momentum, pressure).continuityError;

      auto u = solver.velocity(); auto T = solver.temperature(); auto vol = mesh.cellVolume();
      VectorField ue("ue", nt, 3); ScalarField Te("Te", nt);
      Kokkos::deep_copy(ue, u0);
      {
        auto h = Kokkos::create_mirror_view(Te);
        for (Index c = 0; c < nt; ++c) h(c) = temperature({hcc(c,0), hcc(c,1), hcc(c,2)});
        Kokkos::deep_copy(Te, h);
      }
      Real nu2 = 0.0, nT2 = 0.0, den = 0.0;
      Kokkos::parallel_reduce("l2u", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c, Real& a) {
          Real s = 0.0;
          for (int d = 0; d < 3; ++d) { const Real e = u(c,d) - ue(c,d); s += e*e; }
          a += s * vol(c);
        }, nu2);
      Kokkos::parallel_reduce("l2T", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c, Real& a) {
          const Real e = T(c) - Te(c);
          a += e * e * vol(c);
        }, nT2);
      Kokkos::parallel_reduce("vol", Kokkos::RangePolicy<ExecSpace>(0, nc),
        KOKKOS_LAMBDA(const Index c, Real& a) { a += vol(c); }, den);
      den = comm.sum(den);
      l2u = std::sqrt(comm.sum(nu2) / den);
      l2T = std::sqrt(comm.sum(nT2) / den);
    } catch (const std::exception& e) {
      if (comm.rank() == 0) std::printf("  FAIL: %s\n", e.what());
      rc = 1;
    }
    const Index gcells = comm.sum(nc);
    if (rc == 0 && comm.rank() == 0) {
      std::printf("  ranks %2d   cells %6d   ghosts(r0) %5d   div %.1e   L2 %.14e,%.14e",
                  comm.size(), static_cast<int>(gcells),
                  static_cast<int>(mesh.nGhost()), cont, l2u, l2T);
      if (refU > 0.0 && refT > 0.0) {
        const Real ru = std::abs(l2u - refU) / refU, rt = std::abs(l2T - refT) / refT;
        const bool pass = ru < 1e-10 && rt < 1e-10;
        std::printf("   rel u %.2e  T %.2e  %s", ru, rt, pass ? "PASS" : "FAIL");
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
