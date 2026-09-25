// v0 verification gate, C++ side. Same two manufactured solutions as
// prototype/mms_diffusion.py, on the same vertex sets, so the observed orders
// and the L2 errors must agree with the Python reference to solver tolerance.

#include "mesh/HexMesh.hpp"
#include "discretization/Diffusion.hpp"
#include "linalg/LinearSystem.hpp"
#include "linalg/NativeCG.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {

constexpr Real PI = 3.14159265358979323846;

struct Case {
  const char* name;
  Real (*u)(Real, Real, Real);
  Real (*f)(Real, Real, Real);
};

Real uA(Real x, Real y, Real z) { return std::sin(PI*x) * std::sin(PI*y) * std::sin(PI*z); }
Real fA(Real x, Real y, Real z) { return 3.0 * PI * PI * uA(x, y, z); }
Real uB(Real x, Real y, Real z) { return std::exp(x + 0.5*y) * std::sin(PI*z + 0.3); }
Real fB(Real x, Real y, Real z) { return (PI*PI - 1.25) * uB(x, y, z); }

const Case CASES[] = {
  {"A: zero boundary",     uA, fA},
  {"B: non-zero boundary", uB, fB},
};

struct Result { Index n; Real h, l2, linf; int sweeps; Real nonortho; };

Result run(const Case& c, Index n, const std::string& vfile) {
  auto mesh = HexMesh::fromVertexFile(n, vfile);
  const Index nc = mesh.nCells(), nb = mesh.nBoundaryFaces();

  ScalarField src("src", nc), phiB("phiB", nb), phi("phi", nc), exact("exact", nc);
  auto h_cc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellCentre());
  auto h_bc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.boundaryCentre());
  auto h_src = Kokkos::create_mirror_view(src);
  auto h_ex  = Kokkos::create_mirror_view(exact);
  auto h_pb  = Kokkos::create_mirror_view(phiB);
  for (Index i = 0; i < nc; ++i) {
    h_src(i) = c.f(h_cc(i,0), h_cc(i,1), h_cc(i,2));
    h_ex(i)  = c.u(h_cc(i,0), h_cc(i,1), h_cc(i,2));
  }
  for (Index i = 0; i < nb; ++i) h_pb(i) = c.u(h_bc(i,0), h_bc(i,1), h_bc(i,2));
  Kokkos::deep_copy(src, h_src); Kokkos::deep_copy(exact, h_ex); Kokkos::deep_copy(phiB, h_pb);

  DiffusionOperator op(mesh, 1.0);
  LinearSystem sys(mesh);
  NativeCG cg(mesh);
  const int sweeps = op.solve(sys, cg, src, phiB, phi);

  auto vol = mesh.cellVolume();
  Real num = 0.0, den = 0.0, linf = 0.0;
  Kokkos::parallel_reduce("l2", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index i, Real& a) {
      const Real e = phi(i) - exact(i); a += e * e * vol(i);
    }, num);
  Kokkos::parallel_reduce("vol", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index i, Real& a) { a += vol(i); }, den);
  Kokkos::parallel_reduce("linf", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index i, Real& a) {
      a = Kokkos::max(a, Kokkos::abs(phi(i) - exact(i)));
    }, Kokkos::Max<Real>(linf));

  return {n, 1.0 / n, std::sqrt(num / den), linf, sweeps, mesh.maxNonOrthogonality()};
}

bool report(const char* label, const std::vector<Result>& rows) {
  std::printf("\n%s   (max non-orthogonality %.2f deg)\n", label, rows[0].nonortho);
  std::printf("  %4s %9s %19s %13s %8s %8s\n", "N", "h", "L2 error", "Linf", "order", "sweeps");
  Real last = 0.0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (i == 0) {
      std::printf("  %4d %9.5f %19.12e %13.6e %8s %8d\n",
                  rows[i].n, rows[i].h, rows[i].l2, rows[i].linf, "-", rows[i].sweeps);
    } else {
      last = std::log(rows[i-1].l2 / rows[i].l2) / std::log(rows[i-1].h / rows[i].h);
      std::printf("  %4d %9.5f %19.12e %13.6e %8.3f %8d\n",
                  rows[i].n, rows[i].h, rows[i].l2, rows[i].linf, last, rows[i].sweeps);
    }
  }
  const bool ok = last >= 1.9 && last <= 2.1;
  std::printf("  -> order %.3f in [1.9, 2.1]: %s\n", last, ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const std::string dir = argc > 1 ? argv[1] : "tests/fixtures";
    const Index grids[] = {8, 16, 32};
    bool ok = true;
    for (const auto& c : CASES)
      for (const char* tag : {"s0", "s25"}) {
        std::vector<Result> rows;
        for (Index n : grids)
          rows.push_back(run(c, n, dir + "/vertices_n" + std::to_string(n) + "_" + tag + ".txt"));
        ok &= report((std::string(c.name) + " / " +
                      (std::string(tag) == "s0" ? "orthogonal" : "skewed") + " mesh").c_str(), rows);
      }
    std::printf("\nv0 MMS GATE (C++): %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
  return rc;
}
