// v1 gate, C++ side: convergence of the convection-diffusion operator.
//
// The prescribed velocity comes from a vector potential, so the mass flux is
// divergence-free to machine precision. Without that the discrete divergence
// error multiplies phi in the convective term and the measured order drops in
// a way that looks like a bug in the scheme.
//
// The manufactured solution is non-zero on every boundary, so the convective
// boundary flux is exercised rather than silently zero.

#include "mesh/HexMesh.hpp"
#include "discretization/Convection.hpp"
#include "discretization/FaceFlux.hpp"
#include "linalg/LinearSystem.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace nsflow;

namespace {

constexpr Real PI = 3.14159265358979323846;
constexpr Real GAMMA = 0.02;

Vec3 potential(const Vec3& p) {
  return {std::sin(PI * p.y) * std::sin(PI * p.z) / PI,
          std::sin(PI * p.z) * std::sin(PI * p.x) / PI,
          std::sin(PI * p.x) * std::sin(PI * p.y) / PI};
}
Vec3 velocity(const Vec3& p) {
  return {std::sin(PI * p.x) * (std::cos(PI * p.y) - std::cos(PI * p.z)),
          std::sin(PI * p.y) * (std::cos(PI * p.z) - std::cos(PI * p.x)),
          std::sin(PI * p.z) * (std::cos(PI * p.x) - std::cos(PI * p.y))};
}
Real phiExact(Real x, Real y, Real z) {
  return std::exp(x + 0.5 * y) * std::sin(PI * z + 0.3);
}
Real source(Real x, Real y, Real z) {
  const Real e = std::exp(x + 0.5 * y);
  const Real s = std::sin(PI * z + 0.3), c = std::cos(PI * z + 0.3);
  const Vec3 u = velocity({x, y, z});
  const Real conv = u.x * e * s + u.y * 0.5 * e * s + u.z * PI * e * c;
  return conv + GAMMA * (PI * PI - 1.25) * e * s;   // laplacian = (1.25 - pi^2) phi
}

struct Row { Index n; Real h, l2, linf, pe, div; int sweeps; bool converged; };

Row run(Index n, Real skew) {
  auto mesh = HexMesh::generate(n, skew, skew == 0.0 ? "none" : "smooth");
  const Index nc = mesh.nCells(), nb = mesh.nBoundaryFaces();

  ScalarField fi, fb;
  fluxFromPotential(mesh, potential, fi, fb);
  const Real div = maxDiscreteDivergence(mesh, fi, fb);

  ScalarField src("src", nc), phiB("phiB", nb), phi("phi", nc), ex("ex", nc);
  auto cc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellCentre());
  auto bc = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.boundaryCentre());
  auto hs = Kokkos::create_mirror_view(src);
  auto he = Kokkos::create_mirror_view(ex);
  auto hp = Kokkos::create_mirror_view(phiB);
  for (Index i = 0; i < nc; ++i) {
    hs(i) = source(cc(i, 0), cc(i, 1), cc(i, 2));
    he(i) = phiExact(cc(i, 0), cc(i, 1), cc(i, 2));
  }
  for (Index i = 0; i < nb; ++i) hp(i) = phiExact(bc(i, 0), bc(i, 1), bc(i, 2));
  Kokkos::deep_copy(src, hs); Kokkos::deep_copy(ex, he); Kokkos::deep_copy(phiB, hp);

  ConvectionDiffusion op(mesh, GAMMA, fi, fb);
  LinearSystem sys(mesh);
  NativeBiCGStab solver(mesh);
  bool converged = false;
  const int sweeps = op.solve(sys, solver, src, phiB, phi, converged, 400, 1e-12, 0.7);

  auto vol = mesh.cellVolume();
  Real num = 0.0, den = 0.0, linf = 0.0;
  Kokkos::parallel_reduce("l2", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index i, Real& a) {
      const Real e = phi(i) - ex(i); a += e * e * vol(i);
    }, num);
  Kokkos::parallel_reduce("vol", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index i, Real& a) { a += vol(i); }, den);
  Kokkos::parallel_reduce("linf", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index i, Real& a) {
      a = Kokkos::max(a, Kokkos::abs(phi(i) - ex(i)));
    }, Kokkos::Max<Real>(linf));

  return {n, 1.0 / n, std::sqrt(num / den), linf, op.maxPeclet(), div, sweeps, converged};
}

bool report(const char* label, const std::vector<Row>& rows) {
  Real maxdiv = 0.0;
  for (const auto& r : rows) maxdiv = std::max(maxdiv, r.div);
  std::printf("\n%s   (max discrete div %.1e)\n", label, maxdiv);
  std::printf("  %4s %9s %19s %13s %8s %8s %8s\n",
              "N", "h", "L2 error", "Linf", "order", "maxPe", "sweeps");
  Real last = 0.0;
  bool allConverged = true;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    char o[16] = "       -";
    if (i) {
      last = std::log(rows[i-1].l2 / rows[i].l2) / std::log(rows[i-1].h / rows[i].h);
      std::snprintf(o, sizeof o, "%8.3f", last);
    }
    allConverged &= rows[i].converged;
    std::printf("  %4d %9.5f %19.12e %13.6e %s %8.2f %8d%s\n",
                rows[i].n, rows[i].h, rows[i].l2, rows[i].linf, o,
                rows[i].pe, rows[i].sweeps, rows[i].converged ? "" : "  NOT CONVERGED");
  }
  const bool ok = last >= 1.85 && last <= 2.15 && allConverged;
  std::printf("  -> order %.3f in [1.85, 2.15]: %s\n", last, ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::vector<Index> grids;
    for (int i = 1; i < argc; ++i) grids.push_back(std::stoi(argv[i]));
    if (grids.empty()) grids = {8, 16, 32};

    bool ok = true;
    for (Real skew : {0.0, 0.25}) {
      std::vector<Row> rows;
      for (Index n : grids) rows.push_back(run(n, skew));
      ok &= report(skew == 0.0 ? "convection-diffusion (gamma=0.02) / orthogonal mesh"
                               : "convection-diffusion (gamma=0.02) / smooth distortion",
                   rows);
    }
    std::printf("\nv1 convection MMS GATE (C++): %s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }
  Kokkos::finalize();
  return rc;
}
