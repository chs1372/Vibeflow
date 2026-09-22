#include "discretization/FaceFlux.hpp"
#include "mesh/HexMesh.hpp"
#include <cmath>

namespace nsflow {
namespace {

// Two-point Gauss-Legendre on [0, 1].
constexpr Real G = 0.5 / 1.7320508075688772935;
const Real GT[2] = {0.5 - G, 0.5 + G};
const Real GW[2] = {0.5, 0.5};

Vec3 sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

template <class FaceVerts>
void circulation(const std::vector<Vec3>& pts, const FaceVerts& fv, Index nf,
                 const VectorFn& A, ScalarField& out) {
  auto h = Kokkos::create_mirror_view(out);
  auto hv = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), fv);
  for (Index f = 0; f < nf; ++f) {
    Real total = 0.0;
    for (int e = 0; e < 4; ++e) {
      const Vec3& a = pts[hv(f, e)];
      const Vec3& b = pts[hv(f, (e + 1) % 4)];
      const Vec3 d = sub(b, a);
      for (int q = 0; q < 2; ++q) {
        const Vec3 x{a.x + GT[q] * d.x, a.y + GT[q] * d.y, a.z + GT[q] * d.z};
        total += GW[q] * A(x).dot(d);
      }
    }
    h(f) = total;
  }
  Kokkos::deep_copy(out, h);
}

// Fan the quad into four triangles from the vertex average; each triangle uses
// its three edge midpoints with equal weight, which is exact for quadratics.
template <class HV, class Fn>
void quadQuadrature(const std::vector<Vec3>& pts,
                    const HV& fv, Index f, Fn&& perTriangle) {
  Vec3 avg{0, 0, 0};
  for (int t = 0; t < 4; ++t) {
    const Vec3& p = pts[fv(f, t)];
    avg.x += 0.25 * p.x; avg.y += 0.25 * p.y; avg.z += 0.25 * p.z;
  }
  for (int t = 0; t < 4; ++t) {
    const Vec3& a = pts[fv(f, t)];
    const Vec3& b = pts[fv(f, (t + 1) % 4)];
    const Vec3 tri = cross(sub(a, avg), sub(b, avg));
    const Vec3 half{0.5 * tri.x, 0.5 * tri.y, 0.5 * tri.z};
    const Vec3 m[3] = {{0.5 * (avg.x + a.x), 0.5 * (avg.y + a.y), 0.5 * (avg.z + a.z)},
                       {0.5 * (a.x + b.x), 0.5 * (a.y + b.y), 0.5 * (a.z + b.z)},
                       {0.5 * (b.x + avg.x), 0.5 * (b.y + avg.y), 0.5 * (b.z + avg.z)}};
    perTriangle(half, m);
  }
}

}  // namespace

void fluxFromPotential(const HexMesh& mesh, const VectorFn& A,
                       ScalarField& fi, ScalarField& fb) {
  fi = ScalarField("fInternal", mesh.nInternalFaces());
  fb = ScalarField("fBoundary", mesh.nBoundaryFaces());
  circulation(mesh.points(), mesh.internalFaceVerts(), mesh.nInternalFaces(), A, fi);
  circulation(mesh.points(), mesh.boundaryFaceVerts(), mesh.nBoundaryFaces(), A, fb);
}

void integrateBoundaryFlux(const HexMesh& mesh, const VectorFn& u, ScalarField& fb) {
  fb = ScalarField("fb", mesh.nBoundaryFaces());
  auto h = Kokkos::create_mirror_view(fb);
  auto bv = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                mesh.boundaryFaceVerts());
  const auto& pts = mesh.points();
  for (Index f = 0; f < mesh.nBoundaryFaces(); ++f) {
    Real total = 0.0;
    quadQuadrature(pts, bv, f, [&](const Vec3& sf, const Vec3* m) {
      Vec3 acc{0, 0, 0};
      for (int i = 0; i < 3; ++i) {
        const Vec3 v = u(m[i]);
        acc.x += v.x / 3.0; acc.y += v.y / 3.0; acc.z += v.z / 3.0;
      }
      total += acc.dot(sf);
    });
    h(f) = total;
  }
  Kokkos::deep_copy(fb, h);
}

void averageBoundaryValue(const HexMesh& mesh, const VectorFn& u, VectorField& ub) {
  ub = VectorField("ub", mesh.nBoundaryFaces(), 3);
  auto h = Kokkos::create_mirror_view(ub);
  auto bv = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                mesh.boundaryFaceVerts());
  const auto& pts = mesh.points();
  for (Index f = 0; f < mesh.nBoundaryFaces(); ++f) {
    Vec3 num{0, 0, 0};
    Real den = 0.0;
    quadQuadrature(pts, bv, f, [&](const Vec3& sf, const Vec3* m) {
      const Real area = std::sqrt(sf.mag2());
      Vec3 acc{0, 0, 0};
      for (int i = 0; i < 3; ++i) {
        const Vec3 v = u(m[i]);
        acc.x += v.x / 3.0; acc.y += v.y / 3.0; acc.z += v.z / 3.0;
      }
      num.x += area * acc.x; num.y += area * acc.y; num.z += area * acc.z;
      den += area;
    });
    h(f, 0) = num.x / den; h(f, 1) = num.y / den; h(f, 2) = num.z / den;
  }
  Kokkos::deep_copy(ub, h);
}

Real maxDiscreteDivergence(const HexMesh& mesh, const ScalarField& fi,
                           const ScalarField& fb) {
  ScalarField div("div", mesh.nCells());
  auto own = mesh.owner(); auto nei = mesh.neighbour(); auto bc = mesh.boundaryCell();
  Kokkos::parallel_for("divInt", Kokkos::RangePolicy<ExecSpace>(0, mesh.nInternalFaces()),
    KOKKOS_LAMBDA(const Index f) {
      Kokkos::atomic_add(&div(own(f)), fi(f));
      Kokkos::atomic_add(&div(nei(f)), -fi(f));
    });
  Kokkos::parallel_for("divBnd", Kokkos::RangePolicy<ExecSpace>(0, mesh.nBoundaryFaces()),
    KOKKOS_LAMBDA(const Index f) { Kokkos::atomic_add(&div(bc(f)), fb(f)); });
  Kokkos::fence();
  Real m = 0.0;
  Kokkos::parallel_reduce("divMax", Kokkos::RangePolicy<ExecSpace>(0, mesh.nCells()),
    KOKKOS_LAMBDA(const Index c, Real& a) { a = Kokkos::max(a, Kokkos::abs(div(c))); },
    Kokkos::Max<Real>(m));
  return m;
}

}  // namespace nsflow
