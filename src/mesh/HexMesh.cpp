#include "mesh/HexMesh.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace nsflow {
namespace {

KOKKOS_INLINE_FUNCTION Vec3 sub(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
KOKKOS_INLINE_FUNCTION Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
KOKKOS_INLINE_FUNCTION Real mag(const Vec3& a) { return std::sqrt(a.mag2()); }

}  // namespace

HexMesh::HexMesh(Index n, const std::vector<Vec3>& verts) : n_(n) {
  const Index nv = n + 1;
  if (static_cast<Index>(verts.size()) != nv * nv * nv)
    throw std::runtime_error("HexMesh: vertex count does not match n");
  buildTopology(n);
  computeFaceGeometry(verts);
  computeCellGeometry();
}

void HexMesh::buildTopology(Index n) {
  const Index nv = n + 1;
  auto vid = [nv](Index i, Index j, Index k) { return (i * nv + j) * nv + k; };
  auto cid = [n](Index i, Index j, Index k) { return (i * n + j) * n + k; };

  nCells_ = n * n * n;
  std::vector<Index> own, nei, bc;
  std::vector<std::array<Index, 4>> fv, bv;

  // Face vertex order is chosen so the area vector points along the positive
  // axis; boundary faces on the low side are reversed to point outward.
  auto emit = [&](bool interior, Index lo, Index hi, bool lowSide,
                  const std::array<Index, 4>& vs) {
    if (interior) { own.push_back(lo); nei.push_back(hi); fv.push_back(vs); }
    else if (lowSide) { bc.push_back(hi); bv.push_back({vs[3], vs[2], vs[1], vs[0]}); }
    else { bc.push_back(lo); bv.push_back(vs); }
  };

  for (Index i = 0; i <= n; ++i)
    for (Index j = 0; j < n; ++j)
      for (Index k = 0; k < n; ++k)
        emit(i > 0 && i < n, i > 0 ? cid(i - 1, j, k) : 0,
             i < n ? cid(i, j, k) : 0, i == 0,
             {vid(i, j, k), vid(i, j + 1, k), vid(i, j + 1, k + 1), vid(i, j, k + 1)});

  for (Index j = 0; j <= n; ++j)
    for (Index i = 0; i < n; ++i)
      for (Index k = 0; k < n; ++k)
        emit(j > 0 && j < n, j > 0 ? cid(i, j - 1, k) : 0,
             j < n ? cid(i, j, k) : 0, j == 0,
             {vid(i, j, k), vid(i, j, k + 1), vid(i + 1, j, k + 1), vid(i + 1, j, k)});

  for (Index k = 0; k <= n; ++k)
    for (Index i = 0; i < n; ++i)
      for (Index j = 0; j < n; ++j)
        emit(k > 0 && k < n, k > 0 ? cid(i, j, k - 1) : 0,
             k < n ? cid(i, j, k) : 0, k == 0,
             {vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k), vid(i, j + 1, k)});

  nInternal_ = static_cast<Index>(own.size());
  nBoundary_ = static_cast<Index>(bc.size());
  patches_ = {{"walls", 0, nBoundary_}};

  owner_ = View1<Index>("owner", nInternal_);
  neigh_ = View1<Index>("neigh", nInternal_);
  bCell_ = View1<Index>("bCell", nBoundary_);
  fVerts_ = View2<Index>("fVerts", nInternal_, 4);
  bVerts_ = View2<Index>("bVerts", nBoundary_, 4);

  auto h_own = Kokkos::create_mirror_view(owner_);
  auto h_nei = Kokkos::create_mirror_view(neigh_);
  auto h_bc  = Kokkos::create_mirror_view(bCell_);
  auto h_fv  = Kokkos::create_mirror_view(fVerts_);
  auto h_bv  = Kokkos::create_mirror_view(bVerts_);

  for (Index f = 0; f < nInternal_; ++f) {
    h_own(f) = own[f]; h_nei(f) = nei[f];
    for (int t = 0; t < 4; ++t) h_fv(f, t) = fv[f][t];
  }
  for (Index f = 0; f < nBoundary_; ++f) {
    h_bc(f) = bc[f];
    for (int t = 0; t < 4; ++t) h_bv(f, t) = bv[f][t];
  }
  Kokkos::deep_copy(owner_, h_own);  Kokkos::deep_copy(neigh_, h_nei);
  Kokkos::deep_copy(bCell_, h_bc);
  Kokkos::deep_copy(fVerts_, h_fv);  Kokkos::deep_copy(bVerts_, h_bv);
}

void HexMesh::computeFaceGeometry(const std::vector<Vec3>& verts) {
  VectorField vpos("vpos", verts.size(), 3);
  auto h_v = Kokkos::create_mirror_view(vpos);
  for (size_t v = 0; v < verts.size(); ++v) {
    h_v(v, 0) = verts[v].x; h_v(v, 1) = verts[v].y; h_v(v, 2) = verts[v].z;
  }
  Kokkos::deep_copy(vpos, h_v);

  // Fan triangulation from the vertex average: exact for planar quads and
  // consistent for warped ones. Matches prototype/mesh.py::_quad_geometry.
  auto quad = [&](View2<Index> fverts, Index nf, VectorField area, VectorField centre) {
    Kokkos::parallel_for("quadGeometry", Kokkos::RangePolicy<ExecSpace>(0, nf),
      KOKKOS_LAMBDA(const Index f) {
        Vec3 p[4];
        for (int t = 0; t < 4; ++t) {
          const Index v = fverts(f, t);
          p[t] = {vpos(v, 0), vpos(v, 1), vpos(v, 2)};
        }
        Vec3 avg{0, 0, 0};
        for (int t = 0; t < 4; ++t) { avg.x += p[t].x; avg.y += p[t].y; avg.z += p[t].z; }
        avg.x *= 0.25; avg.y *= 0.25; avg.z *= 0.25;

        Vec3 sf{0, 0, 0}, mom{0, 0, 0};
        Real wsum = 0.0;
        for (int t = 0; t < 4; ++t) {
          const Vec3& a = p[t];
          const Vec3& b = p[(t + 1) % 4];
          const Vec3 tri = cross(sub(a, avg), sub(b, avg));
          const Vec3 half{0.5 * tri.x, 0.5 * tri.y, 0.5 * tri.z};
          const Real w = mag(half);
          const Vec3 tc{(avg.x + a.x + b.x) / 3.0,
                        (avg.y + a.y + b.y) / 3.0,
                        (avg.z + a.z + b.z) / 3.0};
          sf.x += half.x; sf.y += half.y; sf.z += half.z;
          mom.x += w * tc.x; mom.y += w * tc.y; mom.z += w * tc.z;
          wsum += w;
        }
        area(f, 0) = sf.x; area(f, 1) = sf.y; area(f, 2) = sf.z;
        centre(f, 0) = mom.x / wsum;
        centre(f, 1) = mom.y / wsum;
        centre(f, 2) = mom.z / wsum;
      });
  };

  faceArea_   = VectorField("faceArea", nInternal_, 3);
  faceCentre_ = VectorField("faceCentre", nInternal_, 3);
  bArea_      = VectorField("bArea", nBoundary_, 3);
  bCentre_    = VectorField("bCentre", nBoundary_, 3);
  quad(fVerts_, nInternal_, faceArea_, faceCentre_);
  quad(bVerts_, nBoundary_, bArea_, bCentre_);
  Kokkos::fence();
}

void HexMesh::computeCellGeometry() {
  cellCentre_ = VectorField("cellCentre", nCells_, 3);
  cellVolume_ = ScalarField("cellVolume", nCells_);

  VectorField xavg("xavg", nCells_, 3);
  ScalarField cnt("cnt", nCells_);
  auto own = owner_; auto nei = neigh_; auto bc = bCell_;
  auto fc = faceCentre_; auto fa = faceArea_;
  auto bcen = bCentre_; auto bar = bArea_;
  auto cc = cellCentre_; auto cv = cellVolume_;

  // Reference point per cell: average of its face centres.
  Kokkos::parallel_for("xavgInt", Kokkos::RangePolicy<ExecSpace>(0, nInternal_),
    KOKKOS_LAMBDA(const Index f) {
      for (int d = 0; d < 3; ++d) {
        Kokkos::atomic_add(&xavg(own(f), d), fc(f, d));
        Kokkos::atomic_add(&xavg(nei(f), d), fc(f, d));
      }
      Kokkos::atomic_add(&cnt(own(f)), 1.0);
      Kokkos::atomic_add(&cnt(nei(f)), 1.0);
    });
  Kokkos::parallel_for("xavgBnd", Kokkos::RangePolicy<ExecSpace>(0, nBoundary_),
    KOKKOS_LAMBDA(const Index f) {
      for (int d = 0; d < 3; ++d) Kokkos::atomic_add(&xavg(bc(f), d), bcen(f, d));
      Kokkos::atomic_add(&cnt(bc(f)), 1.0);
    });
  Kokkos::fence();
  Kokkos::parallel_for("xavgNorm", Kokkos::RangePolicy<ExecSpace>(0, nCells_),
    KOKKOS_LAMBDA(const Index c) {
      for (int d = 0; d < 3; ++d) xavg(c, d) /= cnt(c);
    });
  Kokkos::fence();

  // Pyramid decomposition: V = sum (1/3)(Cf - xavg).Sf ; centroid weighted by
  // pyramid volume with the 3/4 : 1/4 apex rule.
  VectorField mom("mom", nCells_, 3);
  auto pyramid = KOKKOS_LAMBDA(Index cell, Index f, const VectorField& ctr,
                               const VectorField& area, Real sign) {
    Real d[3], pc[3], dot = 0.0;
    for (int k = 0; k < 3; ++k) {
      d[k] = ctr(f, k) - xavg(cell, k);
      dot += d[k] * area(f, k);
    }
    const Real pv = sign * dot / 3.0;
    for (int k = 0; k < 3; ++k) pc[k] = 0.75 * ctr(f, k) + 0.25 * xavg(cell, k);
    Kokkos::atomic_add(&cv(cell), pv);
    for (int k = 0; k < 3; ++k) Kokkos::atomic_add(&mom(cell, k), pv * pc[k]);
  };

  Kokkos::parallel_for("volInt", Kokkos::RangePolicy<ExecSpace>(0, nInternal_),
    KOKKOS_LAMBDA(const Index f) {
      pyramid(own(f), f, fc, fa, +1.0);
      pyramid(nei(f), f, fc, fa, -1.0);
    });
  Kokkos::parallel_for("volBnd", Kokkos::RangePolicy<ExecSpace>(0, nBoundary_),
    KOKKOS_LAMBDA(const Index f) { pyramid(bc(f), f, bcen, bar, +1.0); });
  Kokkos::fence();

  Kokkos::parallel_for("centroid", Kokkos::RangePolicy<ExecSpace>(0, nCells_),
    KOKKOS_LAMBDA(const Index c) {
      for (int k = 0; k < 3; ++k) cc(c, k) = mom(c, k) / cv(c);
    });
  Kokkos::fence();
}

Real HexMesh::maxClosureError() const {
  VectorField s("closure", nCells_, 3);
  auto own = owner_; auto nei = neigh_; auto bc = bCell_;
  auto fa = faceArea_; auto bar = bArea_;
  Kokkos::parallel_for("closeInt", Kokkos::RangePolicy<ExecSpace>(0, nInternal_),
    KOKKOS_LAMBDA(const Index f) {
      for (int d = 0; d < 3; ++d) {
        Kokkos::atomic_add(&s(own(f), d), fa(f, d));
        Kokkos::atomic_add(&s(nei(f), d), -fa(f, d));
      }
    });
  Kokkos::parallel_for("closeBnd", Kokkos::RangePolicy<ExecSpace>(0, nBoundary_),
    KOKKOS_LAMBDA(const Index f) {
      for (int d = 0; d < 3; ++d) Kokkos::atomic_add(&s(bc(f), d), bar(f, d));
    });
  Kokkos::fence();
  Real m = 0.0;
  Kokkos::parallel_reduce("closeMax", Kokkos::RangePolicy<ExecSpace>(0, nCells_),
    KOKKOS_LAMBDA(const Index c, Real& acc) {
      for (int d = 0; d < 3; ++d) acc = Kokkos::max(acc, Kokkos::abs(s(c, d)));
    }, Kokkos::Max<Real>(m));
  return m;
}

Real HexMesh::maxNonOrthogonality() const {
  auto own = owner_; auto nei = neigh_; auto cc = cellCentre_; auto fa = faceArea_;
  Real m = 0.0;
  Kokkos::parallel_reduce("nonOrtho", Kokkos::RangePolicy<ExecSpace>(0, nInternal_),
    KOKKOS_LAMBDA(const Index f, Real& acc) {
      Real d[3], dot = 0.0, dm = 0.0, sm = 0.0;
      for (int k = 0; k < 3; ++k) {
        d[k] = cc(nei(f), k) - cc(own(f), k);
        dot += d[k] * fa(f, k); dm += d[k] * d[k]; sm += fa(f, k) * fa(f, k);
      }
      const Real c = dot / Kokkos::sqrt(dm * sm);
      acc = Kokkos::max(acc, Kokkos::acos(Kokkos::fmin(1.0, Kokkos::fmax(-1.0, c))));
    }, Kokkos::Max<Real>(m));
  return m * 180.0 / M_PI;
}

Real HexMesh::maxSkewness() const {
  // Distance from the face centre to the owner-neighbour line, normalised by |d|.
  auto own = owner_; auto nei = neigh_; auto cc = cellCentre_; auto fc = faceCentre_;
  Real m = 0.0;
  Kokkos::parallel_reduce("skew", Kokkos::RangePolicy<ExecSpace>(0, nInternal_),
    KOKKOS_LAMBDA(const Index f, Real& acc) {
      Real d[3], r[3], dd = 0.0, rd = 0.0;
      for (int k = 0; k < 3; ++k) {
        d[k] = cc(nei(f), k) - cc(own(f), k);
        r[k] = fc(f, k) - cc(own(f), k);
        dd += d[k] * d[k]; rd += r[k] * d[k];
      }
      const Real t = rd / dd;
      Real off = 0.0;
      for (int k = 0; k < 3; ++k) { const Real e = r[k] - t * d[k]; off += e * e; }
      acc = Kokkos::max(acc, Kokkos::sqrt(off / dd));
    }, Kokkos::Max<Real>(m));
  return m;
}

HexMesh HexMesh::fromVertexFile(Index n, const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open vertex file: " + path);
  std::size_t rows = 0, cols = 0;
  in >> rows >> cols;
  if (cols != 3) throw std::runtime_error("vertex file must have 3 columns");
  std::vector<Vec3> v(rows);
  for (std::size_t i = 0; i < rows; ++i) in >> v[i].x >> v[i].y >> v[i].z;
  return HexMesh(n, v);
}

}  // namespace nsflow
