#include "mesh/HexMesh.hpp"
#include "mesh/Geometry.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace nsflow {
HexMesh::HexMesh(Index n, const std::vector<Vec3>& verts) : n_(n) {
  const Index nv = n + 1;
  if (static_cast<Index>(verts.size()) != nv * nv * nv)
    throw std::runtime_error("HexMesh: vertex count does not match n");
  points_ = verts;
  buildTopology(n);
  buildCellVertices(n);
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

void HexMesh::buildCellVertices(Index n) {
  // VTK_HEXAHEDRON vertex order: bottom face counter-clockwise, then top.
  const Index nv = n + 1;
  auto vid = [nv](Index i, Index j, Index k) { return (i * nv + j) * nv + k; };
  hexes_.resize(static_cast<std::size_t>(n) * n * n);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j)
      for (Index k = 0; k < n; ++k)
        hexes_[(i * n + j) * n + k] = {
            vid(i,     j,     k    ), vid(i + 1, j,     k    ),
            vid(i + 1, j + 1, k    ), vid(i,     j + 1, k    ),
            vid(i,     j,     k + 1), vid(i + 1, j,     k + 1),
            vid(i + 1, j + 1, k + 1), vid(i,     j + 1, k + 1)};
}

void HexMesh::computeFaceGeometry(const std::vector<Vec3>& verts) {
  auto pts = geometry::uploadPoints(verts);
  faceArea_   = VectorField("faceArea", nInternal_, 3);
  faceCentre_ = VectorField("faceCentre", nInternal_, 3);
  bArea_      = VectorField("bArea", nBoundary_, 3);
  bCentre_    = VectorField("bCentre", nBoundary_, 3);
  geometry::quadGeometry(pts, fVerts_, nInternal_, faceArea_, faceCentre_);
  geometry::quadGeometry(pts, bVerts_, nBoundary_, bArea_, bCentre_);
}

void HexMesh::computeCellGeometry() {
  cellCentre_ = VectorField("cellCentre", nCells_, 3);
  cellVolume_ = ScalarField("cellVolume", nCells_);
  geometry::cellGeometry(nCells_, owner_, neigh_, bCell_, faceCentre_, faceArea_,
                         bCentre_, bArea_, cellCentre_, cellVolume_);
}

Real HexMesh::maxClosureError() const {
  return geometry::maxClosureError(nCells_, owner_, neigh_, bCell_, faceArea_, bArea_);
}
Real HexMesh::maxNonOrthogonality() const {
  return geometry::maxNonOrthogonality(owner_, neigh_, cellCentre_, faceArea_);
}
Real HexMesh::maxSkewness() const {
  return geometry::maxSkewness(owner_, neigh_, cellCentre_, faceCentre_);
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
