#include "mesh/PolyMesh.hpp"
#include "mesh/Geometry.hpp"
#include <fstream>
#include <stdexcept>

namespace vibeflow {

PolyMesh::PolyMesh(std::vector<Vec3> pts, std::vector<std::array<Index, 8>> hexes)
    : points_(std::move(pts)), hexes_(std::move(hexes)) {
  nCells_ = static_cast<Index>(hexes_.size());
  build();
}

PolyMesh PolyMesh::fromHexFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open hex mesh file: " + path);
  std::size_t np = 0, nh = 0;
  in >> np >> nh;
  std::vector<Vec3> pts(np);
  for (auto& p : pts) in >> p.x >> p.y >> p.z;
  std::vector<std::array<Index, 8>> hexes(nh);
  for (auto& h : hexes)
    for (int t = 0; t < 8; ++t) in >> h[t];
  if (!in) throw std::runtime_error("hex mesh file ended early: " + path);
  return PolyMesh(std::move(pts), std::move(hexes));
}

void PolyMesh::build() {
  auto topo = geometry::buildFaces(hexes_);
  nInternal_ = static_cast<Index>(topo.owner.size());
  nBoundary_ = static_cast<Index>(topo.bCell.size());
  patches_ = {{"external", 0, nBoundary_}};

  owner_ = View1<Index>("owner", nInternal_);
  neigh_ = View1<Index>("neigh", nInternal_);
  bCell_ = View1<Index>("bCell", nBoundary_);
  fVerts_ = View2<Index>("fVerts", nInternal_, 4);
  bVerts_ = View2<Index>("bVerts", nBoundary_, 4);

  auto h_o = Kokkos::create_mirror_view(owner_);
  auto h_n = Kokkos::create_mirror_view(neigh_);
  auto h_b = Kokkos::create_mirror_view(bCell_);
  auto h_fv = Kokkos::create_mirror_view(fVerts_);
  auto h_bv = Kokkos::create_mirror_view(bVerts_);
  for (Index f = 0; f < nInternal_; ++f) {
    h_o(f) = topo.owner[f]; h_n(f) = topo.neigh[f];
    for (int t = 0; t < 4; ++t) h_fv(f, t) = topo.faceVerts[f][t];
  }
  for (Index f = 0; f < nBoundary_; ++f) {
    h_b(f) = topo.bCell[f];
    for (int t = 0; t < 4; ++t) h_bv(f, t) = topo.bVerts[f][t];
  }
  Kokkos::deep_copy(owner_, h_o); Kokkos::deep_copy(neigh_, h_n);
  Kokkos::deep_copy(bCell_, h_b);
  Kokkos::deep_copy(fVerts_, h_fv); Kokkos::deep_copy(bVerts_, h_bv);

  auto pts = geometry::uploadPoints(points_);
  faceArea_ = VectorField("faceArea", nInternal_, 3);
  faceCentre_ = VectorField("faceCentre", nInternal_, 3);
  bArea_ = VectorField("bArea", nBoundary_, 3);
  bCentre_ = VectorField("bCentre", nBoundary_, 3);
  geometry::quadGeometry(pts, fVerts_, nInternal_, faceArea_, faceCentre_);
  geometry::quadGeometry(pts, bVerts_, nBoundary_, bArea_, bCentre_);

  cellCentre_ = VectorField("cellCentre", nCells_, 3);
  cellVolume_ = ScalarField("cellVolume", nCells_);
  geometry::cellGeometry(nCells_, owner_, neigh_, bCell_, faceCentre_, faceArea_,
                         bCentre_, bArea_, cellCentre_, cellVolume_);
}

Real PolyMesh::maxClosureError() const {
  return geometry::maxClosureError(nCells_, owner_, neigh_, bCell_, faceArea_, bArea_);
}
Real PolyMesh::maxNonOrthogonality() const {
  return geometry::maxNonOrthogonality(owner_, neigh_, cellCentre_, faceArea_);
}
Real PolyMesh::maxSkewness() const {
  return geometry::maxSkewness(owner_, neigh_, cellCentre_, faceCentre_);
}

VectorField PolyMesh::boundaryCorners() const {
  return geometry::faceCorners(points_, bVerts_, nBoundary_);
}

}  // namespace vibeflow
