#include "mesh/HexMesh.hpp"
#include "mesh/RawMesh.hpp"
#include "mesh/Geometry.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace nsflow {
HexMesh::HexMesh(Index n, const std::vector<Vec3>& verts)
    : HexMesh(n, n, n, verts) {}

HexMesh::HexMesh(Index nx, Index ny, Index nz, const std::vector<Vec3>& verts)
    : n_(nx), nx_(nx), ny_(ny), nz_(nz) {
  if (static_cast<Index>(verts.size()) != (nx + 1) * (ny + 1) * (nz + 1))
    throw std::runtime_error("HexMesh: vertex count does not match nx,ny,nz");
  points_ = verts;
  buildTopology(nx, ny, nz);
  buildCellVertices(nx, ny, nz);
  computeFaceGeometry(verts);
  computeCellGeometry();
}

void HexMesh::buildTopology(Index nx, Index ny, Index nz) {
  const Index vy = ny + 1, vz = nz + 1;
  auto vid = [vy, vz](Index i, Index j, Index k) { return (i * vy + j) * vz + k; };
  auto cid = [ny, nz](Index i, Index j, Index k) { return (i * ny + j) * nz + k; };

  nCells_ = nx * ny * nz;
  std::vector<Index> own, nei, bc;
  std::vector<int> side;
  std::vector<std::array<Index, 4>> fv, bv;

  // Face vertex order points the area vector along the positive axis;
  // low-side boundary faces are reversed so theirs points outward.
  auto emit = [&](bool interior, Index lo, Index hi, bool lowSide, int sideId,
                  const std::array<Index, 4>& vs) {
    if (interior) { own.push_back(lo); nei.push_back(hi); fv.push_back(vs); }
    else if (lowSide) {
      bc.push_back(hi); bv.push_back({vs[3], vs[2], vs[1], vs[0]}); side.push_back(sideId);
    } else { bc.push_back(lo); bv.push_back(vs); side.push_back(sideId); }
  };

  for (Index i = 0; i <= nx; ++i)
    for (Index j = 0; j < ny; ++j)
      for (Index k = 0; k < nz; ++k)
        emit(i > 0 && i < nx, i > 0 ? cid(i - 1, j, k) : 0,
             i < nx ? cid(i, j, k) : 0, i == 0, i == 0 ? 0 : 1,
             {vid(i, j, k), vid(i, j + 1, k), vid(i, j + 1, k + 1), vid(i, j, k + 1)});

  for (Index j = 0; j <= ny; ++j)
    for (Index i = 0; i < nx; ++i)
      for (Index k = 0; k < nz; ++k)
        emit(j > 0 && j < ny, j > 0 ? cid(i, j - 1, k) : 0,
             j < ny ? cid(i, j, k) : 0, j == 0, j == 0 ? 2 : 3,
             {vid(i, j, k), vid(i, j, k + 1), vid(i + 1, j, k + 1), vid(i + 1, j, k)});

  for (Index k = 0; k <= nz; ++k)
    for (Index i = 0; i < nx; ++i)
      for (Index j = 0; j < ny; ++j)
        emit(k > 0 && k < nz, k > 0 ? cid(i, j, k - 1) : 0,
             k < nz ? cid(i, j, k) : 0, k == 0, k == 0 ? 4 : 5,
             {vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k), vid(i, j + 1, k)});

  nInternal_ = static_cast<Index>(own.size());
  nBoundary_ = static_cast<Index>(bc.size());
  patches_ = {{"walls", 0, nBoundary_}};

  owner_ = View1<Index>("owner", nInternal_);
  neigh_ = View1<Index>("neigh", nInternal_);
  bCell_ = View1<Index>("bCell", nBoundary_);
  bSide_ = View1<int>("bSide", nBoundary_);
  fVerts_ = View2<Index>("fVerts", nInternal_, 4);
  bVerts_ = View2<Index>("bVerts", nBoundary_, 4);

  auto h_own = Kokkos::create_mirror_view(owner_);
  auto h_nei = Kokkos::create_mirror_view(neigh_);
  auto h_bc  = Kokkos::create_mirror_view(bCell_);
  auto h_sd  = Kokkos::create_mirror_view(bSide_);
  auto h_fv  = Kokkos::create_mirror_view(fVerts_);
  auto h_bv  = Kokkos::create_mirror_view(bVerts_);

  for (Index f = 0; f < nInternal_; ++f) {
    h_own(f) = own[f]; h_nei(f) = nei[f];
    for (int t = 0; t < 4; ++t) h_fv(f, t) = fv[f][t];
  }
  for (Index f = 0; f < nBoundary_; ++f) {
    h_bc(f) = bc[f]; h_sd(f) = side[f];
    for (int t = 0; t < 4; ++t) h_bv(f, t) = bv[f][t];
  }
  Kokkos::deep_copy(owner_, h_own);  Kokkos::deep_copy(neigh_, h_nei);
  Kokkos::deep_copy(bCell_, h_bc);   Kokkos::deep_copy(bSide_, h_sd);
  Kokkos::deep_copy(fVerts_, h_fv);  Kokkos::deep_copy(bVerts_, h_bv);
}

void HexMesh::buildCellVertices(Index nx, Index ny, Index nz) {
  // Shared with RawMesh. The two mesh paths must agree on cell numbering and
  // vertex order or the distributed build silently describes a different mesh
  // from the serial one.
  hexes_ = raw::boxConnectivity(nx, ny, nz);
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

HexMesh HexMesh::generate(Index n, Real skew, const std::string& mode) {
  return HexMesh(n, raw::generateVertices(n, skew, mode));
}

HexMesh HexMesh::box(Index nx, Index ny, Index nz, Real Lx, Real Ly, Real Lz) {
  return HexMesh(nx, ny, nz, raw::boxVertices(nx, ny, nz, Lx, Ly, Lz));
}

HexMesh HexMesh::fromVertexFile(Index n, const std::string& path) {
  return HexMesh(n, raw::readVertexFile(path));
}

}  // namespace nsflow
