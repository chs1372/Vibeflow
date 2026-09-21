#include "mesh/CgnsMesh.hpp"
#include "mesh/Geometry.hpp"

#include <cgnslib.h>
#include <stdexcept>
#include <string>

namespace nsflow {
namespace {

void check(int ierr, const char* what) {
  if (ierr != CG_OK)
    throw std::runtime_error(std::string("CGNS ") + what + ": " + cg_get_error());
}

}  // namespace

CgnsMesh::CgnsMesh(const std::string& path) {
  readFile(path);
  buildFaces();
}

void CgnsMesh::readFile(const std::string& path) {
  int fn = 0;
  check(cg_open(path.c_str(), CG_MODE_READ, &fn), "open");

  int nbases = 0;
  check(cg_nbases(fn, &nbases), "nbases");
  if (nbases < 1) throw std::runtime_error("CGNS: no base");
  const int base = 1;

  char name[33];
  int cellDim = 0, physDim = 0;
  check(cg_base_read(fn, base, name, &cellDim, &physDim), "base_read");
  if (cellDim != 3 || physDim != 3)
    throw std::runtime_error("CGNS: only 3D unstructured grids are supported");

  int nzones = 0;
  check(cg_nzones(fn, base, &nzones), "nzones");
  if (nzones != 1)
    throw std::runtime_error("CGNS: multi-zone files need joining first "
                             "(single zone expected, got " + std::to_string(nzones) + ")");
  const int zone = 1;

  ZoneType_t ztype;
  check(cg_zone_type(fn, base, zone, &ztype), "zone_type");
  if (ztype != Unstructured)
    throw std::runtime_error("CGNS: zone is structured; this reader handles Unstructured");

  cgsize_t size[3] = {0, 0, 0};
  check(cg_zone_read(fn, base, zone, name, size), "zone_read");
  const cgsize_t nVerts = size[0];
  nCells_ = static_cast<Index>(size[1]);

  // -- coordinates
  std::vector<double> x(nVerts), y(nVerts), z(nVerts);
  cgsize_t one = 1;
  check(cg_coord_read(fn, base, zone, "CoordinateX", RealDouble, &one, &nVerts, x.data()), "coordX");
  check(cg_coord_read(fn, base, zone, "CoordinateY", RealDouble, &one, &nVerts, y.data()), "coordY");
  check(cg_coord_read(fn, base, zone, "CoordinateZ", RealDouble, &one, &nVerts, z.data()), "coordZ");
  points_.resize(nVerts);
  for (cgsize_t i = 0; i < nVerts; ++i) points_[i] = {x[i], y[i], z[i]};

  // -- HEXA_8 element sections
  int nsections = 0;
  check(cg_nsections(fn, base, zone, &nsections), "nsections");
  for (int s = 1; s <= nsections; ++s) {
    ElementType_t etype;
    cgsize_t start = 0, end = 0;
    int nbndry = 0, parentFlag = 0;
    check(cg_section_read(fn, base, zone, s, name, &etype, &start, &end,
                          &nbndry, &parentFlag), "section_read");
    if (etype != HEXA_8) continue;           // quad BC sections are skipped

    cgsize_t esize = 0;
    check(cg_ElementDataSize(fn, base, zone, s, &esize), "ElementDataSize");
    std::vector<cgsize_t> conn(esize);
    check(cg_elements_read(fn, base, zone, s, conn.data(), nullptr), "elements_read");

    const cgsize_t ne = end - start + 1;
    for (cgsize_t e = 0; e < ne; ++e) {
      std::array<Index, 8> h{};
      for (int t = 0; t < 8; ++t)
        h[t] = static_cast<Index>(conn[e * 8 + t] - 1);   // CGNS is 1-based
      hexes_.push_back(h);
    }
  }
  check(cg_close(fn), "close");

  if (hexes_.empty())
    throw std::runtime_error("CGNS: no HEXA_8 elements found");
  if (static_cast<Index>(hexes_.size()) != nCells_) nCells_ = static_cast<Index>(hexes_.size());
}

void CgnsMesh::buildFaces() {
  auto topo = geometry::buildFaces(hexes_);
  nInternal_ = static_cast<Index>(topo.owner.size());
  nBoundary_ = static_cast<Index>(topo.bCell.size());
  // Boundary faces are one patch until BC sections are wired up; named patches
  // arrive with the BC reader, which v1 needs and v0 does not.
  patches_ = {{"undefined", 0, nBoundary_}};

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
  faceArea_   = VectorField("faceArea", nInternal_, 3);
  faceCentre_ = VectorField("faceCentre", nInternal_, 3);
  bArea_      = VectorField("bArea", nBoundary_, 3);
  bCentre_    = VectorField("bCentre", nBoundary_, 3);
  geometry::quadGeometry(pts, fVerts_, nInternal_, faceArea_, faceCentre_);
  geometry::quadGeometry(pts, bVerts_, nBoundary_, bArea_, bCentre_);

  cellCentre_ = VectorField("cellCentre", nCells_, 3);
  cellVolume_ = ScalarField("cellVolume", nCells_);
  geometry::cellGeometry(nCells_, owner_, neigh_, bCell_, faceCentre_, faceArea_,
                         bCentre_, bArea_, cellCentre_, cellVolume_);
}

Real CgnsMesh::maxClosureError() const {
  return geometry::maxClosureError(nCells_, owner_, neigh_, bCell_, faceArea_, bArea_);
}
Real CgnsMesh::maxNonOrthogonality() const {
  return geometry::maxNonOrthogonality(owner_, neigh_, cellCentre_, faceArea_);
}
Real CgnsMesh::maxSkewness() const {
  return geometry::maxSkewness(owner_, neigh_, cellCentre_, faceCentre_);
}

}  // namespace nsflow
