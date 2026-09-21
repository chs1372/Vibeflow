#pragma once
// CGNS unstructured-grid reader.
//
// Reads HEXA_8 element sections and BC patches from a CGNS file and derives
// the same face-based geometry as HexMesh. Topology is discovered from the
// element connectivity rather than assumed, so this is the path a real case
// takes: gmsh or cfMesh writes CGNS, this reads it.
//
// Faces are built by hashing each cell face's sorted vertex set: a face seen
// twice is internal (the second cell becomes the neighbour), a face seen once
// is a boundary face.

#include "core/Types.hpp"
#include "mesh/Mesh.hpp"
#include <array>
#include <string>
#include <vector>

namespace nsflow {

class CgnsMesh final : public Mesh {
 public:
  explicit CgnsMesh(const std::string& path);

  Index nCells()         const override { return nCells_; }
  Index nInternalFaces() const override { return nInternal_; }
  Index nBoundaryFaces() const override { return nBoundary_; }
  const std::vector<BoundaryPatch>& patches() const override { return patches_; }

  VectorField cellCentre() const override { return cellCentre_; }
  ScalarField cellVolume() const override { return cellVolume_; }
  View1<Index> owner()      const override { return owner_; }
  View1<Index> neighbour()  const override { return neigh_; }
  VectorField  faceArea()   const override { return faceArea_; }
  VectorField  faceCentre() const override { return faceCentre_; }
  View1<Index> boundaryCell()   const override { return bCell_; }
  VectorField  boundaryArea()   const override { return bArea_; }
  VectorField  boundaryCentre() const override { return bCentre_; }

  Real maxNonOrthogonality() const override;
  Real maxSkewness()         const override;
  Real maxClosureError()     const override;

  const std::vector<Vec3>& points() const { return points_; }
  const std::vector<std::array<Index, 8>>& hexes() const { return hexes_; }

 private:
  void readFile(const std::string& path);
  void buildFaces();

  Index nCells_{}, nInternal_{}, nBoundary_{};
  std::vector<Vec3> points_;
  std::vector<std::array<Index, 8>> hexes_;
  std::vector<BoundaryPatch> patches_;

  View1<Index> owner_, neigh_, bCell_;
  View2<Index> fVerts_, bVerts_;
  VectorField faceArea_, faceCentre_, bArea_, bCentre_, cellCentre_;
  ScalarField cellVolume_;
};

}  // namespace nsflow
