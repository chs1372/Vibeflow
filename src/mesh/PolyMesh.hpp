#pragma once
// Hexahedral mesh from an external generator, given as points plus 8-node
// connectivity.
//
// Face topology is rediscovered by vertex matching, the same path the CGNS
// reader uses, so an arbitrary body-fitted mesh gets exactly the geometry a
// generated one does. Boundary faces come out unnamed; a case classifies them
// by position, which is what an unstructured mesh forces anyway and is more
// robust than trusting a mesher's patch names.

#include "core/Types.hpp"
#include "mesh/Mesh.hpp"
#include <array>
#include <string>
#include <vector>

namespace nsflow {

class PolyMesh final : public Mesh {
 public:
  PolyMesh(std::vector<Vec3> points, std::vector<std::array<Index, 8>> hexes);
  static PolyMesh fromHexFile(const std::string& path);

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
  void build();

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
