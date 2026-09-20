#pragma once
// Structured-topology hexahedral mesh with arbitrary vertex positions.
//
// Topology is rebuilt from n using the same ordering as prototype/mesh.py;
// vertex coordinates are read in. This is deliberately the same shape as the
// CGNS path will have: a reader supplies points and connectivity, the geometry
// layer derives everything else.

#include "core/Types.hpp"
#include <array>
#include "mesh/Mesh.hpp"
#include <string>

namespace nsflow {

class HexMesh final : public Mesh {
 public:
  // Vertices in the (i,j,k) ordering of prototype/mesh.py: id = (i*nv + j)*nv + k.
  HexMesh(Index n, const std::vector<Vec3>& vertices);

  static HexMesh fromVertexFile(Index n, const std::string& path);

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

 private:
  void buildTopology(Index n);
  void computeFaceGeometry(const std::vector<Vec3>& verts);
  void computeCellGeometry();

  Index n_{}, nCells_{}, nInternal_{}, nBoundary_{};
  std::vector<BoundaryPatch> patches_;

  View2<Index> fVerts_, bVerts_;          // (nFaces, 4)
  View1<Index> owner_, neigh_, bCell_;
  VectorField  faceArea_, faceCentre_, bArea_, bCentre_;
  VectorField  cellCentre_;
  ScalarField  cellVolume_;
};

}  // namespace nsflow
