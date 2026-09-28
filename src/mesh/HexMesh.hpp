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

namespace vibeflow {

class HexMesh final : public Mesh {
 public:
  // Vertices in the (i,j,k) ordering of prototype/mesh.py:
  //   id = (i*(ny+1) + j)*(nz+1) + k
  HexMesh(Index n, const std::vector<Vec3>& vertices);
  HexMesh(Index nx, Index ny, Index nz, const std::vector<Vec3>& vertices);

  static HexMesh fromVertexFile(Index n, const std::string& path);

  // Deterministic distorted mesh, matching prototype/mesh.py skew_mode.
  //   "none"    - uniform Cartesian
  //   "smooth"  - fixed analytic distortion applied in x-y and extruded in z,
  //               so every face stays planar (ADR-013). This is the family
  //               order studies run on; it refines toward ONE geometry, where
  //               a randomly perturbed family redraws itself each time and
  //               its mesh quality never converges.
  // Random modes are not generated here: reproducing numpy's PCG64 stream in
  // C++ would be pointless coupling. Those meshes come in through
  // fromVertexFile or CGNS.
  static HexMesh generate(Index n, Real skew = 0.0,
                          const std::string& mode = "smooth");

  // Box [0,Lx] x [0,Ly] x [0,Lz] with independent counts per direction. A 2D
  // benchmark runs as a one-cell-thick slab with slip on the z faces, which
  // needs ny != nz.
  static HexMesh box(Index nx, Index ny, Index nz,
                     Real Lx = 1.0, Real Ly = 1.0, Real Lz = 1.0);

  // Which side of the box each boundary face lies on:
  // 0 = x-, 1 = x+, 2 = y-, 3 = y+, 4 = z-, 5 = z+.
  View1<int> boundarySide() const { return bSide_; }

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
  VectorField  boundaryCorners() const override;

  // Defining points and hex connectivity, for output. Geometry alone cannot
  // draw cells, so the writer needs these.
  const std::vector<Vec3>& points() const { return points_; }
  const std::vector<std::array<Index, 8>>& hexes() const { return hexes_; }
  View2<Index> internalFaceVerts() const { return fVerts_; }
  View2<Index> boundaryFaceVerts() const { return bVerts_; }

  Real maxNonOrthogonality() const override;
  Real maxSkewness()         const override;
  Real maxClosureError()     const override;

 private:
  void buildTopology(Index nx, Index ny, Index nz);
  void buildCellVertices(Index nx, Index ny, Index nz);
  void computeFaceGeometry(const std::vector<Vec3>& verts);
  void computeCellGeometry();

  Index n_{}, nx_{}, ny_{}, nz_{}, nCells_{}, nInternal_{}, nBoundary_{};
  std::vector<BoundaryPatch> patches_;

  View2<Index> fVerts_, bVerts_;          // (nFaces, 4)
  View1<Index> owner_, neigh_, bCell_;
  View1<int> bSide_;
  VectorField  faceArea_, faceCentre_, bArea_, bCentre_;
  VectorField  cellCentre_;
  ScalarField  cellVolume_;
  std::vector<Vec3> points_;
  std::vector<std::array<Index, 8>> hexes_;
};

}  // namespace vibeflow
