#pragma once
// Face and cell geometry, shared by every mesh source.
//
// Factored out of HexMesh when the CGNS reader arrived: two mesh classes
// computing geometry separately is two places for the same bug. The verified
// algorithm lives here once.

#include "core/Types.hpp"
#include <array>
#include <vector>

namespace nsflow::geometry {

// Area vector and centroid of quadrilateral faces, by fan triangulation from
// the vertex average. Exact for planar quads, consistent for warped ones.
void quadGeometry(const VectorField& points, const View2<Index>& faceVerts,
                  Index nFaces, VectorField& area, VectorField& centre);

// Cell volume and centroid by pyramid decomposition from the average of the
// cell's face centres.
void cellGeometry(Index nCells,
                  const View1<Index>& owner, const View1<Index>& neigh,
                  const View1<Index>& bCell,
                  const VectorField& faceCentre, const VectorField& faceArea,
                  const VectorField& bCentre, const VectorField& bArea,
                  VectorField& cellCentre, ScalarField& cellVolume);

Real maxClosureError(Index nCells,
                     const View1<Index>& owner, const View1<Index>& neigh,
                     const View1<Index>& bCell,
                     const VectorField& faceArea, const VectorField& bArea);

Real maxNonOrthogonality(const View1<Index>& owner, const View1<Index>& neigh,
                         const VectorField& cellCentre, const VectorField& faceArea);

Real maxSkewness(const View1<Index>& owner, const View1<Index>& neigh,
                 const VectorField& cellCentre, const VectorField& faceCentre);

// Upload host-side points into a device view.
VectorField uploadPoints(const std::vector<Vec3>& pts);

// The six faces of a VTK_HEXAHEDRON, each ordered so its normal points out of
// the cell. Verified against a unit cube.
inline constexpr int HEX_FACES[6][4] = {
    {0, 3, 2, 1},   // -z
    {4, 5, 6, 7},   // +z
    {0, 1, 5, 4},   // -y
    {1, 2, 6, 5},   // +x
    {2, 3, 7, 6},   // +y
    {3, 0, 4, 7},   // -x
};

// Build face connectivity from hex cells by matching each face's sorted vertex
// set. A face seen twice is internal; a face seen once is a boundary face.
struct FaceTopology {
  std::vector<Index> owner, neigh, bCell;
  std::vector<std::array<Index, 4>> faceVerts, bVerts;
};
FaceTopology buildFaces(const std::vector<std::array<Index, 8>>& hexes);

}  // namespace nsflow::geometry
