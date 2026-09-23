#pragma once
// A mesh before it is a Mesh: points and 8-node cell connectivity, nothing
// else. No faces, no geometry, no device views.
//
// This exists so a rank can decide WHICH cells are its own before paying to
// build any of them. The built mesh is an order of magnitude larger than this
// description -- faces outnumber cells three to one, each carries an area, a
// centroid and its vertices, and the vertex-hash table that discovers them is
// larger still -- so "read the description everywhere, build only your share"
// is most of the way to a memory-scalable partitioner (ADR-023).
//
// It is not all the way there: the description itself is still replicated.
// Distributing the read too means each rank touching only its own byte range
// of the file, which needs a format that says where the cells are. That is a
// separate change and this one does not pretend to include it.

#include "core/Types.hpp"
#include <array>
#include <string>
#include <vector>

namespace nsflow {

struct RawMesh {
  std::vector<Vec3> points;
  std::vector<std::array<Index, 8>> hexes;

  Index nCells() const { return static_cast<Index>(hexes.size()); }

  // Vertex average, used only to give the partitioner a spatial key. It is
  // NOT the cell centroid the geometry uses -- that one needs the faces --
  // and it does not have to be: a partitioner needs cells that are near each
  // other to land together, and this says that just as well.
  std::vector<std::array<Real, 3>> centroids() const;

  // cells sharing at least one vertex with a cell in `seed`, excluding the
  // seed itself. A superset of the face neighbours, which is what makes it
  // cheap: face adjacency would need the global face table this class exists
  // to avoid building. The extra cells are dropped again once the subdomain's
  // own faces reveal which ghosts are actually touched.
  std::vector<Index> vertexNeighbours(const std::vector<int>& part, int me) const;

  static RawMesh fromHexFile(const std::string& path);
  static RawMesh fromVertexFile(Index n, const std::string& path);
  static RawMesh generate(Index n, Real skew = 0.0,
                          const std::string& mode = "smooth");
  static RawMesh box(Index nx, Index ny, Index nz,
                     Real Lx = 1.0, Real Ly = 1.0, Real Lz = 1.0);
};

// Vertex and connectivity generators, shared with HexMesh so the two mesh
// paths cannot drift apart. HexMesh builds a structured face table from the
// same points; RawMesh hands them to the vertex-hash builder instead.
namespace raw {

std::vector<Vec3> generateVertices(Index n, Real skew, const std::string& mode);
std::vector<Vec3> boxVertices(Index nx, Index ny, Index nz,
                              Real Lx, Real Ly, Real Lz);
std::vector<Vec3> readVertexFile(const std::string& path);
std::vector<std::array<Index, 8>> boxConnectivity(Index nx, Index ny, Index nz);

}  // namespace raw
}  // namespace nsflow
