#pragma once
// Unstructured cell-centred mesh: face-based connectivity, device-resident.
//
// FIXED INTERFACE. The Python reference in prototype/mesh.py defines the exact
// values every accessor must return; tests/unit/test_geometry compares against
// fixtures generated from it. Change the geometry algorithm only by changing
// both together.

#include "core/Types.hpp"
#include <string>
#include <vector>

namespace vibeflow {

// A named group of boundary faces carrying one boundary condition.
struct BoundaryPatch {
  std::string name;
  Index begin{};   // offset into the boundary-face arrays
  Index size{};
};

// Fills ghost-cell entries of a field from the ranks that own them.
// Serial meshes return nullptr for halo() and allocate no ghosts.
class HaloExchange {
 public:
  virtual ~HaloExchange() = default;
  virtual void exchange(const ScalarField& f) const = 0;
  virtual void exchange(const VectorField& f) const = 0;
};

class Mesh {
 public:
  virtual ~Mesh() = default;

  // -- counts
  // nCells() is the OWNED cells: every loop that reduces (residuals, error
  // norms, volume sums) runs over these. nTotal() adds ghost cells and is what
  // every field allocation uses -- a field sized nCells() will read out of
  // bounds the first time a face touches a ghost neighbour.
  virtual Index nCells()          const = 0;
  virtual Index nGhost()          const { return 0; }
  Index nTotal()                  const { return nCells() + nGhost(); }
  virtual const HaloExchange* halo() const { return nullptr; }
  virtual Index nInternalFaces()  const = 0;
  virtual Index nBoundaryFaces()  const = 0;
  virtual const std::vector<BoundaryPatch>& patches() const = 0;

  // -- cell geometry
  virtual VectorField cellCentre() const = 0;   // (nCells, 3)
  virtual ScalarField cellVolume() const = 0;   // (nCells)

  // -- internal-face connectivity and geometry
  // faceArea points from owner to neighbour; |faceArea| is the face area.
  virtual View1<Index> owner()      const = 0;
  virtual View1<Index> neighbour()  const = 0;
  virtual VectorField  faceArea()   const = 0;
  virtual VectorField  faceCentre() const = 0;

  // -- boundary faces: outward area vectors
  virtual View1<Index> boundaryCell()   const = 0;
  virtual VectorField  boundaryArea()   const = 0;
  virtual VectorField  boundaryCentre() const = 0;
  // The four corners of each boundary face, in the face's own order: row
  // 4f + t is corner t of boundary face f. The wall distance splits each face
  // into the triangles its geometry uses (ADR-042). A mesh that does not keep
  // its points returns an empty field.
  virtual VectorField  boundaryCorners() const { return VectorField(); }

  // -- quality metrics, checked once at start-up and logged
  virtual Real maxNonOrthogonality() const = 0;  // degrees
  virtual Real maxSkewness()         const = 0;
  virtual Real maxClosureError()     const = 0;  // must be ~machine epsilon
};

}  // namespace vibeflow
