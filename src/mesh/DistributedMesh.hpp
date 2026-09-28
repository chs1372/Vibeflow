#pragma once
// Domain-decomposed view of a global mesh.
//
// v0 partitioning reads the whole mesh on every rank and keeps a slice. That
// is the right trade now -- it makes the parallel-consistency gate possible
// without a parallel reader -- and it is the one thing here that must change
// before the solver is memory-scalable. Recorded as ADR-006.
//
// Partitioning is recursive coordinate bisection on cell centroids: no
// ParMETIS dependency, deterministic, and good enough for a workstation. The
// partitioner is behind an enum so swapping in ParMETIS later touches nothing
// else.
//
// Each rank stores every face incident on one of its owned cells. A face on a
// rank boundary is therefore stored on both sides, each time with the local
// cell as owner and a ghost as neighbour -- the area vector is flipped on the
// side where the global owner is remote, so it always points owner to
// neighbour locally.

#include "core/Parallel.hpp"
#include "core/Types.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/RawMesh.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace vibeflow {

enum class PartitionMethod { Linear, RCB };

class DistributedMesh final : public Mesh {
 public:
  // Takes the RAW description, not a built mesh: building the whole mesh on
  // every rank in order to keep a slice of it is the thing this avoids.
  DistributedMesh(const RawMesh& raw, const Comm& comm,
                  PartitionMethod method = PartitionMethod::RCB);

  // Reads a binary description (`.vmesh`) in parts: each rank reads one
  // block of cells and one block of points and never holds the whole file
  // (ADR-035). Partitioned by Morton order of the cell centroids, split into
  // exactly equal parts, so the partition differs from the RCB one above;
  // the answer must not, and the parallel gates hold it to that.
  DistributedMesh(const std::string& vmeshPath, const Comm& comm);

  Index nCells()         const override { return nOwned_; }
  Index nGhost()         const override { return nGhost_; }
  Index nInternalFaces() const override { return nFaces_; }
  Index nBoundaryFaces() const override { return nBnd_; }
  const std::vector<BoundaryPatch>& patches() const override { return patches_; }
  const HaloExchange* halo() const override { return halo_.get(); }

  VectorField cellCentre() const override { return cellCentre_; }
  ScalarField cellVolume() const override { return cellVolume_; }
  View1<Index> owner()      const override { return owner_; }
  View1<Index> neighbour()  const override { return neigh_; }
  VectorField  faceArea()   const override { return faceArea_; }
  VectorField  faceCentre() const override { return faceCentre_; }
  View1<Index> boundaryCell()   const override { return bCell_; }
  VectorField  boundaryArea()   const override { return bArea_; }
  VectorField  boundaryCentre() const override { return bCentre_; }
  VectorField  boundaryCorners() const override { return bCorners_; }

  Real maxNonOrthogonality() const override;
  Real maxSkewness()         const override;
  Real maxClosureError()     const override;

  // Global cell id of each owned cell, for gathering results in a fixed order.
  const std::vector<Index>& globalCellId() const { return globalId_; }

  static std::vector<int> partition(const RawMesh& raw, int nParts,
                                    PartitionMethod method);

  // What this rank actually constructed, for the memory-scaling gate. Owned
  // plus ghost-ring cells, and every face the vertex hash produced for them
  // -- including the ones later discarded, because they were still built.
  Index nCellsBuilt() const { return cellsBuilt_; }
  Index nFacesBuilt() const { return facesBuilt_; }

  // Peak bytes of the mesh DESCRIPTION -- points at 24 bytes, cells at 64,
  // as in the file -- that this rank held while building. The RawMesh path
  // holds all of it on every rank; the file path should hold about 1/P.
  long long descriptionBytes() const { return descriptionBytes_; }

 private:
  void build(const std::vector<Index>& subGlobal, const std::vector<int>& subRank,
             const std::vector<std::array<Index, 8>>& subHex,
             const std::vector<Vec3>& subPts);

  Comm comm_;
  Index nOwned_{}, nGhost_{}, nFaces_{}, nBnd_{};
  Index cellsBuilt_{}, facesBuilt_{};
  long long descriptionBytes_{};
  std::vector<Index> globalId_;
  std::vector<BoundaryPatch> patches_;
  std::unique_ptr<HaloExchange> halo_;

  View1<Index> owner_, neigh_, bCell_;
  VectorField faceArea_, faceCentre_, bArea_, bCentre_, cellCentre_;
  VectorField bCorners_;            // (4 nBnd, 3), for the wall distance
  ScalarField cellVolume_;
};

}  // namespace vibeflow
