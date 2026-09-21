#pragma once
// ParaView output: VTK XML unstructured grid (.vtu), written directly.
//
// No VTK library dependency. The format is stable, ParaView and VisIt read it
// natively, and writing it ourselves keeps the IO layer free of a heavy build
// dependency that would otherwise be needed just to look at results.
//
// Data is written base64-encoded in binary, not ASCII: ASCII .vtu files are
// roughly 3x larger and lose the last bits of a double.
//
// Parallel output writes one .vtu per rank plus a .pvtu index; ParaView opens
// the .pvtu as a single dataset.

#include "core/Types.hpp"
#include <array>
#include <string>
#include <vector>

namespace nsflow {

class Mesh;

class VtuWriter {
 public:
  // The mesh must expose its defining points and hex connectivity for output;
  // geometry alone is not enough to draw cells.
  VtuWriter(const std::vector<Vec3>& points,
            const std::vector<std::array<Index, 8>>& hexes);

  void addCellField(const std::string& name, const ScalarField& f);
  void addCellField(const std::string& name, const VectorField& f);

  // Writes <basename>.vtu. Returns the path written.
  std::string write(const std::string& basename) const;

  // Writes <basename>_<rank>.vtu and, on rank 0, <basename>.pvtu.
  std::string writeParallel(const std::string& basename, int rank, int nRanks) const;

 private:
  struct Field { std::string name; int ncomp; std::vector<Real> data; };

  std::string writePiece(const std::string& path) const;
  std::string pvtuContents(const std::string& basename, int nRanks) const;

  std::vector<Vec3> points_;
  std::vector<std::array<Index, 8>> hexes_;
  std::vector<Field> fields_;
};

}  // namespace nsflow
