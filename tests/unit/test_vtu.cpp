// Output gate: a written .vtu must round-trip. Verified externally by meshio
// in tests/unit/check_vtu.py -- this program only produces the file, so the
// check is done by an independent reader rather than by our own code.

#include "mesh/HexMesh.hpp"
#include "io/VtuWriter.hpp"
#include <cmath>
#include <cstdio>
#include <string>

using namespace vibeflow;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const std::string fixtures = argc > 1 ? argv[1] : "tests/fixtures";
    const std::string out = argc > 2 ? argv[2] : "build/vtu_check";
    auto mesh = HexMesh::fromVertexFile(8, fixtures + "/vertices_n8_s25.txt");
    const Index nc = mesh.nCells();

    // Analytic fields so the reader can check values, not just structure.
    ScalarField s("scalar", nc);
    VectorField v("vector", nc, 3);
    auto cc = mesh.cellCentre();
    Kokkos::parallel_for("fill", Kokkos::RangePolicy<ExecSpace>(0, nc),
      KOKKOS_LAMBDA(const Index c) {
        s(c) = Kokkos::sin(3.0 * cc(c, 0)) * Kokkos::cos(2.0 * cc(c, 1)) + cc(c, 2);
        v(c, 0) = cc(c, 1); v(c, 1) = -cc(c, 0); v(c, 2) = 0.5 * cc(c, 2);
      });
    Kokkos::fence();

    VtuWriter w(mesh.points(), mesh.hexes());
    w.addCellField("scalar", s);
    w.addCellField("velocity", v);
    w.addCellField("volume", mesh.cellVolume());
    w.addCellField("centre", mesh.cellCentre());
    std::printf("wrote %s\n", w.write(out).c_str());

    VtuWriter w2(mesh.points(), mesh.hexes());
    w2.addCellField("scalar", s);
    std::printf("wrote %s\n", w2.writeParallel(out + "_par", 0, 2).c_str());
    w2.writeParallel(out + "_par", 1, 2);
  }
  Kokkos::finalize();
  return 0;
}
