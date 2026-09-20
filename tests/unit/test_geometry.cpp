// Unit gate: C++ mesh geometry must reproduce the Python reference to machine
// precision on the same vertices. Fixtures come from prototype/dump_fixtures.py.

#include "mesh/HexMesh.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace nsflow;

namespace {

std::vector<std::vector<Real>> readFixture(const std::string& path) {
  std::ifstream in(path);
  if (!in) { std::fprintf(stderr, "missing fixture: %s\n", path.c_str()); std::exit(2); }
  std::size_t r, c; in >> r >> c;
  std::vector<std::vector<Real>> out(r, std::vector<Real>(c));
  for (auto& row : out) for (auto& v : row) in >> v;
  return out;
}

int failures = 0;

template <class HostView>
void compare(const char* name, const HostView& got,
             const std::vector<std::vector<Real>>& want, Real tol, int ncomp) {
  if (static_cast<std::size_t>(got.extent(0)) != want.size()) {
    std::printf("  %-18s FAIL  size %zu != %zu\n", name,
                static_cast<std::size_t>(got.extent(0)), want.size());
    ++failures; return;
  }
  Real worst = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i)
    for (int k = 0; k < ncomp; ++k) {
      const Real g = ncomp == 1 ? got(i, 0) : got(i, k);
      worst = std::max(worst, std::abs(g - want[i][k]));
    }
  const bool ok = worst <= tol;
  std::printf("  %-18s %s  max|diff| = %.3e  (tol %.0e)\n",
              name, ok ? "PASS" : "FAIL", worst, tol);
  if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const std::string dir = argc > 1 ? argv[1] : "tests/fixtures";
    const Index n = 8;
    auto mesh = HexMesh::fromVertexFile(n, dir + "/vertices_n8_s25.txt");

    std::printf("C++ vs Python reference geometry (n=8, skew=0.25)\n");

    // Scalar views are (nCells); wrap them so compare() sees a 2-D shape.
    auto vol = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(),
                                                   mesh.cellVolume());
    Kokkos::View<Real**, Kokkos::HostSpace> vol2("vol2", vol.extent(0), 1);
    for (std::size_t i = 0; i < vol.extent(0); ++i) vol2(i, 0) = vol(i);

    auto cc  = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.cellCentre());
    auto fa  = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.faceArea());
    auto fc  = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.faceCentre());
    auto ba  = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.boundaryArea());
    auto bcn = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.boundaryCentre());

    compare("cell_volume", vol2, readFixture(dir + "/cell_volume.txt"), 1e-14, 1);
    compare("cell_centre", cc,  readFixture(dir + "/cell_centre.txt"),  1e-14, 3);
    compare("face_area",   fa,  readFixture(dir + "/face_area.txt"),    1e-14, 3);
    compare("face_centre", fc,  readFixture(dir + "/face_centre.txt"),  1e-14, 3);
    compare("boundary_area",   ba,  readFixture(dir + "/boundary_area.txt"),   1e-14, 3);
    compare("boundary_centre", bcn, readFixture(dir + "/boundary_centre.txt"), 1e-14, 3);

    // Connectivity: integer arrays must match exactly.
    auto own = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.owner());
    auto nei = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.neighbour());
    auto bc  = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.boundaryCell());
    auto checkInt = [&](const char* name, auto& v, const std::string& file) {
      auto want = readFixture(file);
      std::size_t bad = 0;
      for (std::size_t i = 0; i < want.size(); ++i)
        if (v(i) != static_cast<Index>(want[i][0])) ++bad;
      std::printf("  %-18s %s  %zu mismatches of %zu\n", name,
                  bad == 0 ? "PASS" : "FAIL", bad, want.size());
      if (bad) ++failures;
    };
    checkInt("owner",         own, dir + "/owner.txt");
    checkInt("neighbour",     nei, dir + "/neighbour.txt");
    checkInt("boundary_cell", bc,  dir + "/boundary_cell.txt");

    std::printf("  %-18s %.17g deg\n", "max nonortho", mesh.maxNonOrthogonality());
    std::printf("  %-18s %.3e\n",      "closure error", mesh.maxClosureError());
    std::printf("  %-18s %.3e\n",      "max skewness",  mesh.maxSkewness());

    if (mesh.maxClosureError() > 1e-14) { std::printf("  closure FAIL\n"); ++failures; }
    std::printf("\ngeometry unit gate: %s\n", failures ? "FAIL" : "PASS");
    rc = failures ? 1 : 0;
  }
  Kokkos::finalize();
  return rc;
}
