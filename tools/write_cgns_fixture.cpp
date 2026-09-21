// Generates the CGNS fixture for the reader gate.
//
// meshio's CGNS writer produces a file neither cgnslib nor meshio's own reader
// accepts, so the fixture is written with the CGNS mid-level library instead.
// The reader gate still means something: face topology is rediscovered from
// element connectivity by vertex matching, and the resulting geometry is
// compared against a mesh built by a completely different path.

#include <cgnslib.h>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(int ierr, const char* what) {
  if (ierr != CG_OK) {
    std::fprintf(stderr, "CGNS %s: %s\n", what, cg_get_error());
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string vfile = argc > 1 ? argv[1] : "tests/fixtures/vertices_n8_s25.txt";
  const std::string out   = argc > 2 ? argv[2] : "tests/fixtures/mesh_n8_s25.cgns";
  const int n = argc > 3 ? std::stoi(argv[3]) : 8;

  std::ifstream in(vfile);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", vfile.c_str()); return 1; }
  std::size_t rows = 0, cols = 0;
  in >> rows >> cols;
  std::vector<double> x(rows), y(rows), z(rows);
  for (std::size_t i = 0; i < rows; ++i) in >> x[i] >> y[i] >> z[i];

  const int nv = n + 1;
  auto vid = [nv](int i, int j, int k) { return (i * nv + j) * nv + k + 1; };  // 1-based
  std::vector<cgsize_t> conn;
  conn.reserve(static_cast<std::size_t>(n) * n * n * 8);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        for (int v : {vid(i,j,k), vid(i+1,j,k), vid(i+1,j+1,k), vid(i,j+1,k),
                      vid(i,j,k+1), vid(i+1,j,k+1), vid(i+1,j+1,k+1), vid(i,j+1,k+1)})
          conn.push_back(v);

  std::remove(out.c_str());
  int fn = 0, base = 0, zone = 0, coord = 0, sec = 0;
  check(cg_open(out.c_str(), CG_MODE_WRITE, &fn), "open");
  check(cg_base_write(fn, "Base", 3, 3, &base), "base_write");

  cgsize_t size[3] = {static_cast<cgsize_t>(rows),
                      static_cast<cgsize_t>(n) * n * n, 0};
  check(cg_zone_write(fn, base, "Zone", size, Unstructured, &zone), "zone_write");
  check(cg_coord_write(fn, base, zone, RealDouble, "CoordinateX", x.data(), &coord), "coordX");
  check(cg_coord_write(fn, base, zone, RealDouble, "CoordinateY", y.data(), &coord), "coordY");
  check(cg_coord_write(fn, base, zone, RealDouble, "CoordinateZ", z.data(), &coord), "coordZ");
  check(cg_section_write(fn, base, zone, "Hexes", HEXA_8, 1, size[1], 0,
                         conn.data(), &sec), "section_write");
  check(cg_close(fn), "close");

  std::printf("wrote %s  (%zu points, %lld hexes)\n", out.c_str(), rows,
              static_cast<long long>(size[1]));
  return 0;
}
