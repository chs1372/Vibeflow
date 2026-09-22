// Prints the generated vertex set so it can be diffed against the Python
// reference. Both sides evaluate the same closed-form expression, so they must
// agree to the last bit -- if they do not, the two implementations are solving
// on different geometry and every later comparison is meaningless.

#include "mesh/HexMesh.hpp"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int n = argc > 1 ? std::stoi(argv[1]) : 8;
    const double skew = argc > 2 ? std::stod(argv[2]) : 0.25;
    const std::string mode = argc > 3 ? argv[3] : "smooth";
    auto m = nsflow::HexMesh::generate(n, skew, mode);
    std::printf("%zu 3\n", m.points().size());
    for (const auto& p : m.points())
      std::printf("%.17g %.17g %.17g\n", p.x, p.y, p.z);
  }
  Kokkos::finalize();
  return 0;
}
