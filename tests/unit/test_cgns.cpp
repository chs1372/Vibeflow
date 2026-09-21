// CGNS reader gate: a mesh read from a CGNS file must produce exactly the same
// geometry as the generated mesh it was written from.
//
// This exercises the path a real case takes -- external mesher writes CGNS, we
// read it -- and in particular it tests face discovery by vertex matching,
// which HexMesh bypasses by knowing its own topology.

#include "mesh/HexMesh.hpp"
#ifdef NSFLOW_HAVE_CGNS
#include "mesh/CgnsMesh.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace nsflow;

#ifdef NSFLOW_HAVE_CGNS
namespace {

int failures = 0;

void check(const char* label, bool ok, const std::string& detail) {
  std::printf("  %-28s %s  %s\n", label, ok ? "PASS" : "FAIL", detail.c_str());
  if (!ok) ++failures;
}

// Cell and face ordering differ between the two readers, so quantities are
// compared as sorted multisets -- the geometry must be the same set of values,
// not in the same order.
Real sortedMaxDiff(std::vector<Real> a, std::vector<Real> b) {
  if (a.size() != b.size()) return 1e30;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  Real m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

template <class V> std::vector<Real> flat1(const V& v) {
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
  return std::vector<Real>(h.data(), h.data() + h.extent(0));
}
template <class V> std::vector<Real> mags(const V& v) {
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
  std::vector<Real> out(h.extent(0));
  for (std::size_t i = 0; i < h.extent(0); ++i)
    out[i] = std::sqrt(h(i,0)*h(i,0) + h(i,1)*h(i,1) + h(i,2)*h(i,2));
  return out;
}

}  // namespace
#endif

int main(int argc, char** argv) {
#ifndef NSFLOW_HAVE_CGNS
  (void)argc; (void)argv;
  std::printf("built without CGNS -- skipping\n");
  return 0;
#else
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const std::string dir = argc > 1 ? argv[1] : "tests/fixtures";
    auto ref = HexMesh::fromVertexFile(8, dir + "/vertices_n8_s25.txt");
    CgnsMesh cg(dir + "/mesh_n8_s25.cgns");

    std::printf("CGNS reader vs generated mesh (n=8, skew=0.25)\n");
    check("cell count", cg.nCells() == ref.nCells(),
          std::to_string(cg.nCells()) + " vs " + std::to_string(ref.nCells()));
    check("internal faces", cg.nInternalFaces() == ref.nInternalFaces(),
          std::to_string(cg.nInternalFaces()) + " vs " + std::to_string(ref.nInternalFaces()));
    check("boundary faces", cg.nBoundaryFaces() == ref.nBoundaryFaces(),
          std::to_string(cg.nBoundaryFaces()) + " vs " + std::to_string(ref.nBoundaryFaces()));

    auto fmt = [](Real v) { char b[32]; std::snprintf(b, 32, "max|diff| %.2e", v); return std::string(b); };
    const Real dv = sortedMaxDiff(flat1(cg.cellVolume()), flat1(ref.cellVolume()));
    check("cell volumes", dv < 1e-14, fmt(dv));
    const Real da = sortedMaxDiff(mags(cg.faceArea()), mags(ref.faceArea()));
    check("internal face areas", da < 1e-14, fmt(da));
    const Real db = sortedMaxDiff(mags(cg.boundaryArea()), mags(ref.boundaryArea()));
    check("boundary face areas", db < 1e-14, fmt(db));

    char buf[64];
    std::snprintf(buf, 64, "%.3e", cg.maxClosureError());
    check("closure error", cg.maxClosureError() < 1e-14, buf);
    std::snprintf(buf, 64, "%.6f vs %.6f deg", cg.maxNonOrthogonality(), ref.maxNonOrthogonality());
    check("max non-orthogonality",
          std::abs(cg.maxNonOrthogonality() - ref.maxNonOrthogonality()) < 1e-10, buf);

    Real vsum = 0.0;
    auto vol = cg.cellVolume();
    Kokkos::parallel_reduce("vsum", Kokkos::RangePolicy<ExecSpace>(0, cg.nCells()),
      KOKKOS_LAMBDA(const Index c, Real& a) { a += vol(c); }, vsum);
    std::snprintf(buf, 64, "%.15f", vsum);
    check("volume sums to 1", std::abs(vsum - 1.0) < 1e-13, buf);

    std::printf("\ncgns reader gate: %s\n", failures ? "FAIL" : "PASS");
    rc = failures ? 1 : 0;
  }
  Kokkos::finalize();
  return rc;
#endif
}
