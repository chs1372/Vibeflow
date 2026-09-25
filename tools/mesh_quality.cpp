// Report the mesh quality measures the solver actually reacts to, without
// running the solver. Iterating on a mesh generator by launching a flow
// solve is slow enough that it discourages iterating at all.
#include "mesh/PolyMesh.hpp"
#include <algorithm>
#include <cmath>
#include <array>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: mesh_quality <file.hex>\n"); return 2; }
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    auto mesh = vibeflow::PolyMesh::fromHexFile(argv[1]);
    std::printf("%s\n  cells %d  faces %d/%d  non-orth %.1f deg  skew %.2f  "
                "closure %.1e\n",
                argv[1], static_cast<int>(mesh.nCells()),
                static_cast<int>(mesh.nInternalFaces()),
                static_cast<int>(mesh.nBoundaryFaces()),
                mesh.maxNonOrthogonality(), mesh.maxSkewness(),
                mesh.maxClosureError());

    // A maximum hides the shape of the distribution, and it is the shape that
    // decides whether a solver survives: one tangled cell in twenty thousand
    // does not move the max skewness much and will still wreck a run.
    auto vol = Kokkos::create_mirror_view_and_copy(vibeflow::HostSpace::memory_space(),
                                                   mesh.cellVolume());
    vibeflow::Real vmin = 1e300, vmax = -1e300;
    int negative = 0;
    for (vibeflow::Index c = 0; c < mesh.nCells(); ++c) {
      vmin = std::min(vmin, vol(c));
      vmax = std::max(vmax, vol(c));
      if (vol(c) <= 0.0) ++negative;
    }
    std::printf("  volume  min %.3e  max %.3e  ratio %.0f  non-positive %d\n",
                vmin, vmax, vmax / vmin, negative);
    if (negative) {
      std::printf("  TANGLED: %d cells have non-positive volume\n", negative);
      rc = 1;
    }

    // Per-face non-orthogonality and skewness histograms.
    auto own = Kokkos::create_mirror_view_and_copy(vibeflow::HostSpace::memory_space(),
                                                   mesh.owner());
    auto nei = Kokkos::create_mirror_view_and_copy(vibeflow::HostSpace::memory_space(),
                                                   mesh.neighbour());
    auto cc = Kokkos::create_mirror_view_and_copy(vibeflow::HostSpace::memory_space(),
                                                  mesh.cellCentre());
    auto fa = Kokkos::create_mirror_view_and_copy(vibeflow::HostSpace::memory_space(),
                                                  mesh.faceArea());
    auto fc = Kokkos::create_mirror_view_and_copy(vibeflow::HostSpace::memory_space(),
                                                  mesh.faceCentre());
    const int NB = 5;
    const vibeflow::Real edges[NB] = {30.0, 45.0, 60.0, 70.0, 80.0};
    int above[NB] = {0, 0, 0, 0, 0};
    int skewAbove = 0;
    const vibeflow::Index nf = mesh.nInternalFaces();
    for (vibeflow::Index f = 0; f < nf; ++f) {
      vibeflow::Real d[3], a[3], dd = 0.0, aa = 0.0, da = 0.0;
      for (int i = 0; i < 3; ++i) {
        d[i] = cc(nei(f), i) - cc(own(f), i);
        a[i] = fa(f, i);
        dd += d[i] * d[i]; aa += a[i] * a[i]; da += d[i] * a[i];
      }
      const vibeflow::Real ang = std::acos(std::min(1.0, std::abs(da) /
                                   (std::sqrt(dd) * std::sqrt(aa)))) * 180.0 / M_PI;
      for (int b = 0; b < NB; ++b) if (ang > edges[b]) ++above[b];
      // skewness: distance from the face centre to the owner-neighbour line,
      // as a fraction of that line's length
      vibeflow::Real t = 0.0;
      for (int i = 0; i < 3; ++i) t += (fc(f, i) - cc(own(f), i)) * d[i];
      t /= dd;
      vibeflow::Real s2 = 0.0;
      for (int i = 0; i < 3; ++i) {
        const vibeflow::Real e = fc(f, i) - (cc(own(f), i) + t * d[i]);
        s2 += e * e;
      }
      if (std::sqrt(s2 / dd) > 0.5) ++skewAbove;
    }
    std::printf("  faces over  30deg %d (%.1f%%)  45deg %d  60deg %d  70deg %d  "
                "80deg %d   skew>0.5 %d\n",
                above[0], 100.0 * above[0] / nf, above[1], above[2], above[3],
                above[4], skewAbove);

    // Conditioning of the least-squares gradient stencil, per cell.
    //
    // The gradient solves A g = b with A = sum_neighbours w d(x)d(x)^T. The
    // solver inverts A by cofactors and divides by the determinant with no
    // check on it, so a cell whose neighbour offsets nearly share a plane gets
    // a gradient amplified by 1/det -- garbage in one cell, every step, in the
    // same place. Scaled against (trace/3)^3 this is dimensionless: an
    // isotropic stencil scores order 1, a degenerate one scores zero.
    auto bcl = Kokkos::create_mirror_view_and_copy(vibeflow::HostSpace::memory_space(),
                                                   mesh.boundaryCell());
    auto bcn = Kokkos::create_mirror_view_and_copy(vibeflow::HostSpace::memory_space(),
                                                   mesh.boundaryCentre());
    const vibeflow::Index ncell = mesh.nCells();
    std::array<vibeflow::Real, 9> zero9{};
    std::vector<std::array<vibeflow::Real, 9>> A(static_cast<std::size_t>(ncell), zero9);
    auto accum = [&](vibeflow::Index c, const vibeflow::Real* d) {
      vibeflow::Real dd = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
      if (dd <= 0.0) return;
      const vibeflow::Real w = 1.0 / dd;
      for (int a2 = 0; a2 < 3; ++a2)
        for (int b2 = 0; b2 < 3; ++b2) A[c][a2*3+b2] += w * d[a2] * d[b2];
    };
    for (vibeflow::Index f = 0; f < nf; ++f) {
      vibeflow::Real d[3];
      for (int i = 0; i < 3; ++i) d[i] = cc(nei(f), i) - cc(own(f), i);
      accum(own(f), d);
      accum(nei(f), d);
    }
    for (vibeflow::Index f = 0; f < mesh.nBoundaryFaces(); ++f) {
      vibeflow::Real d[3];
      for (int i = 0; i < 3; ++i) d[i] = bcn(f, i) - cc(bcl(f), i);
      accum(bcl(f), d);
    }
    vibeflow::Real worst = 1e300;
    vibeflow::Index worstCell = -1;
    int belowE3 = 0, belowE6 = 0;
    for (vibeflow::Index c = 0; c < ncell; ++c) {
      const auto& m = A[c];
      const vibeflow::Real det = m[0]*(m[4]*m[8] - m[5]*m[7])
                             - m[1]*(m[3]*m[8] - m[5]*m[6])
                             + m[2]*(m[3]*m[7] - m[4]*m[6]);
      const vibeflow::Real tr = (m[0] + m[4] + m[8]) / 3.0;
      const vibeflow::Real q = (tr > 0.0) ? det / (tr*tr*tr) : 0.0;
      if (q < 1e-3) ++belowE3;
      if (q < 1e-6) ++belowE6;
      if (q < worst) { worst = q; worstCell = c; }
    }
    std::printf("  lsq gradient conditioning: worst det/(tr/3)^3 = %.3e at "
                "(%.3f, %.3f, %.3f) r=%.3f;  below 1e-3: %d   below 1e-6: %d\n",
                worst, cc(worstCell,0), cc(worstCell,1), cc(worstCell,2),
                std::hypot(cc(worstCell,0), cc(worstCell,1)), belowE3, belowE6);
  }
  Kokkos::finalize();
  return rc;
}
