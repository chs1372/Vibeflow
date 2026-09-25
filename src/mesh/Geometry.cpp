#include "mesh/Geometry.hpp"
#include <algorithm>
#include <cmath>
#include <map>

namespace vibeflow::geometry {

VectorField uploadPoints(const std::vector<Vec3>& pts) {
  VectorField v("points", pts.size(), 3);
  auto h = Kokkos::create_mirror_view(v);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    h(i, 0) = pts[i].x; h(i, 1) = pts[i].y; h(i, 2) = pts[i].z;
  }
  Kokkos::deep_copy(v, h);
  return v;
}

void quadGeometry(const VectorField& points, const View2<Index>& faceVerts,
                  Index nFaces, VectorField& area, VectorField& centre) {
  Kokkos::parallel_for("quadGeometry", Kokkos::RangePolicy<ExecSpace>(0, nFaces),
    KOKKOS_LAMBDA(const Index f) {
      Real p[4][3], avg[3] = {0, 0, 0};
      for (int t = 0; t < 4; ++t) {
        const Index v = faceVerts(f, t);
        for (int k = 0; k < 3; ++k) { p[t][k] = points(v, k); avg[k] += p[t][k]; }
      }
      for (int k = 0; k < 3; ++k) avg[k] *= 0.25;

      Real sf[3] = {0, 0, 0}, mom[3] = {0, 0, 0}, wsum = 0.0;
      for (int t = 0; t < 4; ++t) {
        const Real* a = p[t];
        const Real* b = p[(t + 1) % 4];
        const Real ax = a[0] - avg[0], ay = a[1] - avg[1], az = a[2] - avg[2];
        const Real bx = b[0] - avg[0], by = b[1] - avg[1], bz = b[2] - avg[2];
        const Real hx = 0.5 * (ay * bz - az * by);
        const Real hy = 0.5 * (az * bx - ax * bz);
        const Real hz = 0.5 * (ax * by - ay * bx);
        const Real w = Kokkos::sqrt(hx * hx + hy * hy + hz * hz);
        sf[0] += hx; sf[1] += hy; sf[2] += hz;
        for (int k = 0; k < 3; ++k) mom[k] += w * (avg[k] + a[k] + b[k]) / 3.0;
        wsum += w;
      }
      for (int k = 0; k < 3; ++k) { area(f, k) = sf[k]; centre(f, k) = mom[k] / wsum; }
    });
  Kokkos::fence();
}

void cellGeometry(Index nCells,
                  const View1<Index>& owner, const View1<Index>& neigh,
                  const View1<Index>& bCell,
                  const VectorField& faceCentre, const VectorField& faceArea,
                  const VectorField& bCentre, const VectorField& bArea,
                  VectorField& cellCentre, ScalarField& cellVolume) {
  const Index nf = owner.extent(0), nb = bCell.extent(0);
  VectorField xavg("xavg", nCells, 3), mom("mom", nCells, 3);
  ScalarField cnt("cnt", nCells);
  Kokkos::deep_copy(cellVolume, 0.0);

  Kokkos::parallel_for("xavgInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      for (int d = 0; d < 3; ++d) {
        Kokkos::atomic_add(&xavg(owner(f), d), faceCentre(f, d));
        Kokkos::atomic_add(&xavg(neigh(f), d), faceCentre(f, d));
      }
      Kokkos::atomic_add(&cnt(owner(f)), 1.0);
      Kokkos::atomic_add(&cnt(neigh(f)), 1.0);
    });
  Kokkos::parallel_for("xavgBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      for (int d = 0; d < 3; ++d) Kokkos::atomic_add(&xavg(bCell(f), d), bCentre(f, d));
      Kokkos::atomic_add(&cnt(bCell(f)), 1.0);
    });
  Kokkos::fence();
  Kokkos::parallel_for("xavgNorm", Kokkos::RangePolicy<ExecSpace>(0, nCells),
    KOKKOS_LAMBDA(const Index c) { for (int d = 0; d < 3; ++d) xavg(c, d) /= cnt(c); });
  Kokkos::fence();

  auto cv = cellVolume;
  auto pyramid = KOKKOS_LAMBDA(Index cell, Index f, const VectorField& ctr,
                               const VectorField& ar, Real sign) {
    Real d[3], dot = 0.0;
    for (int k = 0; k < 3; ++k) { d[k] = ctr(f, k) - xavg(cell, k); dot += d[k] * ar(f, k); }
    const Real pv = sign * dot / 3.0;
    Kokkos::atomic_add(&cv(cell), pv);
    for (int k = 0; k < 3; ++k)
      Kokkos::atomic_add(&mom(cell, k), pv * (0.75 * ctr(f, k) + 0.25 * xavg(cell, k)));
  };
  Kokkos::parallel_for("volInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      pyramid(owner(f), f, faceCentre, faceArea, +1.0);
      pyramid(neigh(f), f, faceCentre, faceArea, -1.0);
    });
  Kokkos::parallel_for("volBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) { pyramid(bCell(f), f, bCentre, bArea, +1.0); });
  Kokkos::fence();

  Kokkos::parallel_for("centroid", Kokkos::RangePolicy<ExecSpace>(0, nCells),
    KOKKOS_LAMBDA(const Index c) {
      for (int k = 0; k < 3; ++k) cellCentre(c, k) = mom(c, k) / cv(c);
    });
  Kokkos::fence();
}

Real maxClosureError(Index nCells,
                     const View1<Index>& owner, const View1<Index>& neigh,
                     const View1<Index>& bCell,
                     const VectorField& faceArea, const VectorField& bArea) {
  const Index nf = owner.extent(0), nb = bCell.extent(0);
  VectorField s("closure", nCells, 3);
  Kokkos::parallel_for("closeInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      for (int d = 0; d < 3; ++d) {
        Kokkos::atomic_add(&s(owner(f), d),  faceArea(f, d));
        Kokkos::atomic_add(&s(neigh(f), d), -faceArea(f, d));
      }
    });
  Kokkos::parallel_for("closeBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      for (int d = 0; d < 3; ++d) Kokkos::atomic_add(&s(bCell(f), d), bArea(f, d));
    });
  Kokkos::fence();
  Real m = 0.0;
  Kokkos::parallel_reduce("closeMax", Kokkos::RangePolicy<ExecSpace>(0, nCells),
    KOKKOS_LAMBDA(const Index c, Real& acc) {
      for (int d = 0; d < 3; ++d) acc = Kokkos::max(acc, Kokkos::abs(s(c, d)));
    }, Kokkos::Max<Real>(m));
  return m;
}

Real maxNonOrthogonality(const View1<Index>& owner, const View1<Index>& neigh,
                         const VectorField& cellCentre, const VectorField& faceArea) {
  Real m = 0.0;
  Kokkos::parallel_reduce("nonOrtho", Kokkos::RangePolicy<ExecSpace>(0, owner.extent(0)),
    KOKKOS_LAMBDA(const Index f, Real& acc) {
      Real dot = 0.0, dm = 0.0, sm = 0.0;
      for (int k = 0; k < 3; ++k) {
        const Real d = cellCentre(neigh(f), k) - cellCentre(owner(f), k);
        dot += d * faceArea(f, k); dm += d * d; sm += faceArea(f, k) * faceArea(f, k);
      }
      const Real c = dot / Kokkos::sqrt(dm * sm);
      acc = Kokkos::max(acc, Kokkos::acos(Kokkos::fmin(1.0, Kokkos::fmax(-1.0, c))));
    }, Kokkos::Max<Real>(m));
  return m * 180.0 / M_PI;
}

Real maxSkewness(const View1<Index>& owner, const View1<Index>& neigh,
                 const VectorField& cellCentre, const VectorField& faceCentre) {
  Real m = 0.0;
  Kokkos::parallel_reduce("skew", Kokkos::RangePolicy<ExecSpace>(0, owner.extent(0)),
    KOKKOS_LAMBDA(const Index f, Real& acc) {
      Real d[3], r[3], dd = 0.0, rd = 0.0;
      for (int k = 0; k < 3; ++k) {
        d[k] = cellCentre(neigh(f), k) - cellCentre(owner(f), k);
        r[k] = faceCentre(f, k) - cellCentre(owner(f), k);
        dd += d[k] * d[k]; rd += r[k] * d[k];
      }
      const Real t = rd / dd;
      Real off = 0.0;
      for (int k = 0; k < 3; ++k) { const Real e = r[k] - t * d[k]; off += e * e; }
      acc = Kokkos::max(acc, Kokkos::sqrt(off / dd));
    }, Kokkos::Max<Real>(m));
  return m;
}

FaceTopology buildFaces(const std::vector<std::array<Index, 8>>& hexes) {
  FaceTopology t;
  // key = sorted vertex ids -> (cell, face vertices as first seen)
  std::map<std::array<Index, 4>, std::pair<Index, std::array<Index, 4>>> seen;

  for (std::size_t c = 0; c < hexes.size(); ++c) {
    for (int fi = 0; fi < 6; ++fi) {
      std::array<Index, 4> vs{};
      for (int t2 = 0; t2 < 4; ++t2) vs[t2] = hexes[c][HEX_FACES[fi][t2]];
      std::array<Index, 4> key = vs;
      std::sort(key.begin(), key.end());
      auto it = seen.find(key);
      if (it == seen.end()) {
        seen.emplace(key, std::make_pair(static_cast<Index>(c), vs));
      } else {
        // Second sighting: the first cell owns the face, this one neighbours it.
        // The stored vertex order points out of the owner, hence toward here.
        t.owner.push_back(it->second.first);
        t.neigh.push_back(static_cast<Index>(c));
        t.faceVerts.push_back(it->second.second);
        it->second.first = -1;                 // mark matched
      }
    }
  }
  for (const auto& [key, val] : seen) {
    if (val.first >= 0) { t.bCell.push_back(val.first); t.bVerts.push_back(val.second); }
  }
  return t;
}

}  // namespace vibeflow::geometry
