#include "mesh/RawMesh.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace vibeflow {

std::vector<std::array<Real, 3>> RawMesh::centroids() const {
  std::vector<std::array<Real, 3>> c(hexes.size());
  for (std::size_t i = 0; i < hexes.size(); ++i) {
    Real x = 0.0, y = 0.0, z = 0.0;
    for (int t = 0; t < 8; ++t) {
      const Vec3& p = points[static_cast<std::size_t>(hexes[i][t])];
      x += p.x; y += p.y; z += p.z;
    }
    c[i] = {x / 8.0, y / 8.0, z / 8.0};
  }
  return c;
}

std::vector<Index> RawMesh::vertexNeighbours(const std::vector<int>& part,
                                             int me) const {
  // Mark the vertices my cells touch, then collect every other cell touching
  // one of them. Two passes over the connectivity and one bitmap over the
  // points: no face table, no map of sorted vertex quads.
  std::vector<char> mine(points.size(), 0);
  for (std::size_t c = 0; c < hexes.size(); ++c) {
    if (part[c] != me) continue;
    for (int t = 0; t < 8; ++t) mine[static_cast<std::size_t>(hexes[c][t])] = 1;
  }
  std::vector<Index> out;
  for (std::size_t c = 0; c < hexes.size(); ++c) {
    if (part[c] == me) continue;
    for (int t = 0; t < 8; ++t)
      if (mine[static_cast<std::size_t>(hexes[c][t])]) {
        out.push_back(static_cast<Index>(c));
        break;
      }
  }
  return out;
}

RawMesh RawMesh::fromHexFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open hex mesh file: " + path);
  std::size_t np = 0, nh = 0;
  in >> np >> nh;
  RawMesh m;
  m.points.resize(np);
  for (auto& p : m.points) in >> p.x >> p.y >> p.z;
  m.hexes.resize(nh);
  for (auto& h : m.hexes)
    for (int t = 0; t < 8; ++t) in >> h[t];
  if (!in) throw std::runtime_error("hex mesh file ended early: " + path);
  return m;
}

RawMesh RawMesh::fromVertexFile(Index n, const std::string& path) {
  return {raw::readVertexFile(path), raw::boxConnectivity(n, n, n)};
}

RawMesh RawMesh::generate(Index n, Real skew, const std::string& mode) {
  return {raw::generateVertices(n, skew, mode), raw::boxConnectivity(n, n, n)};
}

RawMesh RawMesh::box(Index nx, Index ny, Index nz, Real Lx, Real Ly, Real Lz) {
  return {raw::boxVertices(nx, ny, nz, Lx, Ly, Lz),
          raw::boxConnectivity(nx, ny, nz)};
}

namespace raw {

std::vector<std::array<Index, 8>> boxConnectivity(Index nx, Index ny, Index nz) {
  // VTK_HEXAHEDRON vertex order: bottom face counter-clockwise, then top.
  const Index vy = ny + 1, vz = nz + 1;
  auto vid = [vy, vz](Index i, Index j, Index k) { return (i * vy + j) * vz + k; };
  std::vector<std::array<Index, 8>> hexes(static_cast<std::size_t>(nx) * ny * nz);
  for (Index i = 0; i < nx; ++i)
    for (Index j = 0; j < ny; ++j)
      for (Index k = 0; k < nz; ++k)
        hexes[(i * ny + j) * nz + k] = {
            vid(i,     j,     k    ), vid(i + 1, j,     k    ),
            vid(i + 1, j + 1, k    ), vid(i,     j + 1, k    ),
            vid(i,     j,     k + 1), vid(i + 1, j,     k + 1),
            vid(i + 1, j + 1, k + 1), vid(i,     j + 1, k + 1)};
  return hexes;
}

std::vector<Vec3> generateVertices(Index n, Real skew, const std::string& mode) {
  const Index nv = n + 1;
  std::vector<Vec3> v(static_cast<std::size_t>(nv) * nv * nv);
  auto g = [nv](Index i) { return static_cast<Real>(i) / static_cast<Real>(nv - 1); };

  for (Index i = 0; i < nv; ++i)
    for (Index j = 0; j < nv; ++j)
      for (Index k = 0; k < nv; ++k)
        v[(i * nv + j) * nv + k] = {g(i), g(j), g(k)};

  if (skew > 0.0 && mode == "smooth") {
    // Amplitude is absolute, not a multiple of h: that is what makes the
    // family a refinement of one geometry. Above about 1/(2 pi) the map stops
    // being invertible and the cells tangle.
    const Real amp = 0.2 * skew;
    for (Index i = 0; i < nv; ++i)
      for (Index j = 0; j < nv; ++j) {
        const Real x = g(i), y = g(j);
        const Real bump = std::sin(M_PI * x) * std::sin(M_PI * y);
        const Real dx = amp * std::sin(2.0 * M_PI * y) * bump;
        const Real dy = amp * std::sin(2.0 * M_PI * x) * bump;
        for (Index k = 0; k < nv; ++k) {
          v[(i * nv + j) * nv + k].x += dx;
          v[(i * nv + j) * nv + k].y += dy;
        }
      }
  } else if (skew > 0.0 && mode != "none") {
    throw std::runtime_error("generateVertices: mode '" + mode +
                             "' is not generated in C++; use fromVertexFile");
  }
  return v;
}

std::vector<Vec3> boxVertices(Index nx, Index ny, Index nz,
                              Real Lx, Real Ly, Real Lz) {
  std::vector<Vec3> v(static_cast<std::size_t>(nx + 1) * (ny + 1) * (nz + 1));
  for (Index i = 0; i <= nx; ++i)
    for (Index j = 0; j <= ny; ++j)
      for (Index k = 0; k <= nz; ++k)
        v[(i * (ny + 1) + j) * (nz + 1) + k] = {
            Lx * static_cast<Real>(i) / nx,
            Ly * static_cast<Real>(j) / ny,
            Lz * static_cast<Real>(k) / nz};
  return v;
}

std::vector<Vec3> readVertexFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open vertex file: " + path);
  std::size_t rows = 0, cols = 0;
  in >> rows >> cols;
  if (cols != 3) throw std::runtime_error("vertex file must have 3 columns");
  std::vector<Vec3> v(rows);
  for (std::size_t i = 0; i < rows; ++i) in >> v[i].x >> v[i].y >> v[i].z;
  return v;
}

}  // namespace raw
}  // namespace vibeflow
