#include "mesh/DistributedMesh.hpp"
#include "mesh/RawMesh.hpp"
#include "mesh/Geometry.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <stdexcept>

namespace vibeflow {
namespace {

// Host copies of the global mesh, which the partitioner needs on the CPU.
template <class V> auto host(const V& v) {
  return Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
}

void rcb(std::vector<Index>& ids, const std::vector<std::array<Real, 3>>& x,
         int lo, int hi, std::vector<int>& part, int first, int last) {
  const int nParts = last - first;
  if (nParts <= 1) { for (int i = lo; i < hi; ++i) part[ids[i]] = first; return; }

  // Split along the widest extent so the cut plane is the shortest.
  Real mn[3] = {1e300, 1e300, 1e300}, mx[3] = {-1e300, -1e300, -1e300};
  for (int i = lo; i < hi; ++i)
    for (int d = 0; d < 3; ++d) {
      mn[d] = std::min(mn[d], x[ids[i]][d]);
      mx[d] = std::max(mx[d], x[ids[i]][d]);
    }
  int axis = 0;
  for (int d = 1; d < 3; ++d) if (mx[d] - mn[d] > mx[axis] - mn[axis]) axis = d;

  // Split the part count, not the cell count, in half: uneven part counts must
  // still get proportional cell counts or the load is unbalanced.
  const int leftParts = nParts / 2;
  const int n = hi - lo;
  const int nLeft = static_cast<int>(static_cast<long long>(n) * leftParts / nParts);
  std::nth_element(ids.begin() + lo, ids.begin() + lo + nLeft, ids.begin() + hi,
                   [&](Index a, Index b) { return x[a][axis] < x[b][axis]; });
  rcb(ids, x, lo, lo + nLeft, part, first, first + leftParts);
  rcb(ids, x, lo + nLeft, hi, part, first + leftParts, last);
}

class MpiHalo final : public HaloExchange {
 public:
  MpiHalo(Comm comm, std::vector<int> sendRanks,
          std::vector<std::vector<Index>> sendIdx,
          std::vector<int> recvRanks,
          std::vector<std::vector<Index>> recvIdx)
      : comm_(comm), sendRanks_(std::move(sendRanks)), sendIdx_(std::move(sendIdx)),
        recvRanks_(std::move(recvRanks)), recvIdx_(std::move(recvIdx)) {}

  void exchange(const ScalarField& f) const override { run(f, 1); }
  void exchange(const VectorField& f) const override { run(f, 3); }

 private:
  template <class Field>
  void run(const Field& f, int ncomp) const {
    if (!comm_.parallel()) return;
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), f);

    std::vector<std::vector<Real>> sb(sendIdx_.size()), rb(recvIdx_.size());
    for (std::size_t p = 0; p < sendIdx_.size(); ++p) {
      sb[p].resize(sendIdx_[p].size() * ncomp);
      for (std::size_t i = 0; i < sendIdx_[p].size(); ++i)
        for (int k = 0; k < ncomp; ++k)
          sb[p][i * ncomp + k] = get(h, sendIdx_[p][i], k, ncomp);
    }
    for (std::size_t p = 0; p < recvIdx_.size(); ++p) rb[p].resize(recvIdx_[p].size() * ncomp);

    comm_.exchange(sendRanks_, sb, recvRanks_, rb);

    for (std::size_t p = 0; p < recvIdx_.size(); ++p)
      for (std::size_t i = 0; i < recvIdx_[p].size(); ++i)
        for (int k = 0; k < ncomp; ++k)
          set(h, recvIdx_[p][i], k, ncomp, rb[p][i * ncomp + k]);
    Kokkos::deep_copy(f, h);
  }

  template <class H> static Real get(const H& h, Index i, int k, int ncomp) {
    if constexpr (H::rank() == 1) { (void)k; (void)ncomp; return h(i); }
    else return h(i, k);
  }
  template <class H> static void set(const H& h, Index i, int k, int ncomp, Real v) {
    if constexpr (H::rank() == 1) { (void)k; (void)ncomp; h(i) = v; }
    else h(i, k) = v;
  }

  Comm comm_;
  std::vector<int> sendRanks_, recvRanks_;
  std::vector<std::vector<Index>> sendIdx_, recvIdx_;
};

}  // namespace

std::vector<int> DistributedMesh::partition(const RawMesh& raw, int nParts,
                                            PartitionMethod method) {
  const Index nc = raw.nCells();
  std::vector<int> part(nc, 0);
  if (nParts <= 1) return part;

  if (method == PartitionMethod::Linear) {
    for (Index c = 0; c < nc; ++c)
      part[c] = static_cast<int>(static_cast<long long>(c) * nParts / nc);
    return part;
  }
  auto x = raw.centroids();
  std::vector<Index> ids(nc);
  std::iota(ids.begin(), ids.end(), 0);
  rcb(ids, x, 0, static_cast<int>(nc), part, 0, nParts);
  return part;
}

DistributedMesh::DistributedMesh(const RawMesh& raw, const Comm& comm,
                                 PartitionMethod method)
    : comm_(comm) {
  const Index gnc = raw.nCells();
  const auto part = partition(raw, comm.size(), method);
  const int me = comm.rank();
  // Every rank holds the whole description on this path.
  descriptionBytes_ = static_cast<long long>(raw.points.size()) * 24 +
                      static_cast<long long>(raw.hexes.size()) * 64;

  // -- the subdomain: my cells, then a ring of candidate ghosts.
  //
  // Only these get built. Every rank still holds the raw description -- that
  // is the part this change does not fix (ADR-023) -- but the face table, the
  // vertex hash that discovers it, and all the geometry are now per-subdomain.
  std::vector<Index> subGlobal;            // subset id -> global cell
  std::vector<Index> subOf(gnc, -1);       // global cell -> subset id
  for (Index c = 0; c < gnc; ++c)
    if (part[c] == me) { subOf[c] = static_cast<Index>(subGlobal.size()); subGlobal.push_back(c); }
  nOwned_ = static_cast<Index>(subGlobal.size());
  globalId_ = subGlobal;

  for (Index gc : raw.vertexNeighbours(part, me)) {
    subOf[gc] = static_cast<Index>(subGlobal.size());
    subGlobal.push_back(gc);
  }
  const Index nSub = static_cast<Index>(subGlobal.size());

  // Subset connectivity, with the points it touches renumbered so the vertex
  // hash and the point array are both subdomain-sized.
  std::vector<Index> ptOf(raw.points.size(), -1);
  std::vector<Vec3> subPts;
  std::vector<std::array<Index, 8>> subHex(static_cast<std::size_t>(nSub));
  for (Index c = 0; c < nSub; ++c)
    for (int t = 0; t < 8; ++t) {
      const Index gp = raw.hexes[static_cast<std::size_t>(subGlobal[c])][t];
      if (ptOf[gp] < 0) {
        ptOf[gp] = static_cast<Index>(subPts.size());
        subPts.push_back(raw.points[static_cast<std::size_t>(gp)]);
      }
      subHex[c][t] = ptOf[gp];
    }

  std::vector<int> subRank(static_cast<std::size_t>(nSub));
  for (Index c = 0; c < nSub; ++c) subRank[c] = part[subGlobal[c]];
  build(subGlobal, subRank, subHex, subPts);
}

namespace {

// 21 bits per axis: 63-bit Morton keys, a 2-million-cell resolution along
// each direction of the bounding box.
std::uint64_t spread21(std::uint64_t v) {
  v &= 0x1fffff;
  v = (v | v << 32) & 0x1f00000000ffffULL;
  v = (v | v << 16) & 0x1f0000ff0000ffULL;
  v = (v | v << 8)  & 0x100f00f00f00f00fULL;
  v = (v | v << 4)  & 0x10c30c30c30c30c3ULL;
  v = (v | v << 2)  & 0x1249249249249249ULL;
  return v;
}

std::int64_t blockStart(std::int64_t n, int r, int P) { return n * r / P; }

// Which rank's block holds item g of n split into P equal blocks.
int blockOwner(std::int64_t g, std::int64_t n, int P) {
  int r = static_cast<int>(g * P / n);
  while (r + 1 < P && blockStart(n, r + 1, P) <= g) ++r;
  while (r > 0 && blockStart(n, r, P) > g) --r;
  return r;
}

template <class T> std::size_t findSorted(const std::vector<T>& v, T x) {
  return static_cast<std::size_t>(std::lower_bound(v.begin(), v.end(), x) - v.begin());
}

}  // namespace

DistributedMesh::DistributedMesh(const std::string& path, const Comm& comm)
    : comm_(comm) {
  const int P = comm.size(), me = comm.rank();
  const auto hdr = vmesh::readHeader(path);
  const std::int64_t np = hdr.nPoints, nc = hdr.nCells;
  if (nc < P) throw std::runtime_error("fewer cells than ranks in " + path);

  // -- 1. my block of cells, and my block of points. The point block makes
  // this rank the directory for those point ids: anyone who needs one of
  // them asks this rank, so no rank ever reads the whole point array.
  const std::int64_t c0 = blockStart(nc, me, P), c1 = blockStart(nc, me + 1, P);
  const std::int64_t p0 = blockStart(np, me, P), p1 = blockStart(np, me + 1, P);
  auto block = vmesh::readCells(path, hdr, c0, c1 - c0);
  const auto myPts = vmesh::readPoints(path, hdr, p0, p1 - p0);

  // Coordinates of a sorted, unique list of point ids, from their directory
  // ranks. Replies come back in the order asked, per rank.
  auto fetch = [&](const std::vector<std::int64_t>& ids) {
    std::vector<std::vector<long long>> ask(static_cast<std::size_t>(P));
    for (std::int64_t g : ids) ask[blockOwner(g, np, P)].push_back(g);
    const auto asked = comm.alltoallv(ask);
    std::vector<std::vector<double>> answer(static_cast<std::size_t>(P));
    for (int r = 0; r < P; ++r) {
      answer[r].reserve(asked[r].size() * 3);
      for (long long g : asked[r]) {
        const Vec3& x = myPts[static_cast<std::size_t>(g - p0)];
        answer[r].insert(answer[r].end(), {x.x, x.y, x.z});
      }
    }
    const auto got = comm.alltoallv(answer);
    std::vector<Vec3> out(ids.size());
    std::vector<std::size_t> next(static_cast<std::size_t>(P), 0);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      const int r = blockOwner(ids[i], np, P);
      const std::size_t k = next[r]++;
      out[i] = {got[r][3 * k], got[r][3 * k + 1], got[r][3 * k + 2]};
    }
    return out;
  };
  auto uniqueVerts = [](const std::vector<std::array<Int64, 8>>& cells) {
    std::vector<std::int64_t> ids;
    ids.reserve(cells.size() * 8);
    for (const auto& h : cells) ids.insert(ids.end(), h.begin(), h.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
  };

  // -- 2. a spatial key for each cell in my block: Morton order of the vertex
  // average, scaled to the global bounding box.
  std::vector<std::array<Real, 3>> centroid(block.size());
  long long held = 0;
  {
    const auto ids = uniqueVerts(block);
    const auto xyz = fetch(ids);
    held = (c1 - c0) * 64 + (p1 - p0) * 24 + static_cast<long long>(ids.size()) * 24;
    for (std::size_t i = 0; i < block.size(); ++i) {
      Real a[3] = {0.0, 0.0, 0.0};
      for (Int64 v : block[i]) {
        const Vec3& x = xyz[findSorted(ids, static_cast<std::int64_t>(v))];
        a[0] += x.x; a[1] += x.y; a[2] += x.z;
      }
      centroid[i] = {a[0] / 8.0, a[1] / 8.0, a[2] / 8.0};
    }
  }   // the fetched coordinates go here
  Real lo[3], hi[3];
  for (int d = 0; d < 3; ++d) {
    Real l = 1e300, h = -1e300;
    for (const auto& x : centroid) { l = std::min(l, x[d]); h = std::max(h, x[d]); }
    lo[d] = comm.min(l); hi[d] = comm.max(h);
  }
  struct Rec { std::uint64_t key; std::int64_t gid; std::array<Int64, 8> v; };
  std::vector<Rec> recs(block.size());
  for (std::size_t i = 0; i < block.size(); ++i) {
    std::uint64_t k = 0;
    for (int d = 0; d < 3; ++d) {
      const Real span = hi[d] - lo[d];
      const Real t = span > 0.0 ? (centroid[i][d] - lo[d]) / span : 0.0;
      k |= spread21(static_cast<std::uint64_t>(t * 2097151.0)) << d;
    }
    recs[i] = {k, c0 + static_cast<std::int64_t>(i), block[i]};
  }
  block.clear(); block.shrink_to_fit(); centroid.clear(); centroid.shrink_to_fit();
  auto before = [](const Rec& a, const Rec& b) {
    return a.key != b.key ? a.key < b.key : a.gid < b.gid;
  };
  auto pack = [](const Rec& r, std::vector<long long>& out) {
    out.push_back(static_cast<long long>(r.key)); out.push_back(r.gid);
    for (Int64 v : r.v) out.push_back(v);
  };
  auto unpack = [](const std::vector<long long>& in, std::vector<Rec>& out) {
    for (std::size_t i = 0; i + 10 <= in.size(); i += 10) {
      Rec r;
      r.key = static_cast<std::uint64_t>(in[i]); r.gid = in[i + 1];
      for (int t = 0; t < 8; ++t) r.v[t] = in[i + 2 + t];
      out.push_back(r);
    }
  };

  // -- 3. a parallel sort by (key, global id). Sample sort puts each key
  // range on one rank; a second pass then cuts the global order into exactly
  // equal parts, so the partition does not depend on how the samples fell.
  std::sort(recs.begin(), recs.end(), before);
  if (P > 1) {
    std::vector<long long> sample;
    const std::size_t ns = std::min<std::size_t>(64, recs.size());
    for (std::size_t i = 0; i < ns; ++i) {
      const Rec& r = recs[(2 * i + 1) * recs.size() / (2 * ns)];
      sample.push_back(static_cast<long long>(r.key)); sample.push_back(r.gid);
    }
    const auto all = comm.allgatherv(sample);
    std::vector<std::pair<std::uint64_t, std::int64_t>> s;
    for (std::size_t i = 0; i + 1 < all.size(); i += 2)
      s.push_back({static_cast<std::uint64_t>(all[i]), all[i + 1]});
    std::sort(s.begin(), s.end());
    std::vector<std::pair<std::uint64_t, std::int64_t>> split;
    for (int r = 1; r < P; ++r) split.push_back(s[s.size() * r / P]);
    std::vector<std::vector<long long>> out(static_cast<std::size_t>(P));
    for (const Rec& r : recs) {
      const auto dest = std::upper_bound(split.begin(), split.end(),
                                         std::make_pair(r.key, r.gid)) - split.begin();
      pack(r, out[static_cast<std::size_t>(dest)]);
    }
    recs.clear();
    const auto in = comm.alltoallv(out);
    for (const auto& v : in) unpack(v, recs);
    std::sort(recs.begin(), recs.end(), before);

    // Exact balance: my records occupy global positions [offset, offset+n).
    const auto counts = comm.allgatherv({static_cast<long long>(recs.size())});
    long long offset = 0;
    for (int r = 0; r < me; ++r) offset += counts[r];
    std::vector<std::vector<long long>> out2(static_cast<std::size_t>(P));
    for (std::size_t i = 0; i < recs.size(); ++i)
      pack(recs[i], out2[blockOwner(offset + static_cast<long long>(i), nc, P)]);
    recs.clear();
    const auto in2 = comm.alltoallv(out2);
    for (const auto& v : in2) unpack(v, recs);
  }

  // -- 4. my cells, in increasing global id: the order the RawMesh path uses.
  std::sort(recs.begin(), recs.end(),
            [](const Rec& a, const Rec& b) { return a.gid < b.gid; });
  nOwned_ = static_cast<Index>(recs.size());
  globalId_.resize(recs.size());
  for (std::size_t i = 0; i < recs.size(); ++i) globalId_[i] = static_cast<Index>(recs[i].gid);

  // -- 5. candidate ghosts: every cell sharing a vertex with one of mine, as
  // on the RawMesh path, found through the point directory instead of a scan
  // of the whole mesh. Each rank registers its cells at their vertices'
  // directory ranks; a directory rank that sees one vertex touched from two
  // ranks tells each of them about the other's cells there.
  std::vector<std::int64_t> ghostGid;
  std::vector<int> ghostRank;
  {
    std::vector<std::vector<long long>> reg(static_cast<std::size_t>(P));
    for (const Rec& r : recs)
      for (Int64 v : r.v) { auto& b = reg[blockOwner(v, np, P)]; b.push_back(v); b.push_back(r.gid); }
    const auto regs = comm.alltoallv(reg);
    struct Touch { std::int64_t v, gid; int rank; };
    std::vector<Touch> touch;
    for (int r = 0; r < P; ++r)
      for (std::size_t i = 0; i + 1 < regs[r].size(); i += 2)
        touch.push_back({regs[r][i], regs[r][i + 1], r});
    std::sort(touch.begin(), touch.end(), [](const Touch& a, const Touch& b) {
      return a.v != b.v ? a.v < b.v : a.rank != b.rank ? a.rank < b.rank : a.gid < b.gid;
    });
    std::vector<std::vector<long long>> tell(static_cast<std::size_t>(P));
    for (std::size_t i = 0; i < touch.size();) {
      std::size_t j = i;
      while (j < touch.size() && touch[j].v == touch[i].v) ++j;
      if (touch[i].rank != touch[j - 1].rank)          // shared between ranks
        for (std::size_t a = i; a < j; ++a)
          for (std::size_t b = i; b < j; ++b)
            if (touch[a].rank != touch[b].rank) {
              tell[touch[a].rank].push_back(touch[b].gid);
              tell[touch[a].rank].push_back(touch[b].rank);
            }
      i = j;
    }
    touch.clear(); touch.shrink_to_fit();
    const auto told = comm.alltoallv(tell);
    std::vector<std::pair<std::int64_t, int>> g;
    for (const auto& v : told)
      for (std::size_t i = 0; i + 1 < v.size(); i += 2)
        g.push_back({v[i], static_cast<int>(v[i + 1])});
    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());
    for (const auto& [gid, rank] : g) { ghostGid.push_back(gid); ghostRank.push_back(rank); }
  }

  // -- 6. the ghosts' connectivity, from the ranks that own them.
  std::vector<std::array<Int64, 8>> ghostVerts(ghostGid.size());
  {
    std::vector<std::vector<long long>> ask(static_cast<std::size_t>(P));
    for (std::size_t i = 0; i < ghostGid.size(); ++i) ask[ghostRank[i]].push_back(ghostGid[i]);
    const auto asked = comm.alltoallv(ask);
    std::vector<std::vector<long long>> answer(static_cast<std::size_t>(P));
    for (int r = 0; r < P; ++r)
      for (long long gid : asked[r]) {
        const std::size_t k = findSorted(globalId_, static_cast<Index>(gid));
        if (k >= globalId_.size() || globalId_[k] != gid)
          throw std::runtime_error("distributed read: ghost asked of the wrong rank");
        for (Int64 v : recs[k].v) answer[r].push_back(v);
      }
    const auto got = comm.alltoallv(answer);
    std::vector<std::size_t> next(static_cast<std::size_t>(P), 0);
    for (std::size_t i = 0; i < ghostGid.size(); ++i) {
      const int r = ghostRank[i];
      const std::size_t k = next[r]++;
      for (int t = 0; t < 8; ++t) ghostVerts[i][t] = got[r][8 * k + t];
    }
  }

  // -- 7. the subset, and the coordinates of every point it touches.
  std::vector<std::array<Int64, 8>> subVerts;
  subVerts.reserve(recs.size() + ghostGid.size());
  for (const Rec& r : recs) subVerts.push_back(r.v);
  subVerts.insert(subVerts.end(), ghostVerts.begin(), ghostVerts.end());
  const auto ids = uniqueVerts(subVerts);
  const auto subPts = fetch(ids);
  const long long heldB = (p1 - p0) * 24 +
                          static_cast<long long>(subVerts.size()) * 64 +
                          static_cast<long long>(ids.size()) * 24;
  descriptionBytes_ = std::max(held, heldB);

  const Index nSub = static_cast<Index>(subVerts.size());
  std::vector<Index> subGlobal(static_cast<std::size_t>(nSub));
  std::vector<int> subRank(static_cast<std::size_t>(nSub), me);
  std::vector<std::array<Index, 8>> subHex(static_cast<std::size_t>(nSub));
  for (Index s = 0; s < nSub; ++s) {
    subGlobal[s] = s < nOwned_ ? globalId_[s]
                               : static_cast<Index>(ghostGid[static_cast<std::size_t>(s - nOwned_)]);
    if (s >= nOwned_) subRank[s] = ghostRank[static_cast<std::size_t>(s - nOwned_)];
    for (int t = 0; t < 8; ++t)
      subHex[s][t] = static_cast<Index>(findSorted(ids, static_cast<std::int64_t>(subVerts[s][t])));
  }
  build(subGlobal, subRank, subHex, subPts);
}

void DistributedMesh::build(const std::vector<Index>& subGlobal,
                            const std::vector<int>& subRank,
                            const std::vector<std::array<Index, 8>>& subHex,
                            const std::vector<Vec3>& subPts) {
  // nOwned_ and globalId_ are set by the caller: the first nOwned_ subset
  // cells are this rank's own, in increasing global id.
  const Index nSub = static_cast<Index>(subGlobal.size());
  auto topo = geometry::buildFaces(subHex);
  const Index nSubInt = static_cast<Index>(topo.owner.size());
  const Index nSubBnd = static_cast<Index>(topo.bCell.size());
  cellsBuilt_ = nSub;
  facesBuilt_ = nSubInt + nSubBnd;

  // -- geometry over the WHOLE subset, before any filtering.
  //
  // A ghost's volume and centroid need all six of its faces. The ring gives
  // it them: a face of a ghost whose other side lies outside the subdomain is
  // seen once and comes back as a subset boundary face, so cellGeometry's
  // accumulation still closes every cell. Filter first and the ghosts would
  // get volumes computed from part of their surface.
  View1<Index> so("so", nSubInt), sn("sn", nSubInt), sbc("sbc", nSubBnd);
  View2<Index> sfv("sfv", nSubInt, 4), sbv("sbv", nSubBnd, 4);
  {
    auto h_o = Kokkos::create_mirror_view(so);
    auto h_n = Kokkos::create_mirror_view(sn);
    auto h_b = Kokkos::create_mirror_view(sbc);
    auto h_fv = Kokkos::create_mirror_view(sfv);
    auto h_bv = Kokkos::create_mirror_view(sbv);
    for (Index f = 0; f < nSubInt; ++f) {
      h_o(f) = topo.owner[f]; h_n(f) = topo.neigh[f];
      for (int t = 0; t < 4; ++t) h_fv(f, t) = topo.faceVerts[f][t];
    }
    for (Index f = 0; f < nSubBnd; ++f) {
      h_b(f) = topo.bCell[f];
      for (int t = 0; t < 4; ++t) h_bv(f, t) = topo.bVerts[f][t];
    }
    Kokkos::deep_copy(so, h_o); Kokkos::deep_copy(sn, h_n); Kokkos::deep_copy(sbc, h_b);
    Kokkos::deep_copy(sfv, h_fv); Kokkos::deep_copy(sbv, h_bv);
  }
  auto pts = geometry::uploadPoints(subPts);
  VectorField sfa("sfa", nSubInt, 3), sfc("sfc", nSubInt, 3);
  VectorField sba("sba", nSubBnd, 3), sbcn("sbcn", nSubBnd, 3);
  geometry::quadGeometry(pts, sfv, nSubInt, sfa, sfc);
  geometry::quadGeometry(pts, sbv, nSubBnd, sba, sbcn);
  VectorField scc("scc", nSub, 3);
  ScalarField scv("scv", nSub);
  geometry::cellGeometry(nSub, so, sn, sbc, sfc, sfa, sbcn, sba, scc, scv);

  auto h_sfa = host(sfa); auto h_sfc = host(sfc);
  auto h_sba = host(sba); auto h_sbcn = host(sbcn);
  auto h_scc = host(scc); auto h_scv = host(scv);

  // -- keep only what touches an owned cell, with that cell as the owner.
  std::vector<char> ghostUsed(static_cast<std::size_t>(nSub), 0);
  for (Index f = 0; f < nSubInt; ++f) {
    const Index o = topo.owner[f], n = topo.neigh[f];
    const bool mo = o < nOwned_, mn = n < nOwned_;
    if (mo && !mn) ghostUsed[n] = 1;
    else if (!mo && mn) ghostUsed[o] = 1;
  }
  // Ghost local ids run in (owning rank, global id) order, which makes each
  // receive buffer one contiguous run and puts it in the order the sender
  // builds its send buffer -- both sort by global id, so neither has to be
  // told the other's ordering.
  std::vector<Index> used;
  for (Index s = nOwned_; s < nSub; ++s) if (ghostUsed[s]) used.push_back(s);
  std::sort(used.begin(), used.end(), [&](Index a, Index b) {
    const int ra = subRank[a], rb = subRank[b];
    return ra != rb ? ra < rb : subGlobal[a] < subGlobal[b];
  });
  std::vector<Index> localOfSub(static_cast<std::size_t>(nSub), -1);
  for (Index s = 0; s < nOwned_; ++s) localOfSub[s] = s;
  for (std::size_t i = 0; i < used.size(); ++i)
    localOfSub[used[i]] = nOwned_ + static_cast<Index>(i);
  nGhost_ = static_cast<Index>(used.size());

  struct LFace { Index own, nei, sf; bool flip; };
  std::vector<LFace> faces;
  for (Index f = 0; f < nSubInt; ++f) {
    const Index o = topo.owner[f], n = topo.neigh[f];
    const bool mo = o < nOwned_, mn = n < nOwned_;
    if (mo && mn)       faces.push_back({localOfSub[o], localOfSub[n], f, false});
    else if (mo)        faces.push_back({localOfSub[o], localOfSub[n], f, false});
    else if (mn)        faces.push_back({localOfSub[n], localOfSub[o], f, true});
  }
  nFaces_ = static_cast<Index>(faces.size());

  std::vector<Index> bnd;
  for (Index f = 0; f < nSubBnd; ++f)
    if (topo.bCell[f] < nOwned_) bnd.push_back(f);
  nBnd_ = static_cast<Index>(bnd.size());
  patches_ = {{"undefined", 0, nBnd_}};

  // -- emit
  const Index nt = nTotal();
  owner_ = View1<Index>("owner", nFaces_);
  neigh_ = View1<Index>("neigh", nFaces_);
  bCell_ = View1<Index>("bCell", nBnd_);
  faceArea_ = VectorField("faceArea", nFaces_, 3);
  faceCentre_ = VectorField("faceCentre", nFaces_, 3);
  bArea_ = VectorField("bArea", nBnd_, 3);
  bCentre_ = VectorField("bCentre", nBnd_, 3);
  cellCentre_ = VectorField("cellCentre", nt, 3);
  cellVolume_ = ScalarField("cellVolume", nt);

  auto h_o = Kokkos::create_mirror_view(owner_);
  auto h_n = Kokkos::create_mirror_view(neigh_);
  auto h_fa = Kokkos::create_mirror_view(faceArea_);
  auto h_fc = Kokkos::create_mirror_view(faceCentre_);
  for (Index f = 0; f < nFaces_; ++f) {
    h_o(f) = faces[f].own; h_n(f) = faces[f].nei;
    const Real sgn = faces[f].flip ? -1.0 : 1.0;
    for (int k = 0; k < 3; ++k) {
      h_fa(f, k) = sgn * h_sfa(faces[f].sf, k);
      h_fc(f, k) = h_sfc(faces[f].sf, k);
    }
  }
  auto h_b = Kokkos::create_mirror_view(bCell_);
  auto h_ba = Kokkos::create_mirror_view(bArea_);
  auto h_bcn = Kokkos::create_mirror_view(bCentre_);
  for (Index f = 0; f < nBnd_; ++f) {
    h_b(f) = localOfSub[topo.bCell[bnd[f]]];
    for (int k = 0; k < 3; ++k) {
      h_ba(f, k) = h_sba(bnd[f], k);
      h_bcn(f, k) = h_sbcn(bnd[f], k);
    }
  }
  auto h_cc = Kokkos::create_mirror_view(cellCentre_);
  auto h_cv = Kokkos::create_mirror_view(cellVolume_);
  for (Index s = 0; s < nSub; ++s) {
    const Index l = localOfSub[s];
    if (l < 0) continue;
    for (int k = 0; k < 3; ++k) h_cc(l, k) = h_scc(s, k);
    h_cv(l) = h_scv(s);
  }

  Kokkos::deep_copy(owner_, h_o); Kokkos::deep_copy(neigh_, h_n);
  Kokkos::deep_copy(faceArea_, h_fa); Kokkos::deep_copy(faceCentre_, h_fc);
  Kokkos::deep_copy(bCell_, h_b); Kokkos::deep_copy(bArea_, h_ba);
  Kokkos::deep_copy(bCentre_, h_bcn);
  Kokkos::deep_copy(cellCentre_, h_cc); Kokkos::deep_copy(cellVolume_, h_cv);

  // -- halo schedule, both sides derived from the same local face list.
  // rank r sends me the ghosts I use that r owns; symmetrically I send r the
  // owned cells sitting across a face from one of r's ghosts. Face adjacency
  // is symmetric, so the two lists are the same set seen from either end, and
  // sorting both by global cell id is enough to line them up.
  std::map<int, std::vector<Index>> recvLocal, sendGlobal;
  for (std::size_t i = 0; i < used.size(); ++i)
    recvLocal[subRank[used[i]]].push_back(nOwned_ + static_cast<Index>(i));
  for (const auto& f : faces) {
    if (f.nei < nOwned_) continue;                  // both mine
    const Index gsub = used[static_cast<std::size_t>(f.nei - nOwned_)];
    sendGlobal[subRank[gsub]].push_back(globalId_[f.own]);
  }
  std::vector<int> sr, rr;
  std::vector<std::vector<Index>> si, ri;
  for (auto& [r, v] : sendGlobal) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    sr.push_back(r);
    std::vector<Index> idx;
    idx.reserve(v.size());
    // Owned: subset id == local id, and globalId_ is sorted.
    for (Index gc : v)
      idx.push_back(static_cast<Index>(
          std::lower_bound(globalId_.begin(), globalId_.end(), gc) - globalId_.begin()));
    si.push_back(std::move(idx));
  }
  for (auto& [r, v] : recvLocal) { rr.push_back(r); ri.push_back(std::move(v)); }
  halo_ = std::make_unique<MpiHalo>(comm_, std::move(sr), std::move(si),
                                    std::move(rr), std::move(ri));
}

Real DistributedMesh::maxClosureError() const {
  // Interface faces make a rank's cells non-closed only if a face is missing;
  // with both-sides storage each owned cell still sees all of its faces.
  return comm_.max(geometry::maxClosureError(nTotal(), owner_, neigh_, bCell_,
                                             faceArea_, bArea_));
}
Real DistributedMesh::maxNonOrthogonality() const {
  return comm_.max(geometry::maxNonOrthogonality(owner_, neigh_, cellCentre_, faceArea_));
}
Real DistributedMesh::maxSkewness() const {
  return comm_.max(geometry::maxSkewness(owner_, neigh_, cellCentre_, faceCentre_));
}

}  // namespace vibeflow
