#include "mesh/DistributedMesh.hpp"
#include "mesh/RawMesh.hpp"
#include "mesh/Geometry.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>

namespace nsflow {
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
    const int ra = part[subGlobal[a]], rb = part[subGlobal[b]];
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
    recvLocal[part[subGlobal[used[i]]]].push_back(nOwned_ + static_cast<Index>(i));
  for (const auto& f : faces) {
    if (f.nei < nOwned_) continue;                  // both mine
    const Index gsub = used[static_cast<std::size_t>(f.nei - nOwned_)];
    sendGlobal[part[subGlobal[gsub]]].push_back(globalId_[f.own]);
  }
  std::vector<int> sr, rr;
  std::vector<std::vector<Index>> si, ri;
  for (auto& [r, v] : sendGlobal) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    sr.push_back(r);
    std::vector<Index> idx;
    idx.reserve(v.size());
    for (Index gc : v) idx.push_back(subOf[gc]);     // owned: subset id == local id
    si.push_back(std::move(idx));
  }
  for (auto& [r, v] : recvLocal) { rr.push_back(r); ri.push_back(std::move(v)); }
  halo_ = std::make_unique<MpiHalo>(comm, std::move(sr), std::move(si),
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

}  // namespace nsflow
