#include "mesh/DistributedMesh.hpp"
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

std::vector<int> DistributedMesh::partition(const Mesh& global, int nParts,
                                            PartitionMethod method) {
  const Index nc = global.nCells();
  std::vector<int> part(nc, 0);
  if (nParts <= 1) return part;

  if (method == PartitionMethod::Linear) {
    for (Index c = 0; c < nc; ++c)
      part[c] = static_cast<int>(static_cast<long long>(c) * nParts / nc);
    return part;
  }

  auto cc = host(global.cellCentre());
  std::vector<std::array<Real, 3>> x(nc);
  for (Index c = 0; c < nc; ++c) x[c] = {cc(c, 0), cc(c, 1), cc(c, 2)};
  std::vector<Index> ids(nc);
  std::iota(ids.begin(), ids.end(), 0);
  rcb(ids, x, 0, static_cast<int>(nc), part, 0, nParts);
  return part;
}

DistributedMesh::DistributedMesh(const Mesh& global, const Comm& comm,
                                 PartitionMethod method)
    : comm_(comm) {
  const Index gnc = global.nCells();
  const auto part = partition(global, comm.size(), method);
  const int me = comm.rank();

  // -- owned cells, in global order so results gather deterministically
  std::vector<Index> localOf(gnc, -1);
  for (Index c = 0; c < gnc; ++c)
    if (part[c] == me) { localOf[c] = static_cast<Index>(globalId_.size()); globalId_.push_back(c); }
  nOwned_ = static_cast<Index>(globalId_.size());

  auto g_own = host(global.owner());
  auto g_nei = host(global.neighbour());
  auto g_fa  = host(global.faceArea());
  auto g_fc  = host(global.faceCentre());
  auto g_bc  = host(global.boundaryCell());
  auto g_ba  = host(global.boundaryArea());
  auto g_bcn = host(global.boundaryCentre());
  auto g_cc  = host(global.cellCentre());
  auto g_cv  = host(global.cellVolume());

  // -- ghosts: remote cells across a face from an owned cell
  std::map<Index, Index> ghostOf;                  // global id -> local id
  std::map<int, std::vector<Index>> recvGlobal;    // rank -> global ids to receive
  auto ghost = [&](Index gc) {
    auto it = ghostOf.find(gc);
    if (it != ghostOf.end()) return it->second;
    const Index lid = nOwned_ + static_cast<Index>(ghostOf.size());
    ghostOf.emplace(gc, lid);
    recvGlobal[part[gc]].push_back(gc);
    return lid;
  };

  struct LFace { Index own, nei, gface; bool flip; };
  std::vector<LFace> faces;
  for (Index f = 0; f < global.nInternalFaces(); ++f) {
    const Index o = g_own(f), n = g_nei(f);
    const bool mineO = part[o] == me, mineN = part[n] == me;
    if (mineO && mineN)      faces.push_back({localOf[o], localOf[n], f, false});
    else if (mineO)          faces.push_back({localOf[o], ghost(n), f, false});
    else if (mineN)          faces.push_back({localOf[n], ghost(o), f, true});
  }
  nFaces_ = static_cast<Index>(faces.size());
  nGhost_ = static_cast<Index>(ghostOf.size());

  std::vector<Index> bnd;
  for (Index f = 0; f < global.nBoundaryFaces(); ++f)
    if (part[g_bc(f)] == me) bnd.push_back(f);
  nBnd_ = static_cast<Index>(bnd.size());
  patches_ = {{"undefined", 0, nBnd_}};

  // -- local views
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
    const Real s = faces[f].flip ? -1.0 : 1.0;
    for (int k = 0; k < 3; ++k) {
      h_fa(f, k) = s * g_fa(faces[f].gface, k);
      h_fc(f, k) = g_fc(faces[f].gface, k);
    }
  }
  auto h_b = Kokkos::create_mirror_view(bCell_);
  auto h_ba = Kokkos::create_mirror_view(bArea_);
  auto h_bc2 = Kokkos::create_mirror_view(bCentre_);
  for (Index f = 0; f < nBnd_; ++f) {
    h_b(f) = localOf[g_bc(bnd[f])];
    for (int k = 0; k < 3; ++k) {
      h_ba(f, k) = g_ba(bnd[f], k); h_bc2(f, k) = g_bcn(bnd[f], k);
    }
  }
  auto h_cc = Kokkos::create_mirror_view(cellCentre_);
  auto h_cv = Kokkos::create_mirror_view(cellVolume_);
  for (Index c = 0; c < nOwned_; ++c) {
    for (int k = 0; k < 3; ++k) h_cc(c, k) = g_cc(globalId_[c], k);
    h_cv(c) = g_cv(globalId_[c]);
  }
  // Ghost geometry is filled directly rather than exchanged: it never changes,
  // and the whole global mesh is already here in this v0 partitioner.
  for (const auto& [gc, lid] : ghostOf) {
    for (int k = 0; k < 3; ++k) h_cc(lid, k) = g_cc(gc, k);
    h_cv(lid) = g_cv(gc);
  }

  Kokkos::deep_copy(owner_, h_o); Kokkos::deep_copy(neigh_, h_n);
  Kokkos::deep_copy(faceArea_, h_fa); Kokkos::deep_copy(faceCentre_, h_fc);
  Kokkos::deep_copy(bCell_, h_b); Kokkos::deep_copy(bArea_, h_ba);
  Kokkos::deep_copy(bCentre_, h_bc2);
  Kokkos::deep_copy(cellCentre_, h_cc); Kokkos::deep_copy(cellVolume_, h_cv);

  // -- halo schedule. The send list mirrors the receive list: rank r needs my
  // cell c exactly when c is a ghost of r, which is symmetric across the face.
  std::map<int, std::vector<Index>> sendGlobal;
  for (Index f = 0; f < global.nInternalFaces(); ++f) {
    const Index o = g_own(f), n = g_nei(f);
    if (part[o] == me && part[n] != me) sendGlobal[part[n]].push_back(o);
    if (part[n] == me && part[o] != me) sendGlobal[part[o]].push_back(n);
  }
  std::vector<int> sr, rr;
  std::vector<std::vector<Index>> si, ri;
  for (auto& [r, v] : sendGlobal) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    sr.push_back(r);
    std::vector<Index> idx;
    idx.reserve(v.size());
    for (Index gc : v) idx.push_back(localOf[gc]);
    si.push_back(std::move(idx));
  }
  for (auto& [r, v] : recvGlobal) {
    std::sort(v.begin(), v.end());       // same order the sender builds
    rr.push_back(r);
    std::vector<Index> idx;
    idx.reserve(v.size());
    for (Index gc : v) idx.push_back(ghostOf.at(gc));
    ri.push_back(std::move(idx));
  }
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
