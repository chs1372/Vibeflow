#include "core/Parallel.hpp"

#include <algorithm>
#include <stdexcept>

namespace vibeflow {

ParallelScope::ParallelScope(int& argc, char**& argv) {
#ifdef VIBEFLOW_HAVE_MPI
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) MPI_Init(&argc, &argv);
#else
  (void)argc; (void)argv;
#endif
}

ParallelScope::~ParallelScope() {
#ifdef VIBEFLOW_HAVE_MPI
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) MPI_Finalize();
#endif
}

Comm Comm::world() {
  Comm c;
#ifdef VIBEFLOW_HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &c.rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &c.size_);
#endif
  return c;
}

Real Comm::sum(Real v) const {
#ifdef VIBEFLOW_HAVE_MPI
  if (size_ > 1) { Real o = 0.0; MPI_Allreduce(&v, &o, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); return o; }
#endif
  return v;
}

Real Comm::max(Real v) const {
#ifdef VIBEFLOW_HAVE_MPI
  if (size_ > 1) { Real o = 0.0; MPI_Allreduce(&v, &o, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); return o; }
#endif
  return v;
}

Real Comm::min(Real v) const {
#ifdef VIBEFLOW_HAVE_MPI
  if (size_ > 1) { Real o = 0.0; MPI_Allreduce(&v, &o, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); return o; }
#endif
  return v;
}

long long Comm::max(long long v) const {
#ifdef VIBEFLOW_HAVE_MPI
  if (size_ > 1) { long long o = 0; MPI_Allreduce(&v, &o, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD); return o; }
#endif
  return v;
}

void Comm::barrier() const {
#ifdef VIBEFLOW_HAVE_MPI
  if (size_ > 1) MPI_Barrier(MPI_COMM_WORLD);
#endif
}

namespace {

#ifdef VIBEFLOW_HAVE_MPI
template <class T> MPI_Datatype mpiType();
template <> MPI_Datatype mpiType<long long>() { return MPI_LONG_LONG; }
template <> MPI_Datatype mpiType<double>() { return MPI_DOUBLE; }
#endif

template <class T>
std::vector<std::vector<T>> alltoallvImpl(const std::vector<std::vector<T>>& send,
                                          int size) {
  std::vector<std::vector<T>> recv(static_cast<std::size_t>(size));
  if (size <= 1) {
    if (!send.empty()) recv[0] = send[0];
    return recv;
  }
#ifdef VIBEFLOW_HAVE_MPI
  // Counts first, then one flat exchange. int counts: fine for any mesh that
  // fits a workstation, and checked so a larger one fails loudly.
  std::vector<int> sc(static_cast<std::size_t>(size)), rc(static_cast<std::size_t>(size));
  std::vector<int> sd(static_cast<std::size_t>(size) + 1, 0), rd(static_cast<std::size_t>(size) + 1, 0);
  for (int r = 0; r < size; ++r) {
    const std::size_t n = r < static_cast<int>(send.size()) ? send[r].size() : 0;
    if (n > 2147483647u) throw std::runtime_error("alltoallv: message too large");
    sc[r] = static_cast<int>(n);
  }
  MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, MPI_COMM_WORLD);
  for (int r = 0; r < size; ++r) { sd[r + 1] = sd[r] + sc[r]; rd[r + 1] = rd[r] + rc[r]; }
  std::vector<T> sb(static_cast<std::size_t>(sd[size])), rb(static_cast<std::size_t>(rd[size]));
  for (int r = 0; r < size; ++r)
    if (sc[r]) std::copy(send[r].begin(), send[r].end(), sb.begin() + sd[r]);
  MPI_Alltoallv(sb.data(), sc.data(), sd.data(), mpiType<T>(),
                rb.data(), rc.data(), rd.data(), mpiType<T>(), MPI_COMM_WORLD);
  for (int r = 0; r < size; ++r)
    recv[r].assign(rb.begin() + rd[r], rb.begin() + rd[r + 1]);
#endif
  return recv;
}

}  // namespace

std::vector<std::vector<long long>>
Comm::alltoallv(const std::vector<std::vector<long long>>& send) const {
  return alltoallvImpl(send, size_);
}
std::vector<std::vector<double>>
Comm::alltoallv(const std::vector<std::vector<double>>& send) const {
  return alltoallvImpl(send, size_);
}

std::vector<long long> Comm::allgatherv(const std::vector<long long>& mine) const {
  if (size_ <= 1) return mine;
  std::vector<long long> all;
#ifdef VIBEFLOW_HAVE_MPI
  int n = static_cast<int>(mine.size());
  std::vector<int> counts(static_cast<std::size_t>(size_)), displ(static_cast<std::size_t>(size_) + 1, 0);
  MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  for (int r = 0; r < size_; ++r) displ[r + 1] = displ[r] + counts[r];
  all.resize(static_cast<std::size_t>(displ[size_]));
  MPI_Allgatherv(mine.data(), n, MPI_LONG_LONG, all.data(), counts.data(), displ.data(),
                 MPI_LONG_LONG, MPI_COMM_WORLD);
#endif
  return all;
}

Index Comm::sum(Index v) const {
#ifdef VIBEFLOW_HAVE_MPI
  if (size_ > 1) { long long i = v, o = 0; MPI_Allreduce(&i, &o, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
                   return static_cast<Index>(o); }
#endif
  return v;
}

void Comm::exchange(const std::vector<int>& sendRanks,
                    const std::vector<std::vector<Real>>& sendBufs,
                    const std::vector<int>& recvRanks,
                    std::vector<std::vector<Real>>& recvBufs) const {
#ifdef VIBEFLOW_HAVE_MPI
  if (size_ <= 1) return;
  std::vector<MPI_Request> reqs;
  reqs.reserve(sendRanks.size() + recvRanks.size());
  for (std::size_t i = 0; i < recvRanks.size(); ++i) {
    reqs.emplace_back();
    MPI_Irecv(recvBufs[i].data(), static_cast<int>(recvBufs[i].size()), MPI_DOUBLE,
              recvRanks[i], 7, MPI_COMM_WORLD, &reqs.back());
  }
  for (std::size_t i = 0; i < sendRanks.size(); ++i) {
    reqs.emplace_back();
    MPI_Isend(sendBufs[i].data(), static_cast<int>(sendBufs[i].size()), MPI_DOUBLE,
              sendRanks[i], 7, MPI_COMM_WORLD, &reqs.back());
  }
  MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
#else
  (void)sendRanks; (void)sendBufs; (void)recvRanks; (void)recvBufs;
#endif
}

}  // namespace vibeflow
