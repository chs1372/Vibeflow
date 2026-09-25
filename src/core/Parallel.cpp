#include "core/Parallel.hpp"

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
