#pragma once
// MPI communicator wrapper, with a serial fallback.
//
// Everything above this uses Comm, never MPI directly, so a serial build needs
// no MPI installed and the parallel-consistency gate can compare the two.

#include "core/Types.hpp"
#include <vector>

#ifdef VIBEFLOW_HAVE_MPI
#include <mpi.h>
#endif

namespace vibeflow {

class Comm {
 public:
  static Comm world();

  int rank() const { return rank_; }
  int size() const { return size_; }
  bool parallel() const { return size_ > 1; }

  Real  sum(Real v) const;
  Real  max(Real v) const;
  Real  min(Real v) const;
  Index sum(Index v) const;
  long long max(long long v) const;

  void barrier() const;

  // Personalised all-to-all: element r of `send` goes to rank r, element r of
  // the result came from rank r. The distributed mesh reader is built on
  // these; the solver never needs them.
  std::vector<std::vector<long long>>
  alltoallv(const std::vector<std::vector<long long>>& send) const;
  std::vector<std::vector<double>>
  alltoallv(const std::vector<std::vector<double>>& send) const;

  // Every rank's `mine`, concatenated in rank order, on every rank.
  std::vector<long long> allgatherv(const std::vector<long long>& mine) const;

  // Non-blocking neighbour exchange used by the halo.
  void exchange(const std::vector<int>& sendRanks,
                const std::vector<std::vector<Real>>& sendBufs,
                const std::vector<int>& recvRanks,
                std::vector<std::vector<Real>>& recvBufs) const;

 private:
  int rank_{0}, size_{1};
};

// RAII initialiser. Safe to construct in a serial build.
struct ParallelScope {
  ParallelScope(int& argc, char**& argv);
  ~ParallelScope();
};

}  // namespace vibeflow
