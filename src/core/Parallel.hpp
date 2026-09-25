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
  Index sum(Index v) const;

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
