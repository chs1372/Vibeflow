#pragma once
// Fundamental types and memory layout. Everything above this layer allocates
// through these aliases so a change of execution space is a one-line change here.

#include <Kokkos_Core.hpp>
#include <cstdint>

namespace nsflow {

using Real  = double;
using Index = std::int32_t;   // local indices; global ids are Int64
using Int64 = std::int64_t;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using HostSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace  = ExecSpace::memory_space;

// Structure-of-arrays throughout. LayoutLeft on device (coalesced across cells),
// LayoutRight on host. Kokkos picks this per space; do not hard-code a layout.
template <class T> using View1  = Kokkos::View<T*,  MemSpace>;
template <class T> using View2  = Kokkos::View<T**, MemSpace>;
template <class T> using HView1 = typename View1<T>::HostMirror;
template <class T> using HView2 = typename View2<T>::HostMirror;

using ScalarField = View1<Real>;   // (nCells)
using VectorField = View2<Real>;   // (nCells, 3)

struct Vec3 {
  Real x{}, y{}, z{};
  KOKKOS_INLINE_FUNCTION Real dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
  KOKKOS_INLINE_FUNCTION Real mag2() const { return dot(*this); }
};

}  // namespace nsflow
