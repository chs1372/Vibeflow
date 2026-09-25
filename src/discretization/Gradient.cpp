#include "discretization/Gradient.hpp"
#include "mesh/Mesh.hpp"

namespace vibeflow {

LeastSquaresGradient::LeastSquaresGradient(const Mesh& mesh)
    : m_(mesh),
      Ainv_("Ainv", mesh.nTotal(), 3, 3),
      dInt_("dInt", mesh.nInternalFaces(), 3),
      dBnd_("dBnd", mesh.nBoundaryFaces(), 3),
      wInt_("wInt", mesh.nInternalFaces()),
      wBnd_("wBnd", mesh.nBoundaryFaces()) {
  const Index nc = mesh.nCells(), nt = mesh.nTotal();
  const Index nf = mesh.nInternalFaces(), nb = mesh.nBoundaryFaces();
  auto own = mesh.owner(); auto nei = mesh.neighbour(); auto bc = mesh.boundaryCell();
  auto cc = mesh.cellCentre(); auto bcen = mesh.boundaryCentre();
  auto d = dInt_, db = dBnd_; auto w = wInt_, wb = wBnd_;

  Kokkos::View<Real***, MemSpace> A("A", nt, 3, 3);

  Kokkos::parallel_for("lsqInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      Real dd = 0.0;
      for (int k = 0; k < 3; ++k) {
        d(f, k) = cc(nei(f), k) - cc(own(f), k);
        dd += d(f, k) * d(f, k);
      }
      w(f) = 1.0 / dd;                       // inverse-distance-squared weighting
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          const Real v = w(f) * d(f, a) * d(f, b);
          Kokkos::atomic_add(&A(own(f), a, b), v);
          Kokkos::atomic_add(&A(nei(f), a, b), v);
        }
    });
  Kokkos::parallel_for("lsqBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      Real dd = 0.0;
      for (int k = 0; k < 3; ++k) {
        db(f, k) = bcen(f, k) - cc(bc(f), k);
        dd += db(f, k) * db(f, k);
      }
      wb(f) = 1.0 / dd;
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          Kokkos::atomic_add(&A(bc(f), a, b), wb(f) * db(f, a) * db(f, b));
    });
  Kokkos::fence();

  // Only OWNED cells have a complete face stencil, so only they get an
  // inverse. A ghost cell sees just the faces it shares with this rank; its
  // normal matrix is singular and inverting it would produce inf. Ghost
  // gradients come from the halo exchange instead.
  auto Ai = Ainv_;
  Kokkos::deep_copy(Ai, 0.0);
  Kokkos::parallel_for("invert3", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index c) {
      const Real a = A(c,0,0), b = A(c,0,1), cc3 = A(c,0,2);
      const Real d0 = A(c,1,0), e = A(c,1,1), f2 = A(c,1,2);
      const Real g = A(c,2,0), h = A(c,2,1), i = A(c,2,2);
      const Real det = a*(e*i - f2*h) - b*(d0*i - f2*g) + cc3*(d0*h - e*g);
      const Real id = 1.0 / det;
      Ai(c,0,0) =  (e*i - f2*h) * id;  Ai(c,0,1) = -(b*i - cc3*h) * id;  Ai(c,0,2) =  (b*f2 - cc3*e) * id;
      Ai(c,1,0) = -(d0*i - f2*g) * id; Ai(c,1,1) =  (a*i - cc3*g) * id;  Ai(c,1,2) = -(a*f2 - cc3*d0) * id;
      Ai(c,2,0) =  (d0*h - e*g) * id;  Ai(c,2,1) = -(a*h - b*g) * id;    Ai(c,2,2) =  (a*e - b*d0) * id;
    });
  Kokkos::fence();
}

void LeastSquaresGradient::operator()(const ScalarField& phi, const ScalarField& phiB,
                                      VectorField& grad) const {
  const Index nc = m_.nCells(), nt = m_.nTotal();
  const Index nf = m_.nInternalFaces(), nb = m_.nBoundaryFaces();
  auto own = m_.owner(); auto nei = m_.neighbour(); auto bc = m_.boundaryCell();
  auto d = dInt_, db = dBnd_; auto w = wInt_, wb = wBnd_; auto Ai = Ainv_;

  VectorField rhs("lsqRhs", nt, 3);
  Kokkos::parallel_for("lsqRhsInt", Kokkos::RangePolicy<ExecSpace>(0, nf),
    KOKKOS_LAMBDA(const Index f) {
      const Real s = w(f) * (phi(nei(f)) - phi(own(f)));
      for (int k = 0; k < 3; ++k) {
        Kokkos::atomic_add(&rhs(own(f), k), s * d(f, k));
        Kokkos::atomic_add(&rhs(nei(f), k), s * d(f, k));
      }
    });
  Kokkos::parallel_for("lsqRhsBnd", Kokkos::RangePolicy<ExecSpace>(0, nb),
    KOKKOS_LAMBDA(const Index f) {
      const Real s = wb(f) * (phiB(f) - phi(bc(f)));
      for (int k = 0; k < 3; ++k) Kokkos::atomic_add(&rhs(bc(f), k), s * db(f, k));
    });
  Kokkos::fence();

  Kokkos::parallel_for("lsqApply", Kokkos::RangePolicy<ExecSpace>(0, nc),
    KOKKOS_LAMBDA(const Index c) {
      for (int a = 0; a < 3; ++a) {
        Real acc = 0.0;
        for (int b = 0; b < 3; ++b) acc += Ai(c, a, b) * rhs(c, b);
        grad(c, a) = acc;
      }
    });
  Kokkos::fence();

  // The non-orthogonal correction interpolates the gradient onto faces, so a
  // face on a rank boundary needs the neighbour's gradient, not a local guess.
  if (const auto* h = m_.halo()) h->exchange(grad);
}

}  // namespace vibeflow
