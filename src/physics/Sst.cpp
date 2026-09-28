// The k-omega SST model (ADR-042) and the wall distance. Stubs: the gates
// that judge them are committed before the code.

#include "physics/Piso.hpp"
#include "physics/WallDistance.hpp"
#include <stdexcept>

namespace vibeflow {

namespace {
[[noreturn]] void notYet() {
  throw std::runtime_error("the SST model is not implemented (ADR-042)");
}
}  // namespace

ScalarField wallDistance(const Mesh&, const View1<int>&, const VectorField&, Index,
                         const Comm&) { notYet(); }

void PisoSolver::enableTurbulence(const TurbulenceModel&, const View1<int>&) { notYet(); }
void PisoSolver::setTurbulenceBoundary(const View1<int>&, const ScalarField&,
                                       const ScalarField&) { notYet(); }
void PisoSolver::setTurbulenceSource(const ScalarField&, const ScalarField&) { notYet(); }
void PisoSolver::setTurbulence(const ScalarField&, const ScalarField&) { notYet(); }
void PisoSolver::advanceTurbulenceFrozen(const VectorField&, const ScalarField&,
                                         LinearSolver&) { notYet(); }
void PisoSolver::solveTurbulence(LinearSolver&, const VectorField&) { notYet(); }
void PisoSolver::updateEddyViscosity(const VectorField&) { notYet(); }

}  // namespace vibeflow
