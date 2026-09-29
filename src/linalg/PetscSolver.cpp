#include "linalg/PetscSolver.hpp"
#include "linalg/LinearSystem.hpp"
#include "mesh/Mesh.hpp"

#include <petscksp.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace vibeflow {
namespace {

void chk(PetscErrorCode e, const char* what) {
  if (e) throw std::runtime_error(std::string("PETSc ") + what);
}

}  // namespace

struct PetscSolver::Impl {
  const Mesh& mesh;
  Comm comm;
  std::vector<Index> grow;       // local cell -> global row
  Index nOwned{}, firstRow{};
  Mat A{nullptr};
  Vec b{nullptr}, u{nullptr};
  KSP ksp{nullptr};
  bool dirty{true};
  int pcInterval{1};
  int matrixChanges{0};
  bool constantNullSpace{false};   // ADR-044
  bool nullSpaceSet{false};

  Impl(const Mesh& m, Comm c, std::vector<Index> g) : mesh(m), comm(c), grow(std::move(g)) {
    nOwned = mesh.nCells();
    if (grow.empty()) {                       // serial: rows are local ids
      grow.resize(mesh.nTotal());
      std::iota(grow.begin(), grow.end(), 0);
      firstRow = 0;
    } else {
      firstRow = grow[0];
    }
    const Index nGlobal = comm.sum(nOwned);

    MPI_Comm mc = PETSC_COMM_SELF;
#ifdef VIBEFLOW_HAVE_MPI
    if (comm.parallel()) mc = MPI_COMM_WORLD;
#endif
    chk(MatCreate(mc, &A), "MatCreate");
    chk(MatSetSizes(A, nOwned, nOwned, nGlobal, nGlobal), "MatSetSizes");
    chk(MatSetType(A, comm.parallel() ? MATMPIAIJ : MATSEQAIJ), "MatSetType");
    // A hex cell has at most 6 neighbours, so 7 per row covers the diagonal too.
    chk(MatSeqAIJSetPreallocation(A, 7, nullptr), "SeqPrealloc");
    chk(MatMPIAIJSetPreallocation(A, 7, nullptr, 6, nullptr), "MPIPrealloc");
    chk(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE), "MatSetOption");

    chk(VecCreate(mc, &b), "VecCreate");
    chk(VecSetSizes(b, nOwned, nGlobal), "VecSetSizes");
    chk(VecSetFromOptions(b), "VecSetFromOptions");
    chk(VecDuplicate(b, &u), "VecDuplicate");
    chk(KSPCreate(mc, &ksp), "KSPCreate");
  }

  ~Impl() {
    if (ksp) KSPDestroy(&ksp);
    if (u) VecDestroy(&u);
    if (b) VecDestroy(&b);
    if (A) MatDestroy(&A);
  }

  void configure(const std::string& cfg) {
    const auto plus = cfg.find('+');
    const std::string solver = cfg.substr(0, plus);
    const std::string pc = plus == std::string::npos ? "jacobi" : cfg.substr(plus + 1);

    if (solver == "cg")         chk(KSPSetType(ksp, KSPCG), "KSPSetType");
    else if (solver == "gmres") chk(KSPSetType(ksp, KSPGMRES), "KSPSetType");
    else if (solver == "bicgstab") chk(KSPSetType(ksp, KSPBCGS), "KSPSetType");
    else throw std::runtime_error("unknown KSP type: " + solver);

    PC p;
    chk(KSPGetPC(ksp, &p), "KSPGetPC");
    if (pc == "jacobi")      chk(PCSetType(p, PCJACOBI), "PCSetType");
    else if (pc == "ilu")    chk(PCSetType(p, PCILU), "PCSetType");
    else if (pc == "gamg")   chk(PCSetType(p, PCGAMG), "PCSetType");
    else if (pc == "hypre") {
      if (!PetscSolver::hasHypre())
        throw std::runtime_error(
            "case asked for hypre but this PETSc was built without it");
      chk(PCSetType(p, PCHYPRE), "PCSetType");
      chk(PCHYPRESetType(p, "boomeramg"), "PCHYPRESetType");
    } else throw std::runtime_error("unknown PC type: " + pc);
  }

  // Face-based storage -> CSR rows, owned rows only.
  void assemble(LinearSystem& sys) {
    auto h_diag = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), sys.diag());
    auto h_up   = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), sys.upper());
    auto h_lo   = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), sys.lower());
    auto h_own  = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.owner());
    auto h_nei  = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), mesh.neighbour());

    chk(MatZeroEntries(A), "MatZeroEntries");
    for (Index c = 0; c < nOwned; ++c) {
      const PetscInt r = grow[c];
      const PetscScalar v = h_diag(c);
      chk(MatSetValues(A, 1, &r, 1, &r, &v, ADD_VALUES), "MatSetValues diag");
    }
    for (Index f = 0; f < mesh.nInternalFaces(); ++f) {
      const Index o = h_own(f), n = h_nei(f);
      if (o < nOwned) {
        const PetscInt r = grow[o], c = grow[n];
        const PetscScalar v = h_up(f);
        chk(MatSetValues(A, 1, &r, 1, &c, &v, ADD_VALUES), "MatSetValues upper");
      }
      if (n < nOwned) {
        const PetscInt r = grow[n], c = grow[o];
        const PetscScalar v = h_lo(f);
        chk(MatSetValues(A, 1, &r, 1, &c, &v, ADD_VALUES), "MatSetValues lower");
      }
    }
    chk(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY), "AssemblyBegin");
    chk(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY), "AssemblyEnd");
    // The null space is an attribute of the matrix and survives
    // re-assembly, so it is attached once.
    if (constantNullSpace && !nullSpaceSet) {
      MatNullSpace ns;
      chk(MatNullSpaceCreate(PetscObjectComm(reinterpret_cast<PetscObject>(A)), PETSC_TRUE, 0,
                             nullptr, &ns), "MatNullSpaceCreate");
      chk(MatSetNullSpace(A, ns), "MatSetNullSpace");
      chk(MatNullSpaceDestroy(&ns), "MatNullSpaceDestroy");
      nullSpaceSet = true;
    }
    dirty = false;
  }
};

bool PetscSolver::hasHypre() {
#ifdef PETSC_HAVE_HYPRE
  return true;
#else
  return false;
#endif
}

PetscSolver::PetscSolver(const Mesh& mesh, Comm comm, std::string config,
                         std::vector<Index> globalRowOf, int pcRebuildInterval)
    : impl_(std::make_unique<Impl>(mesh, comm, std::move(globalRowOf))),
      config_(std::move(config)) {
  impl_->pcInterval = std::max(1, pcRebuildInterval);
  impl_->configure(config_);
}

PetscSolver::~PetscSolver() = default;

SolveReport PetscSolver::solve(LinearSystem& sys, ScalarField& x,
                               Real relTol, Real absTol, int maxIter) {
  auto& I = *impl_;
  // Re-assembling here would also force PETSc to rebuild the preconditioner.
  // For BoomerAMG that setup is most of the cost, and the pressure corrector
  // reuses one matrix for every sweep of a step.
  if (I.dirty) {
    I.assemble(sys);
    // Rebuild the preconditioner on the first solve and then only every
    // pcInterval-th matrix change; reuse it otherwise.
    const bool rebuild = (I.matrixChanges % I.pcInterval) == 0;
    chk(KSPSetReusePreconditioner(I.ksp, rebuild ? PETSC_FALSE : PETSC_TRUE),
        "KSPSetReusePreconditioner");
    ++I.matrixChanges;
  }

  auto h_src = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), sys.source());
  auto h_x   = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), x);
  for (Index c = 0; c < I.nOwned; ++c) {
    const PetscInt r = I.grow[c];
    PetscScalar vb = h_src(c), vx = h_x(c);
    chk(VecSetValues(I.b, 1, &r, &vb, INSERT_VALUES), "VecSetValues b");
    chk(VecSetValues(I.u, 1, &r, &vx, INSERT_VALUES), "VecSetValues u");
  }
  for (Vec v : {I.b, I.u}) {
    chk(VecAssemblyBegin(v), "VecAssemblyBegin");
    chk(VecAssemblyEnd(v), "VecAssemblyEnd");
  }

  chk(KSPSetOperators(I.ksp, I.A, I.A), "KSPSetOperators");
  chk(KSPSetTolerances(I.ksp, relTol, absTol, PETSC_DEFAULT, maxIter), "KSPSetTolerances");
  chk(KSPSetInitialGuessNonzero(I.ksp, PETSC_TRUE), "KSPSetInitialGuessNonzero");

  PetscLogDouble t0 = 0, t1 = 0;
  PetscTime(&t0);
  chk(KSPSolve(I.ksp, I.b, I.u), "KSPSolve");
  PetscTime(&t1);

  SolveReport rep;
  PetscInt its = 0;
  PetscReal rnorm = 0.0;
  KSPConvergedReason reason;
  KSPGetIterationNumber(I.ksp, &its);
  KSPGetResidualNorm(I.ksp, &rnorm);
  KSPGetConvergedReason(I.ksp, &reason);
  rep.iterations = static_cast<int>(its);
  rep.finalResidual = rnorm;
  rep.converged = reason > 0;
  rep.wallSeconds = t1 - t0;

  const PetscScalar* arr = nullptr;
  chk(VecGetArrayRead(I.u, &arr), "VecGetArrayRead");
  for (Index c = 0; c < I.nOwned; ++c) h_x(c) = PetscRealPart(arr[c]);
  chk(VecRestoreArrayRead(I.u, &arr), "VecRestoreArrayRead");
  Kokkos::deep_copy(x, h_x);
  if (const auto* h = I.mesh.halo()) h->exchange(x);
  record(rep);
  return rep;
}

void PetscSolver::notifyMatrixChanged() { impl_->dirty = true; }

void PetscSolver::setConstantNullSpace(bool on) {
  impl_->constantNullSpace = on;
  if (!on && impl_->nullSpaceSet) {
    chk(MatSetNullSpace(impl_->A, nullptr), "MatSetNullSpace");
    impl_->nullSpaceSet = false;
  }
}

void PetscSolver::setUnpreconditionedNorm(bool on) {
  chk(KSPSetNormType(impl_->ksp, on ? KSP_NORM_UNPRECONDITIONED : KSP_NORM_DEFAULT),
      "KSPSetNormType");
}

void PetscSolver::setSymmetricAMG() {
  PC p;
  chk(KSPGetPC(impl_->ksp, &p), "KSPGetPC");
  PetscBool isHypre = PETSC_FALSE;
  chk(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(p), PCHYPRE, &isHypre),
      "PetscObjectTypeCompare");
  if (!isHypre) return;
  // A prefix of this solver's own, so that the options reach no other KSP.
  static int serial = 0;
  const std::string prefix = "vfsym" + std::to_string(serial++) + "_";
  chk(KSPSetOptionsPrefix(impl_->ksp, prefix.c_str()), "KSPSetOptionsPrefix");
  const std::string o = "-" + prefix + "pc_hypre_boomeramg_relax_type_";
  chk(PetscOptionsSetValue(nullptr, (o + "all").c_str(), "symmetric-SOR/Jacobi"),
      "PetscOptionsSetValue");
  chk(PetscOptionsSetValue(nullptr, (o + "coarse").c_str(), "symmetric-SOR/Jacobi"),
      "PetscOptionsSetValue");
  chk(PCSetFromOptions(p), "PCSetFromOptions");
}

}  // namespace vibeflow
