// The backward-facing step, NASA TMR's 2D validation case, with the k-omega
// SST model (ADR-045). One grid per run; the gate's comparisons across grids
// and against TMR's CFL3D results are backstep_gate.py's.
//
// Re_H = 36,000 on the reference velocity U = 1, nu = 1/36,000; the step
// height H = 1. The TMR grid (cases/backstep/make_mesh.py), four zones merged
// and extruded one cell in z, spans -130 <= x <= 50; 1 <= y <= 9 upstream of
// the step at x = 0, 0 <= y <= 9 behind it:
//   inlet     x = -130   u = (U_in, 0, 0), k = 5.49e-7, omega = 2.197 (TMR's
//                        free stream for SST at M = 0.128, nu_t/nu = 0.009)
//   walls     y = 1 (x < 0), y = 0 (x > 0), the step's face x = 0 and the top
//             y = 9: slip for x < -110, as TMR's grids mark them, no slip
//             after; k = 0, omega Menter's wall value on the no-slip part
//   outlet    x = 50     pressure outlet p = 0
//   z faces              zero gradient, as the flat plate's slab
// U_in is the inlet speed, given on the command line: ADR-045 sets it once,
// on level 3, so that the channel's centre velocity at x = -4 is CFL3D's.
//
// From the uniform stream, or from another march's state (VIBEFLOW_BS_INIT,
// each cell taking the state of the nearest cell there), with eight PISO
// correctors (ADR-045's fifth revision). The step starts at dt/64 from the
// uniform stream, dt/8 from a state, and follows the largest cell Courant
// number C (the third revision): every 100 steps dt doubles, never past the
// level's dt nor to a dt it was halved from after its first 100 steps (the
// fifth and seventh revisions), if C at the doubled dt is at most 3; whenever
// C exceeds 6 it halves and holds for 100 steps (the sixth revision's
// thresholds). Once the march has held its dt for 1,000 steps -- the level's,
// or the one it may not double -- it ends at the steady criteria of the flat
// plate (u, k and omega
// changing by less than 1e-6 of their size over 50 steps) or ADR-045's
// quasi-steady one: the reattachment point and Cf at x = -4 moving by no more
// than 0.2% over the march's last 20%. Otherwise it ends after the given
// number of steps at that dt, or 20,000 in all. Every 500 steps it keeps a
// checkpoint, <out>.ckpt, from which VIBEFLOW_BS_RESUME continues it.
//
// Written out, for the gate:
//   <out>.cf        x, Cf on the bottom wall (y = 1 upstream, y = 0 behind),
//                   per wall face, on (1/2) rho U^2
//   <out>.cp        x, Cp there, from the wall cells' pressure, shifted to 0 at
//                   x = 40 as TMR shifts its own
//   <out>.prof      x_station, y, u at x = -4, 1, 4, 6, 10: each pair of cells
//                   straddling the station across an x-facing face,
//                   interpolated linearly in x
//   <out>.cpw       x, Cp from the extrapolated wall pressure (reported only)
//   <out>.state     x y u v w p k omega per cell (VIBEFLOW_BS_SAVE)
//   and one line    FINAL xr ... cf4 ... uc4 ... bubble ... steps ... settled ...
// and on stdout the lip print of ADR-046's first revision: LIPX (each process
// start) with the x of the last four upstream bottom faces, and LIP (every
// 50-step line) with step, p40 and those faces' Cp, computed as .cp is.
//
// Run:  backstep <mesh.hex> <out prefix> [dt, default 0.08] [max steps at the
//       full dt, default 5000] [U_in, default 1]
//
// Exploration only, not used by the gate: VIBEFLOW_BS_DEBUG (each step's
// extremes and where they are), VIBEFLOW_BS_SAVE_AT=<step>[,...] (the state
// after those steps too), VIBEFLOW_BS_STOP=<step>, VIBEFLOW_BS_RAMP=<n> (start
// at dt/n), VIBEFLOW_BS_CUP / VIBEFLOW_BS_CDOWN (the ramp's thresholds),
// VIBEFLOW_BS_FROZEN, VIBEFLOW_BS_NO_TRANSPOSE, VIBEFLOW_BS_CKPT_EVERY,
// VIBEFLOW_CONVECTION, VIBEFLOW_NO_NONORTH, VIBEFLOW_CORRECTORS.
//
// ADR-047's equivalence gates only; never a physics verdict:
// VIBEFLOW_BS_SWEEPS=<n> (n non-orthogonal sweeps, nonOrthTol 0) and
// VIBEFLOW_BS_KEEP_CKPT (the checkpoint kept when the march ends).
//
// Under MPI (ADR-047), more than one process: the mesh is distributed by RCB,
// every decision is taken from the same reduced or gathered data on every
// rank, rank 0 writes every file in the formats above (.state in the file's
// cell order), and each rank keeps its own checkpoint, <out>.ckpt.<rank>of<P>,
// renamed from its .tmp only once every rank has written one. A resume never
// crosses rank counts, and every rank's file must hold the same step, time,
// dt, ramp state and histories: otherwise it is REFUSED (exit 3). Every 50 steps
// an EQ line gives the march's decision values at full precision, and before
// FINAL a SOLVES line the linear solvers' counts; both serially too.

#include "mesh/PolyMesh.hpp"
#include "mesh/RawMesh.hpp"
#include "mesh/DistributedMesh.hpp"
#include "physics/Piso.hpp"
#include "linalg/NativeBiCGStab.hpp"
#include "linalg/NativeCG.hpp"
#ifdef VIBEFLOW_HAVE_PETSC
#include "linalg/PetscSolver.hpp"
#include <petscsys.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace vibeflow;

namespace {

constexpr Real NU = 1.0 / 36000.0, K_IN = 5.493e-7, W_IN = 2.197;
constexpr Real X_SLIP = -110.0;               // slip walls upstream of this
constexpr Real X_CF4 = -4.0;                  // the upstream station
const Real STATIONS[] = {-4.0, 1.0, 4.0, 6.0, 10.0};

template <class V> auto host(const V& v) {
  return Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), v);
}

std::unique_ptr<LinearSolver> makePressureSolver(const Mesh& mesh) {
  const char* e = std::getenv("VIBEFLOW_PRESSURE");
#ifdef VIBEFLOW_HAVE_PETSC
  const std::string cfg = e ? e : "cg+hypre";
#else
  const std::string cfg = e ? e : "native";
#endif
  if (cfg == "native") return std::make_unique<NativeCG>(mesh, Comm(), false);
#ifdef VIBEFLOW_HAVE_PETSC
  return std::make_unique<PetscSolver>(mesh, Comm(), cfg);
#else
  std::fprintf(stderr, "built without PETSc; VIBEFLOW_PRESSURE=%s unavailable\n", cfg.c_str());
  std::exit(2);
#endif
}

enum Patch { INLET, OUTLET, SLIP, WALL, SPAN };

// The bottom wall's faces, upstream (y = 1) and behind the step (y = 0),
// sorted by x.
struct Bottom { std::vector<Index> face; std::vector<Real> x; };

// Wall stress on each bottom face from the two-point gradient the momentum
// equation uses there: nu u_x / d_n, signed, so the recirculation is negative.
std::vector<Real> wallStress(const PolyMesh& mesh, const PisoSolver& solver, const Bottom& w) {
  auto u = host(solver.velocity());
  auto bc = host(mesh.boundaryCell()); auto bcen = host(mesh.boundaryCentre());
  auto ba = host(mesh.boundaryArea()); auto cc = host(mesh.cellCentre());
  std::vector<Real> tau(w.face.size());
  for (std::size_t i = 0; i < w.face.size(); ++i) {
    const Index f = w.face[i], c = bc(f);
    const Real mag = std::sqrt(ba(f,0)*ba(f,0) + ba(f,1)*ba(f,1) + ba(f,2)*ba(f,2));
    Real dn = 0.0;
    for (int k = 0; k < 3; ++k) dn += (bcen(f,k) - cc(c,k)) * ba(f,k) / mag;
    tau[i] = NU * u(c, 0) / std::abs(dn);
  }
  return tau;
}

Real interpAt(const std::vector<Real>& x, const std::vector<Real>& v, Real at) {
  for (std::size_t i = 1; i < x.size(); ++i)
    if (x[i] >= at) {
      const Real t = (at - x[i-1]) / (x[i] - x[i-1]);
      return (1.0 - t) * v[i-1] + t * v[i];
    }
  return v.back();
}

// The reattachment point: where Cf last changes sign from negative to
// positive behind the step, linear between faces; the corner bubble: the
// first two sign changes behind the step, if Cf starts positive there.
struct Separation { Real xr = -1.0, c0 = -1.0, c1 = -1.0; };
Separation separation(const std::vector<Real>& x, const std::vector<Real>& cf) {
  Separation s;
  std::vector<Real> zs;
  for (std::size_t i = 1; i < x.size(); ++i) {
    if (x[i-1] < 0.0) continue;
    if ((cf[i-1] < 0.0) != (cf[i] < 0.0))
      zs.push_back(x[i-1] - cf[i-1] * (x[i] - x[i-1]) / (cf[i] - cf[i-1]));
  }
  for (std::size_t i = 1; i < x.size(); ++i) {
    if (x[i-1] < 0.0) continue;
    if (cf[i-1] < 0.0 && cf[i] >= 0.0)
      s.xr = x[i-1] - cf[i-1] * (x[i] - x[i-1]) / (cf[i] - cf[i-1]);
  }
  if (zs.size() >= 3) { s.c0 = zs[0]; s.c1 = zs[1]; }
  return s;
}

// A coarser grid's state and a bucket grid over its cell centres, for the
// nearest coarse cell to a point.
struct CoarseState {
  std::vector<std::array<Real, 8>> v;          // x y u v w p k omega
  Real x0 = 0, y0 = 0, hx = 1, hy = 1;
  int nx = 1, ny = 1;
  std::vector<std::vector<Index>> bucket;
};

CoarseState readState(const std::string& path) {
  CoarseState cs;
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f) { std::printf("cannot read %s\n", path.c_str()); std::exit(2); }
  long n = 0;
  if (std::fscanf(f, "%ld", &n) != 1) { std::printf("bad state file %s\n", path.c_str()); std::exit(2); }
  cs.v.resize(n);
  Real xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
  for (long c = 0; c < n; ++c) {
    double a[8];
    for (int i = 0; i < 8; ++i)
      if (std::fscanf(f, "%lf", &a[i]) != 1) { std::printf("bad state file %s\n", path.c_str()); std::exit(2); }
    for (int i = 0; i < 8; ++i) cs.v[c][i] = a[i];
    xmin = std::min(xmin, a[0]); xmax = std::max(xmax, a[0]);
    ymin = std::min(ymin, a[1]); ymax = std::max(ymax, a[1]);
  }
  std::fclose(f);
  cs.nx = 720; cs.ny = 72;
  cs.x0 = xmin; cs.y0 = ymin;
  cs.hx = (xmax - xmin) / cs.nx * (1.0 + 1e-12) + 1e-300;
  cs.hy = (ymax - ymin) / cs.ny * (1.0 + 1e-12) + 1e-300;
  cs.bucket.resize(static_cast<std::size_t>(cs.nx) * cs.ny);
  for (long c = 0; c < n; ++c) {
    const int i = std::min(cs.nx - 1, static_cast<int>((cs.v[c][0] - cs.x0) / cs.hx));
    const int j = std::min(cs.ny - 1, static_cast<int>((cs.v[c][1] - cs.y0) / cs.hy));
    cs.bucket[static_cast<std::size_t>(j) * cs.nx + i].push_back(c);
  }
  return cs;
}

// The nearest coarse centre: the point's bucket and rings around it until a
// ring beyond the best distance so far has been searched.
Index nearestCoarse(const CoarseState& cs, Real x, Real y) {
  const int i0 = std::clamp(static_cast<int>((x - cs.x0) / cs.hx), 0, cs.nx - 1);
  const int j0 = std::clamp(static_cast<int>((y - cs.y0) / cs.hy), 0, cs.ny - 1);
  Index best = -1;
  Real bd = 1e300;
  for (int r = 0; r < std::max(cs.nx, cs.ny); ++r) {
    for (int j = j0 - r; j <= j0 + r; ++j)
      for (int i = i0 - r; i <= i0 + r; ++i) {
        if (i < 0 || j < 0 || i >= cs.nx || j >= cs.ny) continue;
        if (std::max(std::abs(i - i0), std::abs(j - j0)) != r) continue;
        for (Index c : cs.bucket[static_cast<std::size_t>(j) * cs.nx + i]) {
          const Real dx = cs.v[c][0] - x, dy = cs.v[c][1] - y;
          const Real d = dx * dx + dy * dy;
          if (d < bd) { bd = d; best = c; }
        }
      }
    // Every point of ring r+1 is at least r * min(hx, hy) away.
    const Real reach = r * std::min(cs.hx, cs.hy);
    if (best >= 0 && reach * reach > bd) break;
  }
  return best;
}

// The cell columns around a station: the largest cell-centre x below it and
// the smallest above, each column's cells sorted by y.
struct Columns { std::vector<Index> left, right; Real xl = 0, xr = 0; };

Columns columnsAt(const PolyMesh& mesh, Real at) {
  auto cc = host(mesh.cellCentre());
  Real xl = -1e300, xr = 1e300;
  for (Index c = 0; c < mesh.nCells(); ++c) {
    const Real x = cc(c, 0);
    if (x < at && x > xl) xl = x;
    if (x > at && x < xr) xr = x;
  }
  Columns col; col.xl = xl; col.xr = xr;
  for (Index c = 0; c < mesh.nCells(); ++c) {
    if (std::abs(cc(c,0) - xl) < 1e-9) col.left.push_back(c);
    if (std::abs(cc(c,0) - xr) < 1e-9) col.right.push_back(c);
  }
  auto byY = [&](Index a, Index b) { return cc(a,1) < cc(b,1); };
  std::sort(col.left.begin(), col.left.end(), byY);
  std::sort(col.right.begin(), col.right.end(), byY);
  return col;
}

// A station's profile: for every x-facing face whose two cells' centres
// straddle x = at, the two cells' y and u interpolated linearly in x to it,
// sorted by y. Behind the step TMR's grid lines lean by about 1e-5 across the
// channel, so a column cannot be found by its centres' x (ADR-045's results).
struct Profile { std::vector<Real> y, u; };
template <class U>
Profile stationProfile(const PolyMesh& mesh, const U& u, Real at) {
  auto cc = host(mesh.cellCentre()); auto fa = host(mesh.faceArea());
  auto own = host(mesh.owner()); auto nei = host(mesh.neighbour());
  std::vector<std::pair<Real, Real>> rows;
  for (Index f = 0; f < mesh.nInternalFaces(); ++f) {
    const Real mag = std::sqrt(fa(f,0)*fa(f,0) + fa(f,1)*fa(f,1) + fa(f,2)*fa(f,2));
    if (std::abs(fa(f,0)) < 0.9 * mag) continue;
    Index a = own(f), b = nei(f);
    if (b >= mesh.nCells()) continue;
    if (cc(a,0) > cc(b,0)) std::swap(a, b);
    if (!(cc(a,0) < at && cc(b,0) >= at)) continue;
    const Real t = (at - cc(a,0)) / (cc(b,0) - cc(a,0));
    rows.emplace_back((1.0 - t) * cc(a,1) + t * cc(b,1), (1.0 - t) * u(a,0) + t * u(b,0));
  }
  std::sort(rows.begin(), rows.end());
  Profile pr;
  for (const auto& r : rows) { pr.y.push_back(r.first); pr.u.push_back(r.second); }
  return pr;
}

// A march's checkpoint (ADR-045's fifth revision): the fields, the face
// flux and the march's bookkeeping, every 500 steps, so that a march stopped
// by the machine resumes where it was. Binary, written beside and renamed.
struct Checkpoint {
  long nc = 0, nif = 0;
  int step = 0, sinceChange = 0, holdUntil = 0, refused = 0;
  double time = 0, dtNow = 0, ceiling = 0, seconds = 0;
  std::vector<double> u, p, k, w, F;
  std::vector<int> histStep;
  std::vector<double> histXr, histCf4;
};

template <class T> void putv(std::FILE* f, const std::vector<T>& v) {
  const long n = static_cast<long>(v.size());
  std::fwrite(&n, sizeof n, 1, f);
  if (n) std::fwrite(v.data(), sizeof(T), v.size(), f);
}
template <class T> bool getv(std::FILE* f, std::vector<T>& v) {
  long n = 0;
  if (std::fread(&n, sizeof n, 1, f) != 1 || n < 0) return false;
  v.resize(static_cast<std::size_t>(n));
  return n == 0 || std::fread(v.data(), sizeof(T), v.size(), f) == v.size();
}

void writeCheckpoint(const std::string& path, const Checkpoint& c) {
  const std::string tmp = path + ".tmp";
  std::FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) return;
  std::fwrite("VFBSCK01", 1, 8, f);
  std::fwrite(&c.nc, sizeof c.nc, 1, f); std::fwrite(&c.nif, sizeof c.nif, 1, f);
  const int ints[4] = {c.step, c.sinceChange, c.holdUntil, c.refused};
  const double dbl[4] = {c.time, c.dtNow, c.ceiling, c.seconds};
  std::fwrite(ints, sizeof ints, 1, f); std::fwrite(dbl, sizeof dbl, 1, f);
  putv(f, c.u); putv(f, c.p); putv(f, c.k); putv(f, c.w); putv(f, c.F);
  putv(f, c.histStep); putv(f, c.histXr); putv(f, c.histCf4);
  const bool ok = std::fflush(f) == 0;
  std::fclose(f);
  if (ok) std::rename(tmp.c_str(), path.c_str());
}

bool readCheckpoint(const std::string& path, Checkpoint& c) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[8];
  bool ok = std::fread(magic, 1, 8, f) == 8 && std::string(magic, 8) == "VFBSCK01";
  int ints[4]; double dbl[4];
  ok = ok && std::fread(&c.nc, sizeof c.nc, 1, f) == 1 && std::fread(&c.nif, sizeof c.nif, 1, f) == 1 &&
       std::fread(ints, sizeof ints, 1, f) == 1 && std::fread(dbl, sizeof dbl, 1, f) == 1;
  ok = ok && getv(f, c.u) && getv(f, c.p) && getv(f, c.k) && getv(f, c.w) && getv(f, c.F) &&
       getv(f, c.histStep) && getv(f, c.histXr) && getv(f, c.histCf4);
  std::fclose(f);
  if (!ok) return false;
  c.step = ints[0]; c.sinceChange = ints[1]; c.holdUntil = ints[2]; c.refused = ints[3];
  c.time = dbl[0]; c.dtNow = dbl[1]; c.ceiling = dbl[2]; c.seconds = dbl[3];
  return true;
}

// ADR-047: forwards every call to a solver and counts its solves, iterations
// and the solves that report not converged, which the solver code discards.
// A step's non-orthogonal sweep count is the increase of the pressure
// solver's count over the step.
class Counted final : public LinearSolver {
 public:
  explicit Counted(LinearSolver& s) : s_(s) {}
  SolveReport solve(LinearSystem& sys, ScalarField& x, Real relTol, Real absTol,
                    int maxIter) override {
    const SolveReport r = s_.solve(sys, x, relTol, absTol, maxIter);
    ++solves; iterations += r.iterations;
    if (!r.converged) ++unconverged;
    return r;
  }
  void notifyMatrixChanged() override { s_.notifyMatrixChanged(); }
  std::string backendName() const override { return s_.backendName(); }
  long long solves = 0, iterations = 0, unconverged = 0;

 private:
  LinearSolver& s_;
};

// ADR-047: the checkpoint files beside <out>, by kind: the serial <out>.ckpt,
// and per-rank <out>.ckpt.<rank>of<P> with the P of each.
struct CkptFiles { bool serial = false; std::vector<int> perRankP; };

CkptFiles ckptFiles(const std::string& out) {
  namespace fs = std::filesystem;
  CkptFiles cf;
  const fs::path o(out);
  const fs::path dir = o.has_parent_path() ? o.parent_path() : fs::path(".");
  const std::string base = o.filename().string() + ".ckpt";
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    const std::string n = e.path().filename().string();
    if (n == base) { cf.serial = true; continue; }
    if (n.size() >= 4 && n.compare(n.size() - 4, 4, ".tmp") == 0) continue;   // never renamed
    if (n.rfind(base + ".", 0) != 0) continue;
    const std::string rest = n.substr(base.size() + 1);       // <rank>of<P>
    const auto of = rest.find("of");
    if (of == std::string::npos) continue;
    cf.perRankP.push_back(std::atoi(rest.c_str() + of + 2));
  }
  return cf;
}

// ADR-047: the first index of the largest |a[i]|, as a cell (i / stride).
Index scaleCell(const std::vector<Real>& a, int stride) {
  Real s = -1.0;
  std::size_t at = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::abs(a[i]) > s) { s = std::abs(a[i]); at = i; }
  return static_cast<Index>(at / stride);
}

}  // namespace

// ---------------------------------------------------------------------------
// ADR-047: the harness on more than one process. Its own helpers; the serial
// ones above are left as they were. Every decision below is taken from data
// that is the same on every rank: reduced with comm.max/min, or gathered and
// sorted. Rank 0 prints and writes.
namespace {

bool g_mpiDone = false;     // the MPI path returned normally

// Every error exit under MPI goes through MPI_Abort: a std::exit path in a
// serial helper reaches it through this.
void abortUnlessDone() {
  std::fflush(nullptr);                   // the message that led here, and what rank 0 buffered
#ifdef VIBEFLOW_HAVE_MPI
  if (!g_mpiDone) MPI_Abort(MPI_COMM_WORLD, 2);
#endif
}

[[noreturn]] void mpiAbort(int code) {
  std::fflush(stdout);
#ifdef VIBEFLOW_HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, code);
#endif
  std::exit(code);
}

// An exit every rank reaches together: what rank 0 printed is out before any
// rank aborts the job.
[[noreturn]] void collectiveExit(const Comm& comm, int code) {
  std::fflush(stdout);
  comm.barrier();
  mpiAbort(code);
}

// Every rank's vector, concatenated in rank order, on every rank.
std::vector<double> allGather(const Comm& comm, const std::vector<double>& mine) {
  const std::vector<std::vector<double>> send(static_cast<std::size_t>(comm.size()), mine);
  const auto got = comm.alltoallv(send);
  std::vector<double> all;
  for (const auto& g : got) all.insert(all.end(), g.begin(), g.end());
  return all;
}

// Every rank's vector, concatenated in rank order, on rank 0 (empty elsewhere).
std::vector<double> gatherTo0(const Comm& comm, const std::vector<double>& mine) {
  std::vector<std::vector<double>> send(static_cast<std::size_t>(comm.size()));
  send[0] = mine;
  const auto got = comm.alltoallv(send);
  std::vector<double> all;
  if (comm.rank() == 0)
    for (const auto& g : got) all.insert(all.end(), g.begin(), g.end());
  return all;
}

// A maximum and where it is. Over the ranks the largest value wins and a tie
// goes to the smallest global id: the serial first-index rule, since the
// owned cells are in increasing global id.
struct Loc { Real v = 0.0; long long gid = -1; Real x = 0.0, y = 0.0; };

Loc maxLoc(const Comm& comm, const Loc& mine) {
  const auto all = allGather(comm, {mine.v, static_cast<double>(mine.gid), mine.x, mine.y});
  Loc best;
  for (std::size_t i = 0; i + 3 < all.size(); i += 4) {
    const Loc l{all[i], static_cast<long long>(all[i + 1]), all[i + 2], all[i + 3]};
    if (l.gid < 0) continue;                       // a rank with no candidate
    if (best.gid < 0 || l.v > best.v || (l.v == best.v && l.gid < best.gid)) best = l;
  }
  return best;
}

std::uint64_t fnv1a(const std::vector<Index>& ids) {
  std::uint64_t h = 14695981039346656037ULL;
  for (Index id : ids) {
    const long long v = id;
    const auto* b = reinterpret_cast<const unsigned char*>(&v);
    for (std::size_t k = 0; k < sizeof v; ++k) { h ^= b[k]; h *= 1099511628211ULL; }
  }
  return h;
}

// Each rank's checkpoint, VFBSCK02: P, the rank, its counts, a hash of its
// owned global ids, its own u, p, k, omega, its own copy of F on every local
// face (interface faces included) and the march's bookkeeping as VFBSCK01
// holds it. A restore puts back exactly what the rank had.
struct RankCheckpoint {
  int P = 0, rank = 0;
  long nc = 0, nif = 0;
  std::uint64_t hash = 0;
  int step = 0, sinceChange = 0, holdUntil = 0, refused = 0;
  double time = 0, dtNow = 0, ceiling = 0, seconds = 0;
  std::vector<double> u, p, k, w, F;
  std::vector<int> histStep;
  std::vector<double> histXr, histCf4;
};

// The first of two phases: <path>.tmp, true if all of it reached the file.
// The caller renames on every rank only if every rank wrote its own.
bool writeRankTmp(const std::string& path, const RankCheckpoint& c) {
  const std::string tmp = path + ".tmp";
  std::FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) return false;
  std::fwrite("VFBSCK02", 1, 8, f);
  const int pr[2] = {c.P, c.rank};
  std::fwrite(pr, sizeof pr, 1, f);
  std::fwrite(&c.nc, sizeof c.nc, 1, f); std::fwrite(&c.nif, sizeof c.nif, 1, f);
  std::fwrite(&c.hash, sizeof c.hash, 1, f);
  const int ints[4] = {c.step, c.sinceChange, c.holdUntil, c.refused};
  const double dbl[4] = {c.time, c.dtNow, c.ceiling, c.seconds};
  std::fwrite(ints, sizeof ints, 1, f); std::fwrite(dbl, sizeof dbl, 1, f);
  putv(f, c.u); putv(f, c.p); putv(f, c.k); putv(f, c.w); putv(f, c.F);
  putv(f, c.histStep); putv(f, c.histXr); putv(f, c.histCf4);
  const bool ok = std::fflush(f) == 0 && std::ferror(f) == 0;
  return std::fclose(f) == 0 && ok;
}

// What every rank's checkpoint must hold alike for a resume: the march's
// bookkeeping and a hash of its histories, the hash's halves exact as doubles.
std::vector<double> bookkeeping(const RankCheckpoint& c) {
  std::uint64_t h = 14695981039346656037ULL;
  auto mix = [&](const void* p, std::size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t k = 0; k < n; ++k) { h ^= b[k]; h *= 1099511628211ULL; }
  };
  mix(c.histStep.data(), c.histStep.size() * sizeof(int));
  mix(c.histXr.data(), c.histXr.size() * sizeof(double));
  mix(c.histCf4.data(), c.histCf4.size() * sizeof(double));
  return {static_cast<double>(c.step), c.time, c.dtNow, c.ceiling, static_cast<double>(c.sinceChange),
          static_cast<double>(c.holdUntil), static_cast<double>(c.refused),
          static_cast<double>(c.histStep.size()), static_cast<double>(h >> 32),
          static_cast<double>(h & 0xffffffffULL)};
}

bool readRankCheckpoint(const std::string& path, RankCheckpoint& c) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[8];
  int pr[2], ints[4];
  double dbl[4];
  bool ok = std::fread(magic, 1, 8, f) == 8 && std::string(magic, 8) == "VFBSCK02";
  ok = ok && std::fread(pr, sizeof pr, 1, f) == 1 && std::fread(&c.nc, sizeof c.nc, 1, f) == 1 &&
       std::fread(&c.nif, sizeof c.nif, 1, f) == 1 && std::fread(&c.hash, sizeof c.hash, 1, f) == 1 &&
       std::fread(ints, sizeof ints, 1, f) == 1 && std::fread(dbl, sizeof dbl, 1, f) == 1;
  ok = ok && getv(f, c.u) && getv(f, c.p) && getv(f, c.k) && getv(f, c.w) && getv(f, c.F) &&
       getv(f, c.histStep) && getv(f, c.histXr) && getv(f, c.histCf4);
  std::fclose(f);
  if (!ok) return false;
  c.P = pr[0]; c.rank = pr[1];
  c.step = ints[0]; c.sinceChange = ints[1]; c.holdUntil = ints[2]; c.refused = ints[3];
  c.time = dbl[0]; c.dtNow = dbl[1]; c.ceiling = dbl[2]; c.seconds = dbl[3];
  return true;
}

}  // namespace

int runMpi(int argc, char** argv) {
  const Comm comm = Comm::world();
  const int P = comm.size(), rank = comm.rank();
  const bool r0 = rank == 0;
  std::atexit(abortUnlessDone);
  int rc = 0;
  try {
    if (argc < 3) {
      if (r0) std::printf("usage: backstep <mesh.hex> <out prefix> [dt] [max steps at the full dt] [U_in]\n");
      collectiveExit(comm, 2);
    }
    if (std::getenv("VIBEFLOW_BS_DEBUG")) {
      if (r0) std::printf("VIBEFLOW_BS_DEBUG is refused on more than one rank\n");
      collectiveExit(comm, 2);
    }
#ifdef VIBEFLOW_HAVE_PETSC
    const std::string pcfg = std::getenv("VIBEFLOW_PRESSURE") ? std::getenv("VIBEFLOW_PRESSURE") : "cg+hypre";
    const std::string mcfg = std::getenv("VIBEFLOW_MOMENTUM") ? std::getenv("VIBEFLOW_MOMENTUM") : "bicgstab+bjilu";
#else
    const std::string pcfg = std::getenv("VIBEFLOW_PRESSURE") ? std::getenv("VIBEFLOW_PRESSURE") : "native";
    const std::string mcfg = std::getenv("VIBEFLOW_MOMENTUM") ? std::getenv("VIBEFLOW_MOMENTUM") : "native";
#endif
    for (const std::string& cfg : {pcfg, mcfg}) {
      const auto plus = cfg.find('+');
      const std::string pc = plus == std::string::npos ? "" : cfg.substr(plus + 1);
      if (pc == "ilu" || pc == "gamg") {    // sequential factorisations
        if (r0) std::printf("%s is refused on more than one rank\n", cfg.c_str());
        collectiveExit(comm, 2);
      }
    }
    const std::string meshPath = argv[1], out = argv[2];
    const Real dt = argc > 3 ? std::atof(argv[3]) : 0.08;
    const int maxFull = argc > 4 ? std::atoi(argv[4]) : 5000;
    const Real uIn = argc > 5 ? std::atof(argv[5]) : 1.0;
    const auto t0 = std::chrono::steady_clock::now();

    Real X_IN = 1e300, X_OUT = -1e300, Y_TOP = -1e300, zmin = 1e300, zmax = -1e300;
    std::unique_ptr<DistributedMesh> meshPtr;
    {
      const RawMesh raw = RawMesh::fromHexFile(meshPath);
      for (const auto& q : raw.points) {
        X_IN = std::min(X_IN, q.x); X_OUT = std::max(X_OUT, q.x); Y_TOP = std::max(Y_TOP, q.y);
        zmin = std::min(zmin, q.z); zmax = std::max(zmax, q.z);
      }
      meshPtr = std::make_unique<DistributedMesh>(raw, comm, PartitionMethod::RCB);
    }
    const DistributedMesh& mesh = *meshPtr;
    const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();
    const Index nif = mesh.nInternalFaces();
    const std::vector<Index>& gid = mesh.globalCellId();
    auto bcen = host(mesh.boundaryCentre()); auto ba = host(mesh.boundaryArea());
    View1<int> uType("uType", nb), pType("pType", nb), kind("kind", nb), wallMask("wall", nb);
    VectorField ub("ub", nb, 3);
    ScalarField fb("fb", nb), pval("pval", nb), kB("kB", nb), wB("wB", nb);
    auto hUT = Kokkos::create_mirror_view(uType); auto hPT = Kokkos::create_mirror_view(pType);
    auto hK = Kokkos::create_mirror_view(kind); auto hW = Kokkos::create_mirror_view(wallMask);
    auto hUB = Kokkos::create_mirror_view(ub); auto hFB = Kokkos::create_mirror_view(fb);
    auto hKB = Kokkos::create_mirror_view(kB); auto hWB = Kokkos::create_mirror_view(wB);
    std::vector<Index> bottomFace;            // this rank's bottom-wall faces
    int count[5] = {0, 0, 0, 0, 0};
    int unclassified = 0;
    for (Index f = 0; f < nb; ++f) {
      const Real x = bcen(f,0), y = bcen(f,1);
      const Real mag = std::sqrt(ba(f,0)*ba(f,0) + ba(f,1)*ba(f,1) + ba(f,2)*ba(f,2));
      Patch p = SPAN;
      bool isBottom = false;
      if (std::abs(ba(f,2)) > 0.9 * mag) p = SPAN;
      else if (std::abs(x - X_IN) < 1e-9) p = INLET;
      else if (std::abs(x - X_OUT) < 1e-9) p = OUTLET;
      else if (std::abs(y - Y_TOP) < 1e-9) p = x < X_SLIP ? SLIP : WALL;
      else if (std::abs(y - 1.0) < 1e-9 && x < 0.0) { p = x < X_SLIP ? SLIP : WALL; isBottom = true; }
      else if (std::abs(y) < 1e-12 && x > 0.0) { p = WALL; isBottom = true; }
      else if (std::abs(x) < 1e-12 && y < 1.0) p = WALL;
      else { std::printf("unclassified boundary face at (%.6f, %.6f)\n", x, y); ++unclassified; continue; }
      ++count[p];
      hUB(f,0) = hUB(f,1) = hUB(f,2) = 0.0; hFB(f) = 0.0;
      hKB(f) = 0.0; hWB(f) = 0.0; hW(f) = 0;
      hPT(f) = static_cast<int>(PressureBC::FixedFlux);
      hK(f) = static_cast<int>(TurbulenceBC::ZeroGradient);
      switch (p) {
        case INLET:
          hUT(f) = static_cast<int>(VelocityBC::Dirichlet);
          hUB(f,0) = uIn; hFB(f) = uIn * ba(f,0);
          hK(f) = static_cast<int>(TurbulenceBC::Dirichlet); hKB(f) = K_IN; hWB(f) = W_IN;
          break;
        case OUTLET:
          hUT(f) = static_cast<int>(VelocityBC::ZeroGradient);
          hPT(f) = static_cast<int>(PressureBC::FixedValue);
          break;
        case SLIP:
          hUT(f) = static_cast<int>(VelocityBC::Slip);
          break;
        case WALL:
          hUT(f) = static_cast<int>(VelocityBC::Dirichlet);
          hK(f) = static_cast<int>(TurbulenceBC::Wall); hW(f) = 1;
          if (isBottom) bottomFace.push_back(f);
          break;
        case SPAN:
          hUT(f) = static_cast<int>(VelocityBC::ZeroGradient);
          break;
      }
    }
    if (comm.max(static_cast<Real>(unclassified)) > 0.0) collectiveExit(comm, 2);
    // The bottom wall over every rank, sorted by x as serial sorts it. Face x
    // is the same bit for bit at any rank count. `order` maps the gathered
    // concatenation (rank order, then this rank's order) to sorted position.
    std::vector<Real> bx;                     // sorted x
    std::vector<std::size_t> order;
    {
      std::vector<double> mine;
      for (Index f : bottomFace) mine.push_back(bcen(f,0));
      const auto all = allGather(comm, mine);
      order.resize(all.size());
      for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
      std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return all[a] < all[b]; });
      for (std::size_t i : order) bx.push_back(all[i]);
    }
    // Gathers one value per bottom face (computed on its rank) into x order.
    auto gatherBottom = [&](const std::vector<Real>& mine) {
      const auto all = allGather(comm, mine);
      std::vector<Real> v(order.size());
      for (std::size_t i = 0; i < order.size(); ++i) v[i] = all[order[i]];
      return v;
    };
    Real volume = 0.0;
    {
      auto vol = host(mesh.cellVolume());
      for (Index c = 0; c < nc; ++c) volume += vol(c);
    }
    volume = comm.sum(volume);
    const Index nGlobal = comm.sum(nc);
    int gcount[5];
    for (int i = 0; i < 5; ++i) gcount[i] = static_cast<int>(comm.sum(static_cast<Index>(count[i])));
    if (r0) {
      std::printf("backward-facing step, SST-1994, %s: %d cells, volume %.15g (area %.15g), "
                  "dt %g, U_in %.6f\n", meshPath.c_str(), static_cast<int>(nGlobal), volume,
                  volume / (zmax - zmin), dt, uIn);
      std::printf("  patches: inlet %d  outlet %d  slip %d  wall %d  span %d  (bottom wall %zu)\n",
                  gcount[INLET], gcount[OUTLET], gcount[SLIP], gcount[WALL], gcount[SPAN], bx.size());
      std::printf("  ranks %d, RCB: %d owned cells on rank 0\n", P, static_cast<int>(nc));
    }
    const std::size_t nUp = static_cast<std::size_t>(
        std::count_if(bx.begin(), bx.end(), [](Real x) { return x < 0.0; }));
    if (nUp < 4) { if (r0) std::printf("fewer than four upstream wall faces\n"); collectiveExit(comm, 2); }
    const std::size_t lip0 = nUp - 4;
    if (r0) std::printf("LIPX %.10e %.10e %.10e %.10e\n", bx[lip0], bx[lip0 + 1], bx[lip0 + 2], bx[lip0 + 3]);
    Kokkos::deep_copy(uType, hUT); Kokkos::deep_copy(pType, hPT); Kokkos::deep_copy(kind, hK);
    Kokkos::deep_copy(wallMask, hW); Kokkos::deep_copy(ub, hUB); Kokkos::deep_copy(fb, hFB);
    Kokkos::deep_copy(kB, hKB); Kokkos::deep_copy(wB, hWB);

    PisoControls ctl;
    ctl.outer = 1;
    ctl.correctors = 8;
    if (const char* e = std::getenv("VIBEFLOW_CORRECTORS")) ctl.correctors = std::atoi(e);
    ctl.convection = ConvectionScheme::LinearUpwind;
    if (const char* e = std::getenv("VIBEFLOW_CONVECTION")) {
      const std::string c(e);
      if (c == "linear") ctl.convection = ConvectionScheme::Linear;
      else if (c == "upwind") ctl.deferredCorrection = false;
    }
    if (std::getenv("VIBEFLOW_NO_NONORTH")) ctl.diffusionNonOrth = false;
    if (const char* e = std::getenv("VIBEFLOW_BS_SWEEPS")) {
      ctl.nonOrthCorrectors = std::atoi(e); ctl.nonOrthTol = 0.0;
    }
    const char* initPath = std::getenv("VIBEFLOW_BS_INIT");
    Real dtNow = initPath ? dt / 8.0 : dt / 64.0;
    if (const char* e = std::getenv("VIBEFLOW_BS_RAMP")) dtNow = dt / std::max(1.0, std::atof(e));

    // A resume goes ahead only if every rank finds its own file with this P,
    // this partition and the same step; otherwise, with any checkpoint of any
    // kind present, it is refused, before any file is written or removed.
    const std::string ckName = out + ".ckpt." + std::to_string(rank) + "of" + std::to_string(P);
    const std::uint64_t ownHash = fnv1a(gid);
    RankCheckpoint ck;
    bool resume = false;
    if (std::getenv("VIBEFLOW_BS_RESUME")) {
      const CkptFiles cf = ckptFiles(out);
      bool otherP = false;
      for (int q : cf.perRankP) otherP |= q != P;
      const bool mineRead = readRankCheckpoint(ckName, ck);
      const bool mineOk = mineRead && ck.P == P && ck.rank == rank && ck.nc == static_cast<long>(nc) &&
                          ck.nif == static_cast<long>(nif) && ck.hash == ownHash;
      // Every reduction below runs on every rank: none may be skipped.
      const bool anyFile = comm.max((cf.serial || !cf.perRankP.empty()) ? 1.0 : 0.0) > 0.5;
      const bool allOk = comm.min(mineOk ? 1.0 : 0.0) > 0.5;
      const bool noForeign = comm.max((cf.serial || otherP) ? 1.0 : 0.0) < 0.5;
      // The step, the time, dt, the ramp's state and the histories: alike on
      // every rank, bit for bit, or a rename was cut short between ranks.
      const std::vector<double> mine = bookkeeping(ck);
      const std::vector<double> all = allGather(comm, mine);
      bool booksAgree = all.size() == mine.size() * static_cast<std::size_t>(P);
      for (std::size_t i = mine.size(); booksAgree && i < all.size(); ++i)
        booksAgree = std::memcmp(&all[i], &all[i % mine.size()], sizeof(double)) == 0;
      if (anyFile) {
        if (allOk && noForeign && booksAgree) {
          resume = true;
        } else {
          if (r0) std::printf("REFUSED: the checkpoints beside %s are not this run's %d ranks' own "
                              "(serial %s, another P %s, own files %s, bookkeeping %s)\n", out.c_str(), P,
                              cf.serial ? "present" : "absent", otherP ? "present" : "absent",
                              allOk ? "matching" : "missing or mismatched", booksAgree ? "alike" : "differs");
          collectiveExit(comm, 3);
        }
      }
    }
    if (resume) dtNow = ck.dtNow;
    PisoSolver solver(mesh, NU, dtNow, ctl, comm);
    solver.setBoundaryTypes(uType);
    solver.setPressureBoundary(pType, pval);
    if (r0) std::printf("  PISO correctors %d, first dt %.4g\n", ctl.correctors, dtNow);
    if (resume && r0)
      std::printf("  resumed from %s.ckpt.*of%d: step %d, t %.3f, dt %.4g, %d steps at it\n", out.c_str(), P,
                  ck.step, ck.time, ck.dtNow, ck.sinceChange);

    CoarseState coarse;
    auto ccH = host(mesh.cellCentre());
    if (initPath && !resume) {
      coarse = readState(initPath);
      if (r0) std::printf("  initial state: %s (%zu coarse cells)\n", initPath, coarse.v.size());
    }
    const auto* halo = mesh.halo();
    {
      // The owned cells, then their halo: a ghost takes its owner's values.
      VectorField u0("u0", nt, 3);
      ScalarField p0("p0", nt), F0("F0", nif);
      auto hu = host(u0); auto hp = host(p0);
      for (Index c = 0; c < nc; ++c) {
        if (resume) {
          hu(c,0) = ck.u[c*3]; hu(c,1) = ck.u[c*3 + 1]; hu(c,2) = ck.u[c*3 + 2]; hp(c) = ck.p[c];
        } else if (initPath) {
          const auto& q = coarse.v[nearestCoarse(coarse, ccH(c,0), ccH(c,1))];
          hu(c,0) = q[2]; hu(c,1) = q[3]; hu(c,2) = q[4]; hp(c) = q[5];
        } else {
          hu(c,0) = uIn; hu(c,1) = 0.0; hu(c,2) = 0.0; hp(c) = 0.0;
        }
      }
      Kokkos::deep_copy(u0, hu); Kokkos::deep_copy(p0, hp);
      if (halo) { halo->exchange(u0); halo->exchange(p0); }
      if (resume) {
        auto hF = host(F0);
        for (Index f = 0; f < nif; ++f) hF(f) = ck.F[f];
        Kokkos::deep_copy(F0, hF);
      } else {
        auto fa = mesh.faceArea(); auto fc = mesh.faceCentre(); auto ccd = mesh.cellCentre();
        auto own = mesh.owner(); auto nei = mesh.neighbour();
        Kokkos::parallel_for("F0", Kokkos::RangePolicy<ExecSpace>(0, nif),
          KOKKOS_LAMBDA(const Index f) {
            Real lo = 0.0, ln = 0.0;
            for (int i = 0; i < 3; ++i) {
              const Real ro = fc(f, i) - ccd(own(f), i), rn = fc(f, i) - ccd(nei(f), i);
              lo += ro * ro; ln += rn * rn;
            }
            const Real w = Kokkos::sqrt(ln) / (Kokkos::sqrt(lo) + Kokkos::sqrt(ln));
            Real sum = 0.0;
            for (int i = 0; i < 3; ++i) sum += (w * u0(own(f), i) + (1.0 - w) * u0(nei(f), i)) * fa(f, i);
            F0(f) = sum;
          });
        Kokkos::fence();
      }
      solver.setState(u0, p0, F0);
    }
    TurbulenceModel tm;
    tm.variant = SstVariant::Menter1994;
    if (std::getenv("VIBEFLOW_BS_FROZEN")) tm.frozen = true;
    if (std::getenv("VIBEFLOW_BS_NO_TRANSPOSE")) tm.transposeStress = false;
    solver.enableTurbulence(tm, wallMask);
    solver.setTurbulenceBoundary(kind, kB, wB);
    {
      ScalarField k0("k0", nt), w0("w0", nt);
      Kokkos::deep_copy(k0, K_IN); Kokkos::deep_copy(w0, W_IN);
      if (resume || initPath) {
        auto hk = host(k0); auto hw = host(w0);
        for (Index c = 0; c < nc; ++c) {
          if (resume) { hk(c) = ck.k[c]; hw(c) = ck.w[c]; }
          else {
            const auto& q = coarse.v[nearestCoarse(coarse, ccH(c,0), ccH(c,1))];
            hk(c) = q[6]; hw(c) = q[7];
          }
        }
        Kokkos::deep_copy(k0, hk); Kokkos::deep_copy(w0, hw);
      }
      solver.setTurbulence(k0, w0);         // exchanges the halo
    }
    coarse = CoarseState();

    std::unique_ptr<LinearSolver> momentumPtr, pressurePtr;
    auto makeSolver = [&](const std::string& cfg, bool symmetric) -> std::unique_ptr<LinearSolver> {
      if (cfg == "native") {
        if (symmetric) return std::make_unique<NativeCG>(mesh, comm);
        return std::make_unique<NativeBiCGStab>(mesh, comm);
      }
#ifdef VIBEFLOW_HAVE_PETSC
      return std::make_unique<PetscSolver>(mesh, comm, cfg);
#else
      if (r0) std::printf("built without PETSc: %s unavailable\n", cfg.c_str());
      collectiveExit(comm, 2);
#endif
    };
    momentumPtr = makeSolver(mcfg, false);
    pressurePtr = makeSolver(pcfg, true);
    if (r0) std::printf("  momentum, k and omega solver: %s; pressure: %s\n", mcfg.c_str(), pcfg.c_str());
    Counted momentumC(*momentumPtr), pressureC(*pressurePtr);

    // The channel's centre velocity at x = -4: the two columns around it, the
    // row nearest y = 5 in each (a tie to the lower y), wherever they are owned.
    Real xl = -1e300, xr = 1e300;
    for (Index c = 0; c < nc; ++c) {
      const Real x = ccH(c, 0);
      if (x < X_CF4 && x > xl) xl = x;
      if (x > X_CF4 && x < xr) xr = x;
    }
    xl = comm.max(xl); xr = comm.min(xr);
    struct Near { Real d = 1e300, y = 1e300; Index c = -1; };
    Near nl, nr;
    for (Index c = 0; c < nc; ++c) {
      const Real d = std::abs(ccH(c,1) - 5.0), y = ccH(c,1);
      auto better = [&](const Near& n) { return d < n.d || (d == n.d && y < n.y); };
      if (std::abs(ccH(c,0) - xl) < 1e-9 && better(nl)) nl = {d, y, c};
      if (std::abs(ccH(c,0) - xr) < 1e-9 && better(nr)) nr = {d, y, c};
    }
    // Which rank owns each column's chosen cell: gathered (d, y, rank).
    auto chooseOwner = [&](const Near& n) {
      const auto all = allGather(comm, {n.c >= 0 ? n.d : 1e300, n.y, static_cast<double>(rank)});
      Real bd = 1e300, by = 1e300; int br = -1;
      for (std::size_t i = 0; i + 2 < all.size(); i += 3)
        if (all[i] < bd || (all[i] == bd && all[i + 1] < by)) { bd = all[i]; by = all[i + 1]; br = static_cast<int>(all[i + 2]); }
      return br;
    };
    const int ownerL = chooseOwner(nl), ownerR = chooseOwner(nr);
    auto centreU = [&]() {
      auto u = host(solver.velocity());
      const auto all = allGather(comm, {rank == ownerL ? u(nl.c, 0) : 0.0, rank == ownerR ? u(nr.c, 0) : 0.0});
      const Real a = all[2 * ownerL], b = all[2 * ownerR + 1];
      const Real t = (X_CF4 - xl) / (xr - xl);
      return (1.0 - t) * a + t * b;
    };

    std::vector<Real> uPrev, kPrev, wPrev, uN, kN, wN;
    auto snapshot = [&](std::vector<Real>& uu, std::vector<Real>& kk, std::vector<Real>& ww) {
      auto u = host(solver.velocity()); auto k = host(solver.turbulentKineticEnergy());
      auto w = host(solver.specificDissipation());
      uu.resize(nc * 3); kk.resize(nc); ww.resize(nc);
      for (Index c = 0; c < nc; ++c) {
        for (int d = 0; d < 3; ++d) uu[c*3 + d] = u(c, d);
        kk[c] = k(c); ww[c] = w(c);
      }
    };
    // relChange over the ranks: the largest difference and the largest
    // magnitude each reduced, their cells by the max-location rule.
    struct ChangeD { Real rel = 0.0; Loc diff, scale; };
    auto relChange = [&](const std::vector<Real>& a, const std::vector<Real>& b, int stride) {
      Real d = 0.0, s = 1e-300;
      Index cd = 0, cs = 0;
      bool anyS = false;
      for (std::size_t i = 0; i < a.size(); ++i) {
        const Real di = std::abs(a[i] - b[i]);
        if (di > d) { d = di; cd = static_cast<Index>(i / stride); }
        if (!anyS || std::abs(a[i]) > s) { s = std::max(s, std::abs(a[i])); cs = static_cast<Index>(i / stride); anyS = true; }
      }
      ChangeD r;
      r.diff = maxLoc(comm, {d, nc > 0 ? static_cast<long long>(gid[cd]) : -1, nc > 0 ? ccH(cd,0) : 0.0, nc > 0 ? ccH(cd,1) : 0.0});
      r.scale = maxLoc(comm, {s, nc > 0 ? static_cast<long long>(gid[cs]) : -1, 0.0, 0.0});
      r.rel = r.diff.v / std::max(r.scale.v, 1e-300);
      return r;
    };
    snapshot(uPrev, kPrev, wPrev);

    // The state in the file's cell order, written by rank 0.
    auto saveState = [&](const std::string& path) {
      auto u = host(solver.velocity()); auto p = host(solver.pressure());
      auto k = host(solver.turbulentKineticEnergy()); auto w = host(solver.specificDissipation());
      std::vector<double> mine;
      mine.reserve(static_cast<std::size_t>(nc) * 9);
      for (Index c = 0; c < nc; ++c)
        for (Real v : {static_cast<Real>(gid[c]), ccH(c,0), ccH(c,1), u(c,0), u(c,1), u(c,2), p(c), k(c), w(c)})
          mine.push_back(v);
      const auto all = gatherTo0(comm, mine);
      if (!r0) return;
      std::vector<std::size_t> at(static_cast<std::size_t>(nGlobal));
      for (std::size_t i = 0; i + 8 < all.size(); i += 9) at[static_cast<std::size_t>(all[i])] = i;
      std::FILE* fs = std::fopen(path.c_str(), "w");
      std::fprintf(fs, "%ld\n", static_cast<long>(nGlobal));
      for (std::size_t g = 0; g < at.size(); ++g) {
        const double* r = &all[at[g]];
        std::fprintf(fs, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", r[1], r[2], r[3], r[4], r[5],
                     r[6], r[7], r[8]);
      }
      std::fclose(fs);
    };
    std::vector<int> saveAt;
    if (const char* e = std::getenv("VIBEFLOW_BS_SAVE_AT")) {
      std::string list(e);
      for (std::size_t a = 0; a < list.size();) {
        const std::size_t b = std::min(list.find(',', a), list.size());
        saveAt.push_back(std::atoi(list.substr(a, b - a).c_str()));
        a = b + 1;
      }
    }
    const int stopAt = std::getenv("VIBEFLOW_BS_STOP") ? std::atoi(std::getenv("VIBEFLOW_BS_STOP")) : -1;

    Real C_UP = 3.0, C_DOWN = 6.0;
    if (const char* e = std::getenv("VIBEFLOW_BS_CUP")) C_UP = std::atof(e);
    if (const char* e = std::getenv("VIBEFLOW_BS_CDOWN")) C_DOWN = std::atof(e);
    const int MAX_STEPS = 20000;
    auto ownH = host(mesh.owner()); auto neiH = host(mesh.neighbour());
    auto bclH = host(mesh.boundaryCell()); auto volH = host(mesh.cellVolume());
    std::vector<Real> fsum(nt);
    // The largest cell Courant number over the ranks, and where it is.
    auto courant = [&](Real dtTest) {
      auto F = host(solver.faceFlux()); auto Fb = host(solver.boundaryFlux());
      std::fill(fsum.begin(), fsum.end(), 0.0);
      for (Index f = 0; f < nif; ++f) {
        const Real a = std::abs(F(f));
        fsum[ownH(f)] += a; fsum[neiH(f)] += a;
      }
      for (Index f = 0; f < nb; ++f) fsum[bclH(f)] += std::abs(Fb(f));
      Real best = 0.0;
      Index cell = 0;
      for (Index c = 0; c < nc; ++c) {
        const Real v = fsum[c] / (2.0 * volH(c));
        if (v > best) { best = v; cell = c; }
      }
      Loc r = maxLoc(comm, {best, nc > 0 ? static_cast<long long>(gid[cell]) : -1,
                            nc > 0 ? ccH(cell,0) : 0.0, nc > 0 ? ccH(cell,1) : 0.0});
      r.v *= dtTest;
      return r;
    };
    // Cf on this rank's bottom faces, from the two-point gradient.
    auto wallCf = [&]() {
      auto u = host(solver.velocity());
      std::vector<Real> cf(bottomFace.size());
      for (std::size_t i = 0; i < bottomFace.size(); ++i) {
        const Index f = bottomFace[i], c = bclH(f);
        const Real mag = std::sqrt(ba(f,0)*ba(f,0) + ba(f,1)*ba(f,1) + ba(f,2)*ba(f,2));
        Real dn = 0.0;
        for (int k = 0; k < 3; ++k) dn += (bcen(f,k) - ccH(c,k)) * ba(f,k) / mag;
        cf[i] = 2.0 * (NU * u(c, 0) / std::abs(dn));
      }
      return gatherBottom(cf);
    };
    auto wallP = [&]() {
      auto pH = host(solver.pressure());
      std::vector<Real> pw(bottomFace.size());
      for (std::size_t i = 0; i < bottomFace.size(); ++i) pw[i] = pH(bclH(bottomFace[i]));
      return gatherBottom(pw);
    };

    Real time = 0.0;
    int step = 0, sinceChange = 0, holdUntil = 0;
    bool refused = false, steady = false, quasi = false;
    Real ceiling = 1e300;
    double secondsBefore = 0.0;
    Loc cNow;
    std::vector<int> histStep;
    std::vector<Real> histXr, histCf4;
    Separation sep;
    Real cf4 = 0.0, uc = 0.0;
    int firstStep = 1;
    if (resume) {
      firstStep = ck.step + 1; time = ck.time; sinceChange = ck.sinceChange; holdUntil = ck.holdUntil;
      refused = ck.refused != 0; ceiling = ck.ceiling; secondsBefore = ck.seconds;
      histStep = ck.histStep; histXr = ck.histXr; histCf4 = ck.histCf4;
    }
    ck = RankCheckpoint();
    auto elapsed = [&]() {
      return secondsBefore + std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    bool cappedSaid = false;
    const int ckptEvery = std::getenv("VIBEFLOW_BS_CKPT_EVERY") ?
                          std::max(1, std::atoi(std::getenv("VIBEFLOW_BS_CKPT_EVERY"))) : 500;
    long long sweepsLo = 1LL << 62, sweepsHi = -1;
    int sweepsDiffer = 0;
    VectorField src("src", nt, 3);
    for (step = firstStep; stopAt != 0; ++step) {
      if (step > 1 && (step - 1) % 100 == 0 && dtNow < dt && step > holdUntil) {
        const Real d2 = std::min(2.0 * dtNow, dt);
        if (d2 >= ceiling) {
          refused = true;
          if (!cappedSaid && r0)
            std::printf("  step %6d  ramp: dt %.3e was halved from before: holds\n", step, d2);
          cappedSaid = true;
        } else {
          const Loc c2 = courant(d2);
          refused = c2.v > C_UP;
          if (!refused) { dtNow = d2; solver.setTimeStep(dtNow); sinceChange = 0; }
          if (r0)
            std::printf("  step %6d  ramp: C %.2f at dt %.3e, at (%.4f, %.3e): %s\n", step, c2.v, d2,
                        c2.x, c2.y, refused ? "holds" : "doubles");
        }
      }
      const long long solvesBefore = pressureC.solves;
      solver.advance(ub, fb, src, momentumC, pressureC);
      {
        const long long sw = pressureC.solves - solvesBefore;
        const Real lo = comm.min(static_cast<Real>(sw)), hi = comm.max(static_cast<Real>(sw));
        if (lo != hi) ++sweepsDiffer;
        sweepsLo = std::min(sweepsLo, static_cast<long long>(lo));
        sweepsHi = std::max(sweepsHi, static_cast<long long>(hi));
      }
      time += dtNow;
      ++sinceChange;
      cNow = courant(dtNow);
      if (std::isfinite(cNow.v) && cNow.v > C_DOWN) {
        if (r0)
          std::printf("  step %6d  ramp: C %.2f at dt %.3e, at (%.4f, %.3e): halves\n", step, cNow.v,
                      dtNow, cNow.x, cNow.y);
        if (step > 100) ceiling = dtNow;
        dtNow *= 0.5; solver.setTimeStep(dtNow);
        sinceChange = 0; holdUntil = step + 100; refused = false; cappedSaid = false;
      }
      const bool atFinal = dtNow == dt || refused;
      const int atDt = atFinal ? sinceChange : 0;
      if (std::find(saveAt.begin(), saveAt.end(), step) != saveAt.end())
        saveState(out + ".state." + std::to_string(step));
      if (step % 50 == 0 || atDt >= maxFull || step >= MAX_STEPS) {
        const auto cf = wallCf();
        sep = separation(bx, cf);
        cf4 = interpAt(bx, cf, X_CF4);
        uc = centreU();
        if (!std::isfinite(cf4) || !std::isfinite(uc)) { if (r0) std::printf("  step %d: not finite\n", step); break; }
        histStep.push_back(step); histXr.push_back(sep.xr); histCf4.push_back(cf4);
        snapshot(uN, kN, wN);
        const ChangeD cu = relChange(uN, uPrev, 3), ckk = relChange(kN, kPrev, 1),
                      cw = relChange(wN, wPrev, 1);
        const Real change = std::max({cu.rel, ckk.rel, cw.rel});
        uPrev = uN; kPrev = kN; wPrev = wN;
        Real dxr = 0.0, dcf = 0.0;
        const int from = step - step / 5;
        for (std::size_t i = 0; i < histStep.size(); ++i) {
          if (histStep[i] < from) continue;
          dxr = std::max(dxr, std::abs(histXr[i] - sep.xr) / std::max(std::abs(sep.xr), 1e-300));
          dcf = std::max(dcf, std::abs(histCf4[i] - cf4) / std::abs(cf4));
        }
        const double secs = elapsed();
        const long long bounded = solver.boundedCells();   // already summed over the ranks (Sst.cpp)
        const auto pw = wallP();
        if (r0) {
          std::printf("  step %6d  t %9.3f  dt %.2e  C %.2f  xr %.5f  Cf(-4) %.6e  uc(-4) %.6f  "
                      "bubble %.3f %.3f  drift %.1e %.1e  change/50 steps %.1e  bounded %lld  (%.0f s)\n",
                      step, time, dtNow, cNow.v, sep.xr, cf4, uc, sep.c0, sep.c1, dxr, dcf, change,
                      bounded, secs);
          std::printf("      change: u %.1e at (%.4f, %.3e)  k %.1e at (%.4f, %.3e)  "
                      "omega %.1e at (%.4f, %.3e)\n",
                      cu.rel, cu.diff.x, cu.diff.y, ckk.rel, ckk.diff.x, ckk.diff.y,
                      cw.rel, cw.diff.x, cw.diff.y);
          const Real p40 = interpAt(bx, pw, 40.0);
          std::printf("LIP %d %.10e %.10e %.10e %.10e %.10e\n", step, p40,
                      2.0 * (pw[lip0] - p40), 2.0 * (pw[lip0 + 1] - p40),
                      2.0 * (pw[lip0 + 2] - p40), 2.0 * (pw[lip0 + 3] - p40));
          std::printf("EQ %d %.17g %.17g %lld %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g "
                      "%.17g %lld %lld %lld %d %lld %lld\n",
                      step, dtNow, cNow.v, cNow.gid, sep.xr, cf4, uc, sep.c0, sep.c1, dxr, dcf, change,
                      cu.rel, ckk.rel, cw.rel, cu.scale.gid, ckk.scale.gid, cw.scale.gid, sweepsDiffer,
                      sweepsLo, sweepsHi);
          std::fflush(stdout);
        }
        sweepsLo = 1LL << 62; sweepsHi = -1; sweepsDiffer = 0;
        if (atDt >= 1000 && change < 1e-6) { steady = true; break; }
        if (atDt >= 1000 && sep.xr > 0.0 && dxr <= 2e-3 && dcf <= 2e-3) { quasi = true; break; }
        if (atDt >= maxFull || step >= MAX_STEPS) break;
      }
      if (step % ckptEvery == 0) {
        RankCheckpoint c;
        c.P = P; c.rank = rank; c.nc = nc; c.nif = nif; c.hash = ownHash;
        c.step = step; c.sinceChange = sinceChange; c.holdUntil = holdUntil; c.refused = refused ? 1 : 0;
        c.time = time; c.dtNow = dtNow; c.ceiling = ceiling; c.seconds = elapsed();
        auto u = host(solver.velocity()); auto p = host(solver.pressure());
        auto k = host(solver.turbulentKineticEnergy()); auto w = host(solver.specificDissipation());
        auto F = host(solver.faceFlux());
        c.u.resize(nc * 3); c.p.resize(nc); c.k.resize(nc); c.w.resize(nc); c.F.resize(nif);
        for (Index i = 0; i < nc; ++i) {
          for (int d = 0; d < 3; ++d) c.u[i*3 + d] = u(i, d);
          c.p[i] = p(i); c.k[i] = k(i); c.w[i] = w(i);
        }
        for (Index f = 0; f < nif; ++f) c.F[f] = F(f);
        c.histStep = histStep; c.histXr = histXr; c.histCf4 = histCf4;
        // Two phases: every rank's .tmp, then the renames only if all were
        // written. A set cut short between renames is refused on resume.
        const bool wrote = writeRankTmp(ckName, c);
        if (comm.min(wrote ? 1.0 : 0.0) > 0.5) {
          if (std::rename((ckName + ".tmp").c_str(), ckName.c_str()) != 0)
            std::printf("rank %d: checkpoint at step %d not renamed\n", rank, step);
        } else if (r0) {
          std::printf("checkpoint at step %d not written on every rank: the previous set is kept\n", step);
        }
        comm.barrier();
      }
      if (step == stopAt) break;
    }

    {
      const auto cf = wallCf();
      const auto pw = wallP();
      std::vector<Real> pf;
      {
        auto pb = host(solver.boundaryPressure());      // every rank: it exchanges halos
        std::vector<Real> mine(bottomFace.size());
        for (std::size_t i = 0; i < bottomFace.size(); ++i) mine[i] = pb(bottomFace[i]);
        pf = gatherBottom(mine);
      }
      // The station profiles: an interface face once, on the rank whose cell
      // has the smaller global id; the ghost's own centroid and velocity.
      ScalarField gidF("gid", nt);
      {
        auto h = Kokkos::create_mirror_view(gidF);
        for (Index c = 0; c < nc; ++c) h(c) = static_cast<Real>(gid[c]);
        Kokkos::deep_copy(gidF, h);
        if (halo) halo->exchange(gidF);
      }
      auto gidH = host(gidF);
      VectorField uh("uh", nt, 3);
      Kokkos::deep_copy(uh, solver.velocity());
      if (halo) halo->exchange(uh);
      auto u = host(uh); auto fa = host(mesh.faceArea());
      std::vector<std::vector<double>> prof;
      for (Real at : STATIONS) {
        std::vector<double> rows;
        for (Index f = 0; f < nif; ++f) {
          const Real mag = std::sqrt(fa(f,0)*fa(f,0) + fa(f,1)*fa(f,1) + fa(f,2)*fa(f,2));
          if (std::abs(fa(f,0)) < 0.9 * mag) continue;
          Index a = ownH(f), b = neiH(f);
          if (b >= nc && !(gidH(a) < gidH(b))) continue;
          if (ccH(a,0) > ccH(b,0)) std::swap(a, b);
          if (!(ccH(a,0) < at && ccH(b,0) >= at)) continue;
          const Real t = (at - ccH(a,0)) / (ccH(b,0) - ccH(a,0));
          rows.push_back((1.0 - t) * ccH(a,1) + t * ccH(b,1));
          rows.push_back((1.0 - t) * u(a,0) + t * u(b,0));
        }
        prof.push_back(gatherTo0(comm, rows));
      }
      if (r0) {
        std::FILE* fc = std::fopen((out + ".cf").c_str(), "w");
        std::fprintf(fc, "# x  Cf on the bottom wall (backstep, SST-1994)\n");
        for (std::size_t i = 0; i < bx.size(); ++i) std::fprintf(fc, "%.10e %.10e\n", bx[i], cf[i]);
        std::fclose(fc);
        const Real p40 = interpAt(bx, pw, 40.0);
        std::FILE* fp = std::fopen((out + ".cp").c_str(), "w");
        std::fprintf(fp, "# x  Cp on the bottom wall, 0 at x = 40\n");
        for (std::size_t i = 0; i < bx.size(); ++i) std::fprintf(fp, "%.10e %.10e\n", bx[i], 2.0 * (pw[i] - p40));
        std::fclose(fp);
        const Real pf40 = interpAt(bx, pf, 40.0);
        std::FILE* fw = std::fopen((out + ".cpw").c_str(), "w");
        std::fprintf(fw, "# x  Cp from the extrapolated wall pressure, 0 at x = 40\n");
        for (std::size_t i = 0; i < bx.size(); ++i) std::fprintf(fw, "%.10e %.10e\n", bx[i], 2.0 * (pf[i] - pf40));
        std::fclose(fw);
        std::FILE* fq = std::fopen((out + ".prof").c_str(), "w");
        std::fprintf(fq, "# x_station  y  u  (each pair of cells straddling it, interpolated in x)\n");
        for (std::size_t s = 0; s < prof.size(); ++s) {
          std::vector<std::pair<Real, Real>> rows;
          for (std::size_t i = 0; i + 1 < prof[s].size(); i += 2) rows.emplace_back(prof[s][i], prof[s][i + 1]);
          std::sort(rows.begin(), rows.end());
          for (const auto& r : rows) std::fprintf(fq, "%.4f %.10e %.10e\n", STATIONS[s], r.first, r.second);
        }
        std::fclose(fq);
      }
      if (std::getenv("VIBEFLOW_BS_SAVE")) saveState(out + ".state");
      const long long bounded = solver.boundedCells();   // already summed over the ranks (Sst.cpp)
      const Real unconvP = comm.max(static_cast<Real>(pressureC.unconverged));
      const Real unconvM = comm.max(static_cast<Real>(momentumC.unconverged));
      if (r0) {
        const PisoTimings& ts = solver.timings();
        const double secs = elapsed();
        std::printf("timings: total %.0f s  pressure solve %.0f  momentum solve %.0f  assembly %.0f\n",
                    ts.total, ts.pressureSolve, ts.momentumSolve, ts.assemble + ts.pressureAssembly);
        std::printf("SOLVES ranks %d pressure %s solves %lld iterations %lld unconverged %lld "
                    "momentum %s solves %lld iterations %lld unconverged %lld\n",
                    P, pressureC.backendName().c_str(), pressureC.solves, pressureC.iterations,
                    static_cast<long long>(unconvP), momentumC.backendName().c_str(), momentumC.solves,
                    momentumC.iterations, static_cast<long long>(unconvM));
        std::printf("FINAL xr %.6f cf4 %.8e uc4 %.8f bubble %.5f %.5f steps %d full %d dt %.6g time %.3f "
                    "settled %s bounded %lld seconds %.0f\n",
                    sep.xr, cf4, uc, sep.c0, sep.c1, step, (dtNow == dt || refused) ? sinceChange : 0,
                    dtNow, time, steady ? "steady" : quasi ? "quasi" : "no", bounded, secs);
        std::fflush(stdout);
      }
    }
    comm.barrier();
    if (!std::getenv("VIBEFLOW_BS_KEEP_CKPT")) std::remove(ckName.c_str());
    rc = (steady || quasi) ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("rank %d: %s\n", rank, e.what());
    mpiAbort(2);
  }
  g_mpiDone = true;
  return rc;
}

int main(int argc, char** argv) {
  ParallelScope mpi(argc, argv);            // before PETSc, as test_backends does (ADR-047)
#ifdef VIBEFLOW_HAVE_PETSC
  PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif
  Kokkos::initialize(argc, argv);
  int rc = 0;
  if (Comm::world().size() > 1) {
    rc = runMpi(argc, argv);                // ADR-047: more than one process
  } else {
    if (argc < 3) {
      std::printf("usage: backstep <mesh.hex> <out prefix> [dt] [max steps at the full dt] [U_in]\n");
      Kokkos::finalize();
#ifdef VIBEFLOW_HAVE_PETSC
      PetscFinalize();
#endif
      return 2;
    }
    const std::string meshPath = argv[1], out = argv[2];
    const Real dt = argc > 3 ? std::atof(argv[3]) : 0.08;
    const int maxFull = argc > 4 ? std::atoi(argv[4]) : 5000;
    const Real uIn = argc > 5 ? std::atof(argv[5]) : 1.0;
    const auto t0 = std::chrono::steady_clock::now();
    // ADR-047: never resume from another rank count's checkpoint.
    if (std::getenv("VIBEFLOW_BS_RESUME") && !ckptFiles(out).perRankP.empty()) {
      std::printf("REFUSED: per-rank checkpoints %s.ckpt.*of* present; a serial march does not resume from them\n",
                  out.c_str());
      std::fflush(stdout);
      Kokkos::finalize();
#ifdef VIBEFLOW_HAVE_PETSC
      PetscFinalize();
#endif
      return 3;
    }

    const PolyMesh mesh = PolyMesh::fromHexFile(meshPath);
    const Index nc = mesh.nCells(), nt = mesh.nTotal(), nb = mesh.nBoundaryFaces();
    auto bcen = host(mesh.boundaryCentre()); auto ba = host(mesh.boundaryArea());
    Real X_IN = 1e300, X_OUT = -1e300, Y_TOP = -1e300;
    for (const auto& q : mesh.points()) {
      X_IN = std::min(X_IN, q.x); X_OUT = std::max(X_OUT, q.x); Y_TOP = std::max(Y_TOP, q.y);
    }
    View1<int> uType("uType", nb), pType("pType", nb), kind("kind", nb), wallMask("wall", nb);
    VectorField ub("ub", nb, 3);
    ScalarField fb("fb", nb), pval("pval", nb), kB("kB", nb), wB("wB", nb);
    auto hUT = Kokkos::create_mirror_view(uType); auto hPT = Kokkos::create_mirror_view(pType);
    auto hK = Kokkos::create_mirror_view(kind); auto hW = Kokkos::create_mirror_view(wallMask);
    auto hUB = Kokkos::create_mirror_view(ub); auto hFB = Kokkos::create_mirror_view(fb);
    auto hKB = Kokkos::create_mirror_view(kB); auto hWB = Kokkos::create_mirror_view(wB);
    Bottom bottom;
    int count[5] = {0, 0, 0, 0, 0};
    Real volume = 0.0;
    for (Index f = 0; f < nb; ++f) {
      const Real x = bcen(f,0), y = bcen(f,1);
      const Real mag = std::sqrt(ba(f,0)*ba(f,0) + ba(f,1)*ba(f,1) + ba(f,2)*ba(f,2));
      Patch p;
      bool isBottom = false;
      if (std::abs(ba(f,2)) > 0.9 * mag) p = SPAN;
      else if (std::abs(x - X_IN) < 1e-9) p = INLET;
      else if (std::abs(x - X_OUT) < 1e-9) p = OUTLET;
      else if (std::abs(y - Y_TOP) < 1e-9) p = x < X_SLIP ? SLIP : WALL;
      else if (std::abs(y - 1.0) < 1e-9 && x < 0.0) { p = x < X_SLIP ? SLIP : WALL; isBottom = true; }
      else if (std::abs(y) < 1e-12 && x > 0.0) { p = WALL; isBottom = true; }
      else if (std::abs(x) < 1e-12 && y < 1.0) p = WALL;             // the step's face
      else { std::printf("unclassified boundary face at (%.6f, %.6f)\n", x, y); std::exit(2); }
      ++count[p];
      hUB(f,0) = hUB(f,1) = hUB(f,2) = 0.0; hFB(f) = 0.0;
      hKB(f) = 0.0; hWB(f) = 0.0; hW(f) = 0;
      hPT(f) = static_cast<int>(PressureBC::FixedFlux);
      hK(f) = static_cast<int>(TurbulenceBC::ZeroGradient);
      switch (p) {
        case INLET:
          hUT(f) = static_cast<int>(VelocityBC::Dirichlet);
          hUB(f,0) = uIn; hFB(f) = uIn * ba(f,0);
          hK(f) = static_cast<int>(TurbulenceBC::Dirichlet); hKB(f) = K_IN; hWB(f) = W_IN;
          break;
        case OUTLET:
          hUT(f) = static_cast<int>(VelocityBC::ZeroGradient);
          hPT(f) = static_cast<int>(PressureBC::FixedValue);
          break;
        case SLIP:
          hUT(f) = static_cast<int>(VelocityBC::Slip);
          break;
        case WALL:
          hUT(f) = static_cast<int>(VelocityBC::Dirichlet);
          hK(f) = static_cast<int>(TurbulenceBC::Wall); hW(f) = 1;
          if (isBottom) bottom.face.push_back(f);
          break;
        case SPAN:
          hUT(f) = static_cast<int>(VelocityBC::ZeroGradient);
          break;
      }
    }
    std::sort(bottom.face.begin(), bottom.face.end(),
              [&](Index a, Index b) { return bcen(a,0) < bcen(b,0); });
    for (Index f : bottom.face) bottom.x.push_back(bcen(f,0));
    {
      auto vol = host(mesh.cellVolume());
      for (Index c = 0; c < nc; ++c) volume += vol(c);
    }
    Real zmin = 1e300, zmax = -1e300;
    for (const auto& q : mesh.points()) { zmin = std::min(zmin, q.z); zmax = std::max(zmax, q.z); }
    std::printf("backward-facing step, SST-1994, %s: %d cells, volume %.15g (area %.15g), "
                "dt %g, U_in %.6f\n", meshPath.c_str(), static_cast<int>(nc), volume,
                volume / (zmax - zmin), dt, uIn);
    std::printf("  patches: inlet %d  outlet %d  slip %d  wall %d  span %d  (bottom wall %zu)\n",
                count[INLET], count[OUTLET], count[SLIP], count[WALL], count[SPAN],
                bottom.face.size());
    // The lip faces (ADR-046's first revision): the four upstream entries of
    // the sorted bottom wall nearest the step, in increasing x.
    const std::size_t nUp = static_cast<std::size_t>(
        std::count_if(bottom.x.begin(), bottom.x.end(), [](Real x) { return x < 0.0; }));
    if (nUp < 4) { std::printf("fewer than four upstream wall faces\n"); std::exit(2); }
    const std::size_t lip0 = nUp - 4;
    std::printf("LIPX %.10e %.10e %.10e %.10e\n",
                bottom.x[lip0], bottom.x[lip0 + 1], bottom.x[lip0 + 2], bottom.x[lip0 + 3]);
    Kokkos::deep_copy(uType, hUT); Kokkos::deep_copy(pType, hPT); Kokkos::deep_copy(kind, hK);
    Kokkos::deep_copy(wallMask, hW); Kokkos::deep_copy(ub, hUB); Kokkos::deep_copy(fb, hFB);
    Kokkos::deep_copy(kB, hKB); Kokkos::deep_copy(wB, hWB);

    PisoControls ctl;
    ctl.outer = 1;
    // Eight correctors (ADR-045's fifth revision): four, the flat plate's
    // (ADR-043), let a disturbance grow behind the step's corner.
    ctl.correctors = 8;
    if (const char* e = std::getenv("VIBEFLOW_CORRECTORS")) ctl.correctors = std::atoi(e);
    ctl.convection = ConvectionScheme::LinearUpwind;
    // Exploration only: the face value (linear, or upwind for first order)
    // and the explicit non-orthogonal diffusion.
    if (const char* e = std::getenv("VIBEFLOW_CONVECTION")) {
      const std::string c(e);
      if (c == "linear") ctl.convection = ConvectionScheme::Linear;
      else if (c == "upwind") ctl.deferredCorrection = false;
    }
    if (std::getenv("VIBEFLOW_NO_NONORTH")) ctl.diffusionNonOrth = false;
    // ADR-047's equivalence gates only: a fixed number of sweeps.
    if (const char* e = std::getenv("VIBEFLOW_BS_SWEEPS")) {
      ctl.nonOrthCorrectors = std::atoi(e); ctl.nonOrthTol = 0.0;
    }
    // From the uniform stream the ramp starts at dt/64; from another march's
    // developed state at dt/8 (the fifth revision).
    const char* initPath = std::getenv("VIBEFLOW_BS_INIT");
    Real dtNow = initPath ? dt / 8.0 : dt / 64.0;
    if (const char* e = std::getenv("VIBEFLOW_BS_RAMP")) dtNow = dt / std::max(1.0, std::atof(e));
    // A stopped march resumes from its checkpoint (VIBEFLOW_BS_RESUME).
    Checkpoint ck;
    const bool resume = std::getenv("VIBEFLOW_BS_RESUME") && readCheckpoint(out + ".ckpt", ck) &&
                        ck.nc == static_cast<long>(nc) && ck.nif == static_cast<long>(mesh.nInternalFaces());
    if (resume) dtNow = ck.dtNow;
    PisoSolver solver(mesh, NU, dtNow, ctl);
    solver.setBoundaryTypes(uType);
    solver.setPressureBoundary(pType, pval);
    std::printf("  PISO correctors %d, first dt %.4g\n", ctl.correctors, dtNow);
    if (resume)
      std::printf("  resumed from %s.ckpt: step %d, t %.3f, dt %.4g, %d steps at it\n", out.c_str(), ck.step,
                  ck.time, ck.dtNow, ck.sinceChange);

    CoarseState coarse;
    std::vector<Index> fromCoarse;
    auto ccH = host(mesh.cellCentre());
    if (initPath && !resume) {
      coarse = readState(initPath);
      fromCoarse.resize(nt);
      for (Index c = 0; c < nt; ++c) fromCoarse[c] = nearestCoarse(coarse, ccH(c,0), ccH(c,1));
      std::printf("  initial state: %s (%zu coarse cells)\n", initPath, coarse.v.size());
    }
    {
      VectorField u0("u0", nt, 3);
      ScalarField p0("p0", nt), F0("F0", mesh.nInternalFaces());
      auto hu = host(u0); auto hp = host(p0);
      for (Index c = 0; c < nt; ++c) {
        if (resume && c < nc) {
          hu(c,0) = ck.u[c*3]; hu(c,1) = ck.u[c*3 + 1]; hu(c,2) = ck.u[c*3 + 2]; hp(c) = ck.p[c];
        } else if (initPath && !resume) {
          const auto& q = coarse.v[fromCoarse[c]];
          hu(c,0) = q[2]; hu(c,1) = q[3]; hu(c,2) = q[4]; hp(c) = q[5];
        } else {
          hu(c,0) = uIn; hu(c,1) = 0.0; hu(c,2) = 0.0; hp(c) = 0.0;
        }
      }
      Kokkos::deep_copy(u0, hu); Kokkos::deep_copy(p0, hp);
      auto fa = mesh.faceArea(); auto fc = mesh.faceCentre(); auto ccd = mesh.cellCentre();
      auto own = mesh.owner(); auto nei = mesh.neighbour();
      Kokkos::parallel_for("F0", Kokkos::RangePolicy<ExecSpace>(0, mesh.nInternalFaces()),
        KOKKOS_LAMBDA(const Index f) {
          Real lo = 0.0, ln = 0.0;
          for (int i = 0; i < 3; ++i) {
            const Real ro = fc(f, i) - ccd(own(f), i), rn = fc(f, i) - ccd(nei(f), i);
            lo += ro * ro; ln += rn * rn;
          }
          const Real w = Kokkos::sqrt(ln) / (Kokkos::sqrt(lo) + Kokkos::sqrt(ln));
          Real sum = 0.0;
          for (int i = 0; i < 3; ++i) sum += (w * u0(own(f), i) + (1.0 - w) * u0(nei(f), i)) * fa(f, i);
          F0(f) = sum;
        });
      Kokkos::fence();
      if (resume) {
        // The checkpoint's own face flux, not an interpolation of it.
        auto hF = host(F0);
        for (Index f = 0; f < mesh.nInternalFaces(); ++f) hF(f) = ck.F[f];
        Kokkos::deep_copy(F0, hF);
      }
      solver.setState(u0, p0, F0);
    }
    TurbulenceModel tm;
    tm.variant = SstVariant::Menter1994;
    if (std::getenv("VIBEFLOW_BS_FROZEN")) tm.frozen = true;      // exploration only
    if (std::getenv("VIBEFLOW_BS_NO_TRANSPOSE")) tm.transposeStress = false;   // exploration only
    solver.enableTurbulence(tm, wallMask);
    solver.setTurbulenceBoundary(kind, kB, wB);
    {
      ScalarField k0("k0", nt), w0("w0", nt);
      Kokkos::deep_copy(k0, K_IN); Kokkos::deep_copy(w0, W_IN);
      if (resume) {
        auto hk = host(k0); auto hw = host(w0);
        for (Index c = 0; c < nc; ++c) { hk(c) = ck.k[c]; hw(c) = ck.w[c]; }
        Kokkos::deep_copy(k0, hk); Kokkos::deep_copy(w0, hw);
      } else if (initPath) {
        auto hk = host(k0); auto hw = host(w0);
        for (Index c = 0; c < nt; ++c) {
          hk(c) = coarse.v[fromCoarse[c]][6]; hw(c) = coarse.v[fromCoarse[c]][7];
        }
        Kokkos::deep_copy(k0, hk); Kokkos::deep_copy(w0, hw);
      }
      solver.setTurbulence(k0, w0);
    }

    std::unique_ptr<LinearSolver> momentumPtr;
    {
      const char* e = std::getenv("VIBEFLOW_MOMENTUM");
#ifdef VIBEFLOW_HAVE_PETSC
      const std::string cfg = e ? e : "bicgstab+ilu";
#else
      const std::string cfg = e ? e : "native";
#endif
      if (cfg == "native") momentumPtr = std::make_unique<NativeBiCGStab>(mesh);
#ifdef VIBEFLOW_HAVE_PETSC
      else momentumPtr = std::make_unique<PetscSolver>(mesh, Comm(), cfg);
#else
      else { std::fprintf(stderr, "built without PETSc\n"); std::exit(2); }
#endif
      std::printf("  momentum, k and omega solver: %s\n", cfg.c_str());
    }
    LinearSolver& momentum = *momentumPtr;
    auto pressure = makePressureSolver(mesh);
    // ADR-047: the counts behind the SOLVES line and each step's sweep count.
    Counted momentumC(momentum), pressureC(*pressure);
    long long sweepsLo = 1LL << 62, sweepsHi = -1;

    // The channel's centre velocity at x = -4: the two columns around it at
    // the row nearest y = 5.
    const Columns c4 = columnsAt(mesh, X_CF4);
    auto centreU = [&]() {
      auto u = host(solver.velocity());
      auto near5 = [&](const std::vector<Index>& col) {
        Index best = col[0];
        for (Index c : col) if (std::abs(ccH(c,1) - 5.0) < std::abs(ccH(best,1) - 5.0)) best = c;
        return best;
      };
      const Index a = near5(c4.left), b = near5(c4.right);
      const Real t = (X_CF4 - c4.xl) / (c4.xr - c4.xl);
      return (1.0 - t) * u(a, 0) + t * u(b, 0);
    };

    std::vector<Real> uPrev, kPrev, wPrev, uN, kN, wN;
    auto snapshot = [&](std::vector<Real>& uu, std::vector<Real>& kk, std::vector<Real>& ww) {
      auto u = host(solver.velocity()); auto k = host(solver.turbulentKineticEnergy());
      auto w = host(solver.specificDissipation());
      uu.resize(nc * 3); kk.resize(nc); ww.resize(nc);
      for (Index c = 0; c < nc; ++c) {
        for (int d = 0; d < 3; ++d) uu[c*3 + d] = u(c, d);
        kk[c] = k(c); ww[c] = w(c);
      }
    };
    struct Change { Real rel = 0.0; Index cell = 0; };
    auto relChange = [](const std::vector<Real>& a, const std::vector<Real>& b, int stride) {
      Real d = 0.0, s = 1e-300;
      Change c;
      for (std::size_t i = 0; i < a.size(); ++i) {
        const Real di = std::abs(a[i] - b[i]);
        if (di > d) { d = di; c.cell = static_cast<Index>(i / stride); }
        s = std::max(s, std::abs(a[i]));
      }
      c.rel = d / s;
      return c;
    };
    snapshot(uPrev, kPrev, wPrev);

    // The state for a finer grid's start: x y u v w p k omega per cell.
    auto saveState = [&](const std::string& path) {
      auto u = host(solver.velocity()); auto p = host(solver.pressure());
      auto k = host(solver.turbulentKineticEnergy()); auto w = host(solver.specificDissipation());
      std::FILE* fs = std::fopen(path.c_str(), "w");
      std::fprintf(fs, "%ld\n", static_cast<long>(nc));
      for (Index c = 0; c < nc; ++c)
        std::fprintf(fs, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", ccH(c,0), ccH(c,1),
                     u(c,0), u(c,1), u(c,2), p(c), k(c), w(c));
      std::fclose(fs);
    };
    // Exploration only: VIBEFLOW_BS_SAVE_AT=<step>[,<step>...] also saves
    // the state after those steps, to <out>.state.<step>; VIBEFLOW_BS_STOP
    // ends the march after the step it names.
    std::vector<int> saveAt;
    if (const char* e = std::getenv("VIBEFLOW_BS_SAVE_AT")) {
      std::string list(e);
      for (std::size_t a = 0; a < list.size();) {
        const std::size_t b = std::min(list.find(',', a), list.size());
        saveAt.push_back(std::atoi(list.substr(a, b - a).c_str()));
        a = b + 1;
      }
    }
    const int stopAt = std::getenv("VIBEFLOW_BS_STOP") ? std::atoi(std::getenv("VIBEFLOW_BS_STOP")) : -1;

    // The ramp follows the flow's largest cell Courant number (ADR-045's
    // third revision): C = dt sum_f |F_f| / 2V. Every 100 steps dt doubles,
    // never past the level's dt, if C at the doubled dt is at most C_UP;
    // whenever C exceeds C_DOWN it halves and holds for 100 steps.
    // 3 and 6 since ADR-045's sixth revision (8 and 12 before it).
    Real C_UP = 3.0, C_DOWN = 6.0;
    // Exploration only: other thresholds (VIBEFLOW_BS_CUP, VIBEFLOW_BS_CDOWN).
    if (const char* e = std::getenv("VIBEFLOW_BS_CUP")) C_UP = std::atof(e);
    if (const char* e = std::getenv("VIBEFLOW_BS_CDOWN")) C_DOWN = std::atof(e);
    const int MAX_STEPS = 20000;
    auto ownH = host(mesh.owner()); auto neiH = host(mesh.neighbour());
    auto bclH = host(mesh.boundaryCell()); auto volH = host(mesh.cellVolume());
    const Index nif = mesh.nInternalFaces();
    std::vector<Real> fsum(nt);
    struct Cfl { Real c = 0.0; Index cell = 0; };
    auto courant = [&](Real dtTest) {
      auto F = host(solver.faceFlux()); auto Fb = host(solver.boundaryFlux());
      std::fill(fsum.begin(), fsum.end(), 0.0);
      for (Index f = 0; f < nif; ++f) {
        const Real a = std::abs(F(f));
        fsum[ownH(f)] += a; fsum[neiH(f)] += a;
      }
      for (Index f = 0; f < nb; ++f) fsum[bclH(f)] += std::abs(Fb(f));
      Cfl r;
      for (Index c = 0; c < nc; ++c) {
        const Real v = fsum[c] / (2.0 * volH(c));
        if (v > r.c) { r.c = v; r.cell = c; }
      }
      r.c *= dtTest;
      return r;
    };

    VectorField src("src", nt, 3);
    Real time = 0.0;
    int step = 0, sinceChange = 0, holdUntil = 0;
    bool refused = false, steady = false, quasi = false;
    // A dt the ramp halved from is not taken again (the fifth revision).
    Real ceiling = 1e300;
    double secondsBefore = 0.0;
    Cfl cNow;
    // Every 50 steps: the reattachment point and Cf at x = -4, with the step
    // they were taken at, for the quasi-steady test.
    std::vector<int> histStep;
    std::vector<Real> histXr, histCf4;
    Separation sep;
    Real cf4 = 0.0, uc = 0.0;
    int firstStep = 1;
    if (resume) {
      firstStep = ck.step + 1; time = ck.time; sinceChange = ck.sinceChange; holdUntil = ck.holdUntil;
      refused = ck.refused != 0; ceiling = ck.ceiling; secondsBefore = ck.seconds;
      histStep = ck.histStep; histXr = ck.histXr; histCf4 = ck.histCf4;
    }
    auto elapsed = [&]() {
      return secondsBefore + std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    bool cappedSaid = false;
    // The checkpoint every 500 steps (VIBEFLOW_BS_CKPT_EVERY, for testing).
    const int ckptEvery = std::getenv("VIBEFLOW_BS_CKPT_EVERY") ?
                          std::max(1, std::atoi(std::getenv("VIBEFLOW_BS_CKPT_EVERY"))) : 500;
    // VIBEFLOW_BS_STOP=0: no step at all -- the outputs of the initial state
    // (used to extract a finished march's profiles again from its state).
    for (step = firstStep; stopAt != 0; ++step) {
      if (step > 1 && (step - 1) % 100 == 0 && dtNow < dt && step > holdUntil) {
        const Real d2 = std::min(2.0 * dtNow, dt);
        if (d2 >= ceiling) {
          refused = true;
          if (!cappedSaid)
            std::printf("  step %6d  ramp: dt %.3e was halved from before: holds\n", step, d2);
          cappedSaid = true;
        } else {
          const Cfl c2 = courant(d2);
          refused = c2.c > C_UP;
          if (!refused) { dtNow = d2; solver.setTimeStep(dtNow); sinceChange = 0; }
          std::printf("  step %6d  ramp: C %.2f at dt %.3e, at (%.4f, %.3e): %s\n", step, c2.c, d2,
                      ccH(c2.cell,0), ccH(c2.cell,1), refused ? "holds" : "doubles");
        }
      }
      const long long solvesBefore = pressureC.solves;
      solver.advance(ub, fb, src, momentumC, pressureC);
      {
        const long long sw = pressureC.solves - solvesBefore;   // this step's sweeps (ADR-047)
        sweepsLo = std::min(sweepsLo, sw); sweepsHi = std::max(sweepsHi, sw);
      }
      time += dtNow;
      ++sinceChange;
      cNow = courant(dtNow);
      if (std::isfinite(cNow.c) && cNow.c > C_DOWN) {
        std::printf("  step %6d  ramp: C %.2f at dt %.3e, at (%.4f, %.3e): halves\n", step, cNow.c,
                    dtNow, ccH(cNow.cell,0), ccH(cNow.cell,1));
        // A halving in the march's first 100 steps -- the start's own
        // transient -- sets no ceiling (the seventh revision).
        if (step > 100) ceiling = dtNow;
        dtNow *= 0.5; solver.setTimeStep(dtNow);
        sinceChange = 0; holdUntil = step + 100; refused = false; cappedSaid = false;
      }
      // Steps at the dt the march settles at: the level's, or the one the
      // cap last kept it at.
      const bool atFinal = dtNow == dt || refused;
      const int atDt = atFinal ? sinceChange : 0;
      if (std::find(saveAt.begin(), saveAt.end(), step) != saveAt.end())
        saveState(out + ".state." + std::to_string(step));
      if (std::getenv("VIBEFLOW_BS_DEBUG")) {
        // Exploration only: where the fields go, step by step.
        auto u = host(solver.velocity()); auto k = host(solver.turbulentKineticEnergy());
        auto w = host(solver.specificDissipation()); auto nut = host(solver.eddyViscosity());
        Index iu = 0, ik = 0, iw = 0, in = 0;
        for (Index c = 0; c < nc; ++c) {
          if (std::abs(u(c,0)) + std::abs(u(c,1)) > std::abs(u(iu,0)) + std::abs(u(iu,1))) iu = c;
          if (k(c) < k(ik)) ik = c;
          if (w(c) > w(iw)) iw = c;
          if (nut(c) > nut(in)) in = c;
        }
        std::printf("    dbg %4d  u (%.3e, %.3e) |u| max %.3e at (%.3f, %.3e)  k min %.3e at (%.3f, %.3e)  "
                    "w max %.3e at (%.3f, %.3e)  nut/nu max %.3e at (%.3f, %.3e)  bounded %lld\n",
                    step, u(iu,0), u(iu,1), std::abs(u(iu,0)) + std::abs(u(iu,1)), ccH(iu,0), ccH(iu,1), k(ik),
                    ccH(ik,0), ccH(ik,1), w(iw), ccH(iw,0), ccH(iw,1), nut(in) / NU, ccH(in,0),
                    ccH(in,1), solver.boundedCells());
        std::fflush(stdout);
      }
      if (step % 50 == 0 || atDt >= maxFull || step >= MAX_STEPS) {
        const auto tau = wallStress(mesh, solver, bottom);
        std::vector<Real> cf(tau.size());
        for (std::size_t i = 0; i < tau.size(); ++i) cf[i] = 2.0 * tau[i];
        sep = separation(bottom.x, cf);
        cf4 = interpAt(bottom.x, cf, X_CF4);
        uc = centreU();
        if (!std::isfinite(cf4) || !std::isfinite(uc)) { std::printf("  step %d: not finite\n", step); break; }
        histStep.push_back(step); histXr.push_back(sep.xr); histCf4.push_back(cf4);
        snapshot(uN, kN, wN);
        const Change cu = relChange(uN, uPrev, 3), ck = relChange(kN, kPrev, 1),
                     cw = relChange(wN, wPrev, 1);
        const Real change = std::max({cu.rel, ck.rel, cw.rel});
        uPrev = uN; kPrev = kN; wPrev = wN;
        // Over the last 20% of the march: how far x_r and Cf(-4) moved.
        Real dxr = 0.0, dcf = 0.0;
        const int from = step - step / 5;
        for (std::size_t i = 0; i < histStep.size(); ++i) {
          if (histStep[i] < from) continue;
          dxr = std::max(dxr, std::abs(histXr[i] - sep.xr) / std::max(std::abs(sep.xr), 1e-300));
          dcf = std::max(dcf, std::abs(histCf4[i] - cf4) / std::abs(cf4));
        }
        const double secs = elapsed();
        std::printf("  step %6d  t %9.3f  dt %.2e  C %.2f  xr %.5f  Cf(-4) %.6e  uc(-4) %.6f  "
                    "bubble %.3f %.3f  drift %.1e %.1e  change/50 steps %.1e  bounded %lld  (%.0f s)\n",
                    step, time, dtNow, cNow.c, sep.xr, cf4, uc, sep.c0, sep.c1, dxr, dcf, change,
                    solver.boundedCells(), secs);
        std::printf("      change: u %.1e at (%.4f, %.3e)  k %.1e at (%.4f, %.3e)  "
                    "omega %.1e at (%.4f, %.3e)\n",
                    cu.rel, ccH(cu.cell,0), ccH(cu.cell,1), ck.rel, ccH(ck.cell,0), ccH(ck.cell,1),
                    cw.rel, ccH(cw.cell,0), ccH(cw.cell,1));
        {
          // The lip print (ADR-046's first revision): .cp's expressions, no
          // time field, nothing called on the solver but pressure().
          auto pH = host(solver.pressure());
          std::vector<Real> pw(bottom.face.size());
          for (std::size_t i = 0; i < pw.size(); ++i) pw[i] = pH(bclH(bottom.face[i]));
          const Real p40 = interpAt(bottom.x, pw, 40.0);
          std::printf("LIP %d %.10e %.10e %.10e %.10e %.10e\n", step, p40,
                      2.0 * (pw[lip0] - p40), 2.0 * (pw[lip0 + 1] - p40),
                      2.0 * (pw[lip0 + 2] - p40), 2.0 * (pw[lip0 + 3] - p40));
        }
        // ADR-047's EQ line: the decision values at full precision; a cell is
        // its global id (serially the file's order). The last three fields:
        // steps since the last line whose sweep count differed between ranks
        // (none, serially), and the fewest and most sweeps over those steps.
        std::printf("EQ %d %.17g %.17g %lld %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g "
                    "%.17g %lld %lld %lld %d %lld %lld\n",
                    step, dtNow, cNow.c, static_cast<long long>(cNow.cell), sep.xr, cf4, uc, sep.c0,
                    sep.c1, dxr, dcf, change, cu.rel, ck.rel, cw.rel,
                    static_cast<long long>(scaleCell(uN, 3)), static_cast<long long>(scaleCell(kN, 1)),
                    static_cast<long long>(scaleCell(wN, 1)), 0, sweepsLo, sweepsHi);
        sweepsLo = 1LL << 62; sweepsHi = -1;
        std::fflush(stdout);
        if (atDt >= 1000 && change < 1e-6) { steady = true; break; }
        if (atDt >= 1000 && sep.xr > 0.0 && dxr <= 2e-3 && dcf <= 2e-3) { quasi = true; break; }
        if (atDt >= maxFull || step >= MAX_STEPS) break;
      }
      if (step % ckptEvery == 0) {
        Checkpoint c;
        c.nc = nc; c.nif = mesh.nInternalFaces();
        c.step = step; c.sinceChange = sinceChange; c.holdUntil = holdUntil; c.refused = refused ? 1 : 0;
        c.time = time; c.dtNow = dtNow; c.ceiling = ceiling; c.seconds = elapsed();
        auto u = host(solver.velocity()); auto p = host(solver.pressure());
        auto k = host(solver.turbulentKineticEnergy()); auto w = host(solver.specificDissipation());
        auto F = host(solver.faceFlux());
        c.u.resize(nc * 3); c.p.resize(nc); c.k.resize(nc); c.w.resize(nc); c.F.resize(c.nif);
        for (Index i = 0; i < nc; ++i) {
          for (int d = 0; d < 3; ++d) c.u[i*3 + d] = u(i, d);
          c.p[i] = p(i); c.k[i] = k(i); c.w[i] = w(i);
        }
        for (Index f = 0; f < c.nif; ++f) c.F[f] = F(f);
        c.histStep = histStep; c.histXr = histXr; c.histCf4 = histCf4;
        writeCheckpoint(out + ".ckpt", c);
      }
      if (step == stopAt) break;
    }

    {
      const auto tau = wallStress(mesh, solver, bottom);
      std::FILE* fc = std::fopen((out + ".cf").c_str(), "w");
      std::fprintf(fc, "# x  Cf on the bottom wall (backstep, SST-1994)\n");
      for (std::size_t i = 0; i < tau.size(); ++i)
        std::fprintf(fc, "%.10e %.10e\n", bottom.x[i], 2.0 * tau[i]);
      std::fclose(fc);
      // Cp from the wall cells' pressure, shifted to zero at x = 40.
      auto p = host(solver.pressure()); auto bcell = host(mesh.boundaryCell());
      std::vector<Real> pw(bottom.face.size());
      for (std::size_t i = 0; i < pw.size(); ++i) pw[i] = p(bcell(bottom.face[i]));
      const Real p40 = interpAt(bottom.x, pw, 40.0);
      std::FILE* fp = std::fopen((out + ".cp").c_str(), "w");
      std::fprintf(fp, "# x  Cp on the bottom wall, 0 at x = 40\n");
      for (std::size_t i = 0; i < pw.size(); ++i)
        std::fprintf(fp, "%.10e %.10e\n", bottom.x[i], 2.0 * (pw[i] - p40));
      std::fclose(fp);
      // Reported, not gated (after ADR-045's results): Cp from the solver's
      // own wall pressure, the cell's extrapolated to the face.
      {
        auto pb = host(solver.boundaryPressure());
        std::vector<Real> pf(bottom.face.size());
        for (std::size_t i = 0; i < pf.size(); ++i) pf[i] = pb(bottom.face[i]);
        const Real pf40 = interpAt(bottom.x, pf, 40.0);
        std::FILE* fw = std::fopen((out + ".cpw").c_str(), "w");
        std::fprintf(fw, "# x  Cp from the extrapolated wall pressure, 0 at x = 40\n");
        for (std::size_t i = 0; i < pf.size(); ++i)
          std::fprintf(fw, "%.10e %.10e\n", bottom.x[i], 2.0 * (pf[i] - pf40));
        std::fclose(fw);
      }
      // The profiles at the stations.
      auto u = host(solver.velocity());
      std::FILE* fq = std::fopen((out + ".prof").c_str(), "w");
      std::fprintf(fq, "# x_station  y  u  (each pair of cells straddling it, interpolated in x)\n");
      for (Real xs : STATIONS) {
        const Profile pr = stationProfile(mesh, u, xs);
        for (std::size_t j = 0; j < pr.y.size(); ++j)
          std::fprintf(fq, "%.4f %.10e %.10e\n", xs, pr.y[j], pr.u[j]);
      }
      std::fclose(fq);
      if (std::getenv("VIBEFLOW_BS_SAVE")) saveState(out + ".state");
      const PisoTimings& ts = solver.timings();
      const double secs = elapsed();
      std::printf("timings: total %.0f s  pressure solve %.0f  momentum solve %.0f  assembly %.0f\n",
                  ts.total, ts.pressureSolve, ts.momentumSolve, ts.assemble + ts.pressureAssembly);
      // ADR-047's SOLVES line: reported, not judged.
      std::printf("SOLVES ranks 1 pressure %s solves %lld iterations %lld unconverged %lld "
                  "momentum %s solves %lld iterations %lld unconverged %lld\n",
                  pressureC.backendName().c_str(), pressureC.solves, pressureC.iterations,
                  pressureC.unconverged, momentumC.backendName().c_str(), momentumC.solves,
                  momentumC.iterations, momentumC.unconverged);
      std::printf("FINAL xr %.6f cf4 %.8e uc4 %.8f bubble %.5f %.5f steps %d full %d dt %.6g time %.3f "
                  "settled %s bounded %lld seconds %.0f\n",
                  sep.xr, cf4, uc, sep.c0, sep.c1, step, (dtNow == dt || refused) ? sinceChange : 0,
                  dtNow, time,
                  steady ? "steady" : quasi ? "quasi" : "no", solver.boundedCells(), secs);
    }
    if (!std::getenv("VIBEFLOW_BS_KEEP_CKPT"))   // ADR-047's equivalence gates keep it
      std::remove((out + ".ckpt").c_str());     // the march is done
    rc = (steady || quasi) ? 0 : 1;
  }
  Kokkos::finalize();
#ifdef VIBEFLOW_HAVE_PETSC
  PetscFinalize();
#endif
  return rc;
}
