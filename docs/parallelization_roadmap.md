# Parallelization Roadmap for PolyStokes

*Goal: speed up a single large simulation. Target: multi-node cluster.*

## Context

The production workload (`test_sim_dumbbells_thermal.py`) is: `mm_HI=False`, a **single
trapped colloid**, `Nm ~ 10^5` monomers (50k dumbbells), `box=[10,10,10]`, `dt=1e-5`,
`tmax=5.0` → **~500k timesteps**. Runs are currently **serial single-rank**: PETSc uses
`PETSC_COMM_WORLD` as a handle but every `Vec`/data `Mat` is `PETSC_COMM_SELF`/
`VecCreateSeq`, so running on >1 MPI rank today just replicates work with no speedup
(`src/matfree_A.cpp:133-142`, `src/init.cpp:39-84`, `src/arrays.cpp`). The latest commit
message is literally "todo: run on cluster."

This regime is favorable to parallelize. With `mm_HI=False` the grand mobility is the
**arrowhead** operator (`src/matfree_A.cpp`): monomer self-mobility is diagonal
(`beta_inv*I`), the only stored coupling is the small colloid↔monomer block `Mcm_block`
(nc11 × nm3, nc11 = 11 for one colloid), and the KSP matvec is **O(Nm)**, not O(Nm²). All
per-step cost is O(Nm) work that partitions cleanly over monomers.

### Where the per-step time goes
Per step the sequence `check_dist(); RHS(); mob(); solve` runs **twice** (predictor +
corrector, `src/run.cpp:86-118`), and `drift()` runs `mob()`+saddle-solve **twice more**
(RFD probes at ±eps, `src/drift.cpp:32-54`). So `mob()` assembly + a GMRES solve are
evaluated **~4× per step**. Dominant O(Nm) costs:

- **`mob()` AB loop** building `Mcm_block` (`src/mob.cpp:975-1002`) — already uses the
  **thread-safe** `mobility_AB()` (`src/mob.cpp:362-537`) writing stack-local buffers into
  disjoint dense columns. Nearly OpenMP-ready.
- **`check_dist()`** pairwise separations; **`RHS()`** bond/trap/WCA forces (WCA via the
  per-object monomer cell list, `src/cell_list.cpp`).
- **GMRES+Jacobi solve**: matvec is `ArrowheadMult` (`src/matfree_A.cpp:38-80`), two dense
  `Mcm_block` matvecs O(Nc·Nm) — threaded/distributable BLAS.
- The Brownian slip-velocity and colloid Schur eigendecomposition are **negligible** here
  (nc11 = 11; dense syev on an 11×11, `src/slip_vel.cpp:81-306`).

## Recommended approach — staged, lowest-risk-first

### Stage 0 — Profiling baseline *(prerequisite, ~0.5 day)*
Turn on `PetscLogView` (already stubbed at `src/run.cpp:148`) and run at production Nm for
a few hundred steps. Record the wall-time split across `mob()`, `KSPSolve`, `drift()`,
`check_dist`/`RHS`. This ranks the OpenMP/MPI targets by real cost and sets the speedup
yardstick. **Deliverable:** a `-log_view` table committed to `docs/`.

### Stage 1 — OpenMP within a node *(fast win, ~1 week)*
Shared-memory threading of the O(Nm) loops. No change to the global `arrays::` namespace
(single process, shared memory); correctness hinges on partitioning *writes*.

- **`mob()` AB loop** (`src/mob.cpp:983-1001`): add `#pragma omp parallel for`. Already
  safe — `mobility_AB()` is `const` and writes only its output buffers; each iteration
  writes disjoint columns of the `Mcm_block` dense array. Privatize the `la/lb/lbt/lgt`
  buffers (`src/mob.cpp:979-982`) per thread.
- **`check_dist()` / `RHS()` force loops**: parallelize over monomers/cells. For WCA via
  the cell list, parallelize over *cells* and accumulate per-thread force partials (or
  atomic adds) to avoid the neighbor write race.
- **`drift()`**: the O(N) position-perturb / difference loops (`src/drift.cpp:32-57`) are
  trivially parallel; the two embedded `mob()`+solve calls inherit Stage-1 threading.
- **Threaded BLAS** for the dense `Mcm_block` matvecs: link threaded OpenBLAS/MKL, set
  `OMP_NUM_THREADS`.
- **Per-thread RNG**: the single `std::mt19937_64 rng` (`src/Stokes.h:56`) serializes the
  noise fills (`sample_slip_vel`, `sample_drift_displacement`). Give each thread an
  independent stream (e.g. seed = base_seed ^ thread_id).
- Re-enable OpenMP in CMake (a prior OpenMP attempt was reverted per git log):
  `find_package(OpenMP)` + `target_link_libraries(... OpenMP::OpenMP_CXX)`.

**Value:** near-core-count speedup on one node, de-risks Stage 2, and Stage-0 numbers say
whether going multi-node is even worth it.

### Stage 2 — MPI particle decomposition via PETSc *(multi-node scaling, ~3–5 weeks)*
This is what "one big run on a cluster" ultimately requires. The arrowhead structure makes
it clean because the monomer block is diagonal and the colloid block is tiny.

- **Distribute monomer DOFs across ranks.** Convert the arrowhead work vectors and saddle
  Vecs from `VecCreateSeq(PETSC_COMM_SELF, …)` to parallel `VecCreate(PETSC_COMM_WORLD)`
  with monomer rows partitioned (`src/matfree_A.cpp:133-140`, `src/arrays.cpp` Vec
  creations). Keep the 11 colloid DOFs replicated/owned by one rank.
- **`Mcm_block` → distributed** `MPIDENSE`/`MPIAIJ` with monomer *columns* local to each
  rank. The arrowhead matvec (`ArrowheadMult`) becomes: each rank applies its local monomer
  coupling; the colloid↔monomer contraction is a small `MPI_Allreduce` over nc11 = 11
  scalars. `M^mm = beta_inv·I` is embarrassingly parallel. GMRES + Jacobi PC over
  `PETSC_COMM_WORLD` then work directly on distributed operands.
- **Distribute force/neighbor work.** Partition monomers spatially; the WCA cell list
  (`src/cell_list.cpp`) needs ghost/halo monomers from neighbor ranks (one halo exchange
  per step), or reuse a PETSc `DMSwarm`/star-forest for the monomer–colloid pairs. Since HI
  is a single central colloid, every rank needs only that colloid's state (broadcast,
  trivial); the expensive monomer–monomer overlap is short-range and local.
- **`set_vars()` pair enumeration** (`src/init.cpp:165-347`) must build only the local
  monomer partition's lists instead of the full replicated lists.
- **RNG**: independent parallel streams per rank (rank-offset seed, or PETSc `PetscRandom`).
- **Colloid Schur / slip-vel** stays serial on the colloid-owning rank (nc11 = 11, cheap);
  broadcast the resulting colloid slip velocity.
- **Global `arrays::` namespace is fine for MPI** (each rank is its own process) — but its
  contents must become *distributed* storage, not the current replicated `SELF` storage, or
  ranks just duplicate work. This is the bulk of the effort.

### Stage 3 — Hybrid + optional GPU *(only if Stage 0/2 justify it)*
- **Hybrid MPI+OpenMP**: MPI across nodes, OpenMP (Stage 1) within each node — the usual
  cluster sweet spot; avoids over-decomposing to tiny per-rank monomer counts.
- **GPU (optional)**: offload the dense `Mcm_block` matvecs via `MATDENSECUDA`/`VECCUDA`;
  the per-pair `mobility_AB` arithmetic is a natural CUDA kernel. Most impactful for the
  separate `mm_HI=True` **dense O(N²)** regime (SLEPc MFN sqrt, `src/slip_vel.cpp:27-49`) if
  that is ever run at scale — that path wants distributed/GPU dense BLAS specifically.

## GPU: a different design, not a different annotation

The CPU stages above parallelize the existing loops *in place* — you keep the data layout
and add `#pragma omp` / distribute PETSc objects. A GPU port is a different exercise: you
**invert ownership of the state**. The device holds positions, forces, `Mcm_block`, the
noise, and the solve vectors resident across timesteps, and the host becomes an
orchestrator that only touches data at sample points. Everything else follows from that.

### The dominant constraint: data residency across ~500k tiny steps
`dt=1e-5`, `tmax=5.0` → ~500k steps, each O(Nm) and individually cheap. On CPU you never
think about where data lives. On GPU the killer is **host↔device transfer per step**: a
single round-trip of the nm3 position/force arrays each step would dwarf the compute. So
the *entire* per-step pipeline — `check_dist`, WCA forces, `mob()` assembly, the matvec,
noise, integration — must run on-device, with a copy to host **only every `samplerate`
steps** (currently 100) for output. This single requirement is what makes it a rewrite of
the inner loop's data model rather than an incremental change.

### What maps *well* to a GPU (and better than to OpenMP)
- **`mobility_AB` pair assembly** (`src/mob.cpp:362-537`): O(Nm) independent per-pair tensor
  evaluations with high arithmetic intensity — a textbook CUDA kernel. Each thread computes
  one monomer–colloid pair and writes its disjoint columns of `Mcm_block` in device memory.
  This is a *better* fit than OpenMP because the arithmetic-per-pair keeps the ALUs busy.
- **WCA cell-list forces** (`src/cell_list.cpp`, `src/rhs.cpp`): GPU neighbor lists are a
  solved problem (HOOMD-blue, LAMMPS `KOKKOS`/`GPU` packages do exactly this). Rebuild the
  cell list on-device each step; force kernel is one thread per monomer accumulating over
  its neighbors (no atomics if you gather per-particle).
- **Brownian noise**: replace the host `std::mt19937_64` (`src/Stokes.h:56`) with **cuRAND**
  generating the nm3 Gaussian fills directly into device buffers. Natural, well-supported.
- **Integration / elementwise O(N) work**: the predictor/corrector position updates and the
  RFD difference loops (`src/drift.cpp:32-57`, `src/run.cpp:107-124`) are trivial device
  kernels or Thrust transforms.

### What's *awkward* on a GPU (and why)
- **`long double` tensors.** The mobility tensors are `boost::multi_array<long double,…>`
  (`src/multi_arrays.h:6-12`). GPUs have no 80-bit float — this must drop to `double`.
  Check whether the RPY overlap regularization (the whole reason for this branch, see the
  Zuk-et-al work) still behaves in double before committing. **This is the first thing to
  validate.**
- **The arrowhead matvec is skinny and memory-bound.** With one colloid, `Mcm_block` is
  11 × 3·10⁵ — a GEMV, not a GEMM. GPUs run it fine but it's bandwidth-limited, not the
  compute win people expect from a GPU. The *assembly*, not the matvec, is the GPU-friendly
  part here.
- **GMRES kernel-launch overhead.** ~4 solves/step × 500k steps × (iterations) small
  kernels (matvec + Arnoldi dot-products/reductions). Launch latency becomes visible; expect
  to need **CUDA Graphs** or fused kernels to amortize it. The Arnoldi orthogonalization has
  unavoidable small host-side scalar traffic.
- **The colloid Schur solve stays tiny and serial** (11×11 syev, `src/slip_vel.cpp:81-306`)
  — leave it on the host or a one-block device kernel; negligible either way.

### How it changes the *strategy* vs. the CPU cluster plan
- **One GPU may replace the whole MPI effort for this size.** Nm~10⁵ with a matrix-free
  O(Nm) operator fits comfortably on a single modern GPU (A100/H100) in both memory and
  throughput. So the "multi-node cluster" goal could collapse to **one GPU per run**, making
  Stage 2's distributed-PETSc refactor unnecessary at this problem size. Multi-GPU only
  becomes relevant for much larger Nm (then: the same particle decomposition as Stage 2, but
  with GPU-aware MPI and device-resident halos).
- **The `mm_HI=True` dense regime is where a GPU wins asymptotically.** That path is dense
  O(N²)/O(N³) — the grand-mobility GMRES matvec and the SLEPc MFN matrix-square-root
  (`src/slip_vel.cpp:27-49`). Dense BLAS-2/3 and the MFN Krylov iterations are exactly what
  GPUs (cuBLAS, cuSOLVER/MAGMA, SLEPc's GPU backend) accelerate most. If you ever run monomer
  HI at scale, GPU is the highest-leverage option, more so than for the O(Nm) arrowhead.

### Tooling — two levels of effort
- **Offload-lite (low effort, likely *not* worth it here):** configure PETSc `--with-cuda`,
  set `-vec_type cuda -mat_type aijcusparse`, keep assembly on the host and only run the
  solve on-device. For the O(Nm) arrowhead this is **transfer-bound** (you copy `Mcm_block`
  and vectors every step) and probably a net loss. It *is* worth trying for the dense
  `mm_HI=True` path, where the solve dominates and transfers amortize.
- **Device-resident (the real port):** custom CUDA kernels for `mobility_AB` assembly and
  WCA forces writing into device buffers; a `MatShell` whose `ArrowheadMult` operates on
  `VecCUDAGetArray` device pointers; cuRAND for noise; state never leaves the GPU between
  samples. This is the version that actually pays off — and it's a larger, different effort
  than either OpenMP (Stage 1) or MPI (Stage 2), not a superset of them.

### Verification differences
GPU reductions are non-deterministic in summation order, and the forced `long double→double`
drop changes results at the ULP level — so **bitwise** agreement with the serial CPU run is
off the table. Validate **statistically** with the existing `bond_statistics` /
`trap_statistics` oracles (trap Boltzmann distribution, zero bond drift), and watch the
`mm_HI=False` positive-definiteness / negative-eigenvalue diagnostics that motivated the
double-precision concern above.

## Orthogonal algorithmic lever (big payoff, independent of the above)
`drift()` **doubles** the per-step `mob()`+solve cost with two RFD probes
(`src/drift.cpp:32-54`). Two cheap wins that cut serial *and* parallel cost identically:
(1) warm-start every saddle solve from the previous solution — the `run.cpp:52` TODO
"re-use last solution as initial guess" is unimplemented for the deterministic/predictor
solves; (2) reuse the predictor-stage `mob()` inside the RFD probe where the geometry
barely moves. Worth doing before/alongside Stage 1.

## Critical files
- `src/matfree_A.cpp` — arrowhead shell + work vectors (Stage 1 BLAS, Stage 2 distribute).
- `src/mob.cpp:362-537, 975-1002` — thread-safe `mobility_AB` + AB assembly loop (Stage 1).
- `src/cell_list.cpp`, `src/check_dist.cpp`, `src/rhs.cpp` — O(Nm) force/neighbor loops.
- `src/drift.cpp`, `src/run.cpp:52,86-124` — RFD + predictor/corrector (warm-start lever).
- `src/init.cpp:26-347`, `src/arrays.cpp` — solver/comm setup + global storage to distribute.
- `CMakeLists.txt` — re-enable `find_package(OpenMP)` / `MPI` (both currently commented out).

## Verification
- **Stage 0**: `-log_view` table at production Nm; commit as the baseline.
- **Correctness (every stage)**: a threaded/MPI run must reproduce the serial run's
  physics. Use the existing statistical checks in `test_sim_dumbbells_thermal.py`
  (`bond_statistics`, `trap_statistics`) — the trapped colloid must still sample the
  harmonic-trap Boltzmann distribution and bond displacements must not drift. Run a small
  Nm case serial vs. 1-rank vs. N-rank and confirm matching trajectories for a fixed seed
  (bitwise where RNG allows, statistically otherwise).
- **Scaling**: strong-scaling sweep (fixed Nm; cores/ranks = 1,2,4,8,…) plotting step/s and
  parallel efficiency. Weak-scaling (Nm ∝ ranks) for Stage 2.
- **Regression**: keep the serial build working — guard OpenMP/MPI paths so a 1-thread/
  1-rank build matches today's numbers.

## Notes / non-goals
- The `arrays::` global namespace and the per-`run()` `SlepcInitialize` with no matching
  finalize (`src/cleanup.cpp:39-41`) mean **two sims cannot share one process** — but that
  only blocks the *ensemble* axis (separate OS processes / job arrays), which is out of
  scope for "speed up one big run" and needs no code change anyway.
- `src/saddle.cpp` is legacy/dead (references a matrix `M` that no longer exists); ignore.
