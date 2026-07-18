# Parallelization Roadmap for PolyStokes

*Goal: speed up a single large simulation. Target: multi-node cluster.*

> **Status (updated after `feature/openmp`).** Stage 0 (profiling) and Stage 1 are **done**;
> see [`profiling_baseline_serial.md`](profiling_baseline_serial.md). Key finding: the linear
> **solve is ~91%** of runtime, not the O(Nm) assembly (~6.5%), so shared-memory OpenMP is a
> **dead end** here (assembly threading measured a *net loss*). The realized serial wins were
> **algorithmic + library: GMRES→MINRES (~29%) and f2cblas→OpenBLAS (~34%), ~56% combined.**
> With single-node exhausted, **Stage 2 (MPI) is the active work** for scaling to ~100K
> dumbbells. The solver is now **MINRES** (symmetric-indefinite saddle), not GMRES.

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
- **MINRES+Jacobi solve** (`src/init.cpp:51,58`): matvec is `ArrowheadMult`
  (`src/matfree_A.cpp:38-80`), two dense `Mcm_block` matvecs O(Nc·Nm). **Stage-0 profiling put
  this at ~91% of wall time** — its Krylov `Vec` reductions and dense `MatMult` are what MPI
  distributes (they do not thread within a node); the assembly below is only ~6.5%.
- The Brownian slip-velocity and colloid Schur eigendecomposition are **negligible** here
  (nc11 = 11; dense syev on an 11×11, `src/slip_vel.cpp:81-306`).

## Recommended approach — staged, lowest-risk-first

### Stage 0 — Profiling baseline *(prerequisite, ~0.5 day)* — ✅ **DONE**
`PetscLogView` is gated behind `POLYSTOKES_LOGVIEW=1` (`src/run.cpp`) with per-function log
events (`mob`/`check_dist`/`RHS`/`drift`). Result committed to
[`docs/profiling_baseline_serial.md`](profiling_baseline_serial.md): **KSPSolve ~91%**, `mob`
~5%, `check_dist` ~1%, `RHS` ~0.3% — which redirected the whole effort to the solve.

### Stage 1 — OpenMP within a node — ✅ **DONE, but a net loss → not adopted**
Evaluated and **reverted**. Profiling (Stage 0) showed the O(Nm) assembly is only ~6.5% of
runtime, so Amdahl caps OpenMP at ~1.07×; worse, threading the `mob()` AB loop measured **~4×
slower at 8 threads** (per-thread `boost::multi_array` allocation contention), and the skinny
`Mcm_block` gemv does not benefit from BLAS threads either. **The realized single-node wins
were instead algorithmic + library:** GMRES→**MINRES** (~29%, `src/init.cpp`; the saddle
operator is symmetric-indefinite so the short 3-term recurrence removes GMRES's ~47%
orthogonalization) and f2cblas→**OpenBLAS** (~34% serial). Run single-threaded
(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`). Details in
[`profiling_baseline_serial.md`](profiling_baseline_serial.md). The original OpenMP sketch is
retained below for reference / the `find_package(OpenMP)` hook remains for future large-Nm work.

<details><summary>Original Stage-1 OpenMP sketch (not adopted)</summary>

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

</details>

### Stage 2 — MPI particle decomposition via PETSc *(multi-node scaling, ~3–5 weeks)* — **the current target**
With single-node exhausted (Stage 1 above), reaching ~100K dumbbells (Nm ≈ 2·10⁵) requires
**distributed memory**. Today the code runs single-rank: all data `Mat`/`Vec` are
`PETSC_COMM_SELF`/`VecCreateSeq`, so `mpiexec -n N` just replicates work. The arrowhead
structure makes distribution clean — the monomer self-block is diagonal and the colloid
block is tiny.

**Scope: production regime first.** `mm_HI=False`, `mono_ev=False`, `tether=False`, single
colloid — the stated 100K-dumbbell workload. **Partition monomers by dumbbell** (contiguous
index blocks, both beads of a dumbbell on the same rank). Then: intra-dumbbell **bonds stay
on-rank**, there is **no monomer–monomer WCA**, and the *only* cross-rank coupling is the
single central colloid → a **halo-free** decomposition (broadcast the colloid + one small
`MPI_Allreduce`). The general case (`mono_ev=True` WCA, tethers, multi-colloid) needs a
spatial decomposition with a ghost halo and is deferred to sub-stage **2d**.

**DOF layout: keep the three saddle blocks separate (MatNest/VecNest)** — no global reindex.
The saddle vector is `x = [u_m (nm3) | u_c (nc11=11) | l (nm3nc6)]` (`src/matfree_A.cpp:38-102`);
only `nm3`, `nm3nc6`, `nm3nc11`, `nm6nc17` scale with Nm (`src/Stokes.cpp:74-83`). Each block
keeps monomer-natural indexing; `u_m` and `l` distribute over the monomer partition, `u_c`
replicates on every rank.

Sub-stages, **each preserving `1-rank == today's serial run bitwise`**:

**2a — MPI bring-up (plumbing only).** ✅ *implemented.* Query `mpi_rank`/`mpi_size` from
`PETSC_COMM_WORLD` (`src/init.cpp`; use **PETSc's own MPI** via `petscsys.h` — do **not**
`find_package(MPI)`, which could pull the system OpenMPI on `LD_LIBRARY_PATH` and clash with
PETSc's `--download-mpich`), guard output to rank 0 (`src/run.cpp`), keep the **same** RNG seed
on all ranks for now (per-rank streams move to 2c, once noise draws are local — otherwise
replicated ranks diverge). **Guarantee: 1-rank == serial bitwise** (verified via
`scratch_mpi_check.py`). *Note:* there is **no "replicated N-rank" middle ground** — PETSc's
`global = Σ local` model means the existing `PETSC_DECIDE` world vectors (`rhs,X,…`) split
across ranks while the replicated `SELF` operator does not, so `mpiexec -n >1` aborts
(`PCApply: local rows ≠ vector size`) until 2b actually distributes the operator. N-rank
correctness therefore lands with 2b, not here.

**2b — Distribute the operator + solve (the core).** ✅ *implemented & verified*
(`src/matfree_A_mpi.cpp`; commits "Stage 2b m1/m2/m3"). Realized with a per-rank-contiguous
distributed vector `[u_m_loc | l_m_loc | (rank0: u_c, l_c)]` rather than a formal VecNest —
`solve_saddle()` dispatches to `solve_saddle_distributed()` when `mpi_size>1` (persistent
monomer-partitioned MINRES; colloid coupling = one `MPI_Allreduce` over nc11 + a broadcast per
matvec; `B=[-I;0]` is a trivial local injection). Verified against serial: distributed matvec
exact (~1e-15), distributed solve identical velocity-norm across 1/2/4 ranks, and the **full
timestep loop** matches serial trajectories to the MINRES tolerance (~2e-6 over 100 steps) on
2/4 ranks. Assembly is still replicated per rank (the bridge) — the per-step vectors are on
`arrays::rep_comm` (= `PETSC_COMM_SELF` when `mpi_size>1`), so each rank assembles redundantly
and only the solve is distributed. The solve (the ~91%) now scales; 2c removes the assembly bridge.

<details><summary>Original 2b design sketch (VecNest formulation)</summary>

Represent the state as a **VecNest** of
`u_m` (nm3, MPI, monomer partition), `u_c` (nc11, replicated), `l` (nm3nc6, MPI). Rework
`ArrowheadMult`/`ArrowheadGetDiagonal` (`src/matfree_A.cpp:38-102`) to act on the sub-vecs
instead of manual index slices:
  - `M^mm = beta_inv·I` on `u_m` — embarrassingly parallel, local.
  - `M^cm·u_m` → colloid result via each rank's local `Mcm_block` columns + an **`MPI_Allreduce`**
    over nc11 scalars; `M^mc·u_c`, `M^cc·u_c` computed locally (`u_c` replicated, cheap).
  - `B·l`, `B^T·u` with `B` as **MPIAIJ**, rows partitioned to match the monomer blocks
    (`src/arrays.cpp:350-370`).
  - `Mcm_block` → **MPIDENSE** with monomer *columns* local (`src/arrays.cpp:329`);
    `Mcc_block`/`Smat` stay replicated SELF. Switch the shell/`A` `MatSetSizes` to `PETSC_DECIDE`.
  **MINRES + Jacobi** already run on `PETSC_COMM_WORLD` (`src/init.cpp:51,58`) and parallelize
  transparently once operands are distributed — the Krylov reductions become `MPI_Allreduce`.
  *Milestone:* distributed solve matches the serial solve for a fixed RHS on 1/2/4 ranks (to tol).

</details>

**2c — Distribute assembly, forces, enumeration** *(compute ✅ done; memory/RNG remain).*
The O(Nm) **compute** is now distributed and verified (serial byte-identical; N-rank matches
serial trajectories to the MINRES tolerance on 1/2/4 ranks):
- **`mob()`** fills only local `Mcm` columns; the Schur `M^cm(M^cm)^T` and slip-vel `M^cm·xi_m`
  become local partials + `MPI_Allreduce` (nc11-sized). *(2c-1)*
- **`check_dist()`** computes pd/vlist only for local monomer rows; `pair_interaction()` WCA is
  then local via the Verlet list, and `RHS()` reduces the partial colloid force before adding
  the replicated trap. *(2c-2)*
- `bond_forces` stays replicated (writes only monomer slots — correct per-rank-local).

**Remaining 2c (optional polish, not compute-critical):** `Mcm_block` still allocates the full
`nc11 × nm3` per rank (only local columns filled) — switching to `nc11 × nm3_local` caps
per-rank memory for *very* large Nm (touches the Schur/slip-vel matrix dims + the solver's
column read). Per-rank RNG streams (then validation moves from trajectory-match to the
`bond`/`trap` statistical oracles). `set_vars()` still builds the full replicated lists.

<details><summary>Original 2c sketch</summary>

`set_vars()` builds only the **local
dumbbells'** pair/bond/chain lists (`src/init.cpp:202-324`); the `mob()` AB loop fills only local
`Mcm_block` columns (`src/mob.cpp:983-1000`); `RHS()`/`bond_forces`/`trapping_forces` assemble the
**local** force block (bonds intra-dumbbell → on-rank; trapping on the colloid-owning rank,
`src/rhs.cpp:44-210`). **Broadcast** the colloid position/velocity each step (after `step()`) and
**allreduce** any colloid force contribution. Draw the slip/drift noise **locally** per rank; the
colloid Schur `syev` + slip stays replicated with the `M^cm·xi_m` allreduce from 2b (`src/slip_vel.cpp`).

</details>

**2d — General case (later, out of primary scope).** `mono_ev=True` monomer–monomer WCA needs a
**spatial** decomposition with a per-step **ghost-monomer halo exchange** + **reverse force
scatter** for the Newton's-3rd-law writes in `monomer_wca` (`src/cell_list.cpp:98-172`), plus
tether-to-colloid bonds and multi-colloid coupling. Note it; do not build it first.

**Effort:** 2a ~2–3 days, 2b ~2 weeks (the crux), 2c ~1–2 weeks, 2d separate.
The global `arrays::` namespace is fine for MPI (each rank is its own process) — but its
contents must become *distributed* storage, which is the bulk of 2b–2c.

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

## Orthogonal algorithmic lever
`drift()` **doubles** the per-step `mob()`+solve cost with two RFD probes
(`src/drift.cpp:32-54`).
- **Warm-start the saddle solves — tried, net loss, reverted.** Reusing the previous solution
  as the initial guess *increased* MINRES iterations (~19.5 → ~24 matvecs/solve): the operator
  is reassembled every step and the adaptive Schur floor shifts it, so the prior solution is no
  better than a zero guess and only adds iterations. Do not pursue. (Details in
  [`profiling_baseline_serial.md`](profiling_baseline_serial.md).)
- **Still open:** reuse the predictor-stage `mob()` inside the RFD probe where the geometry
  barely moves — cuts a mobility assembly per step, serial *and* parallel. Not yet done.

## Critical files
- `src/matfree_A.cpp` — arrowhead shell + work vectors (Stage 1 BLAS, Stage 2 distribute).
- `src/mob.cpp:362-537, 975-1002` — thread-safe `mobility_AB` + AB assembly loop (Stage 1).
- `src/cell_list.cpp`, `src/check_dist.cpp`, `src/rhs.cpp` — O(Nm) force/neighbor loops.
- `src/drift.cpp`, `src/run.cpp:52,86-124` — RFD + predictor/corrector (warm-start lever).
- `src/init.cpp:26-347`, `src/arrays.cpp` — solver/comm setup + global storage to distribute.
- `CMakeLists.txt` — `find_package(OpenMP)` is now in (optional/unused); **re-enable `MPI`**
  (still commented out) for Stage 2.

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
