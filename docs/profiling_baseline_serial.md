# Stage-0 Profiling Baseline (serial)

Baseline for the OpenMP branch, per `docs/parallelization_roadmap.md` Stage 0. This is the
speedup yardstick every later change is measured against.

## Setup
- Case: `mm_HI=False`, single trapped colloid, **5000 dumbbells (Nm = 10,000 monomers)**,
  `dt=1e-5`, `box=[10,10,10]`, `kT=1`, `mono_ev=False`, `kbond=1`, `beta=0.001`.
- **300 timesteps**, no trajectory I/O (`samplerate` set huge).
- `OMP_NUM_THREADS=1`, serial single-rank.
- Driver: `scratch_profile.py 5000 300` with `POLYSTOKES_LOGVIEW=1`.
- Build: PETSc 3.22.4, `--download-f2cblaslapack` (single-threaded reference BLAS),
  `-O3 -march=native`, gcc.
- Instrumentation: PETSc log events `mob`/`check_dist`/`RHS`/`drift` (RAII `PetscEventScope`)
  + PETSc's automatic KSP/Mat/Vec events. `drift` is *inclusive* (counts the mob/RHS/solves
  it calls); `mob`/`check_dist`/`RHS` are totals over all 4 calls/step.

## Result: the GMRES solve dominates — not the assembly loops

**Total wall time: 46.37 s / 300 steps ≈ 0.155 s/step.**

| Event | Calls | Time (s) | % wall | Notes |
|---|---|---|---|---|
| **KSPSolve** | 1500 | **41.97** | **91%** | ~5 saddle solves/step; **the bottleneck** |
| &nbsp;&nbsp;└ KSPGMRESOrthog | 28642 | 21.65 | 47% | Gram–Schmidt: VecMDot + VecMAXPY |
| &nbsp;&nbsp;└ MatMult (arrowhead) | 29244 | 16.00 | 35% | dense `Mcm_block` gemv (BLAS-2) |
| &nbsp;&nbsp;└ VecMDot | 28642 | 17.36 | 37% | GMRES dot products (PETSc kernel) |
| &nbsp;&nbsp;└ VecMAXPY | 29243 | 4.70 | 10% | GMRES axpys |
| &nbsp;&nbsp;└ VecNormalize/VecNorm | ~30k | 4.07 | 9% | |
| drift (inclusive) | 300 | 14.70 | 32% | its 2 probe solves + 2 mob dominate it |
| **mob** (assembly) | 1200 | **2.11** | **5%** | the "already thread-ready" AB loop |
| **check_dist** | 1200 | **0.23** | **1%** | pairwise separations |
| **RHS** | 1200 | **0.13** | **0.3%** | bond/trap/WCA forces |

GMRES runs **~19.5 iterations per solve** (29244 matvecs / 1500 solves) under the Jacobi
preconditioner — orthogonalization cost grows ~quadratically with that count.

## Implications for Stage 1 (this reorders the roadmap's priorities)

The user's premise — that force/pair-distance loops are the cost — is **not** what dominates.
The plain-C++ O(Nm) assembly the roadmap targets for `#pragma omp` (`mob` + `check_dist` +
`RHS`) is only **~6.5%** of wall time. By Amdahl, threading it perfectly caps at ~1.07× total.

The real cost is **inside `KSPSolve` (91%)**: dense `MatMult` (35%, BLAS gemv) and the GMRES
vector kernels `VecMDot`/`VecMAXPY` (~57% combined). So the high-leverage levers are:

1. **Cut GMRES iteration count** — warm-start the deterministic solves (Lever 1). Fewer
   iterations reduces MatMult *and* the ~quadratic orthogonalization directly, serial or not.
2. **Thread the solve** — relink PETSc against threaded OpenBLAS (Lever 2): the dense
   `MatMult` gemv (35%) threads via BLAS. The `Vec*` GMRES kernels (~57%) are PETSc's own
   loops and need PETSc's OpenMP/threaded backend (or MPI) — a heavier item; note this is the
   part shared-memory OpenMP annotations alone cannot reach.
3. **OpenMP the assembly loops** (roadmap Stage 1a–1d) — correct and low-risk, but ~6.5%
   ceiling; do it for completeness and larger-Nm headroom, not as the primary win.

## Lever 1 result: warm-starting the solves does NOT help (reverted)

Warm-starting the saddle solves from the previous solution was measured at 5000 dumbbells /
300 steps and **increased** cost in every configuration:

| Config | matvecs/solve | total (s) |
|---|---|---|
| Cold (baseline) | **19.50** | **46.4** |
| Warm: deterministic + stochastic solves | 24.38 | 67.4 |
| Warm + `KSPConvergedDefaultSetUMIRNorm` | 26.84 | 75.2 |
| Warm: deterministic solves only | 23.69 | 65.9 |

Why: PETSc's default convergence test already measures `rtol` against `‖b‖` for a nonzero
initial guess (`KSPConvergedDefault`, `rnorm0 = snorm`), so warm-starting does *not*
over-tighten — yet the previous solution is still **not a better guess than zero** here: the
operator `A` is reassembled every step, `sync_mcc_schur_correction()` re-applies an adaptive
eigenvalue floor, and the saddle constraint block couples tightly, so `‖b - A·x_prev‖`
exceeds `‖b‖` and GMRES needs *more* iterations. `UMIRNorm` makes it worse still because it
switches the reference to `min(‖b‖,‖r₀‖)` = `‖r₀‖` for a warm start, over-tightening.
Conclusion: keep all solves cold. The lever for the 91% is the **iteration count itself**
(preconditioning) and/or **threading the Vec/Mat kernels** (BLAS/MPI).

## Lever result: GMRES → MINRES is a ~29% win (adopted)

The saddle operator `A = [[Mob, B],[Bᵀ, 0]]` is **symmetric indefinite** (`Mob` symmetric with
`M^mc = (M^cm)ᵀ` and symmetric `M^cc`; constraint block transpose-coupled). GMRES pays a full
orthogonalization every iteration (the ~47% `KSPGMRESOrthog` / `VecMDot`). Switching to
**MINRES** — a short 3-term Lanczos recurrence, O(1) work/iteration — removes that entirely at
the *same* iteration count. Clean back-to-back A/B (5000 dumbbells, 300 steps):

| Solver | total (s) | KSPSolve (s) | MatMult (s) | Orthog (s) |
|---|---|---|---|---|
| GMRES (old) | 53.6 | 49.0 | 19.0 | **25.1** |
| **MINRES (new default)** | **38.0** | **33.4** | 19.8 | **0.0** |

**~29% faster overall**, KSPSolve −32%. Validated: with an identical initial config + the same
Brownian seed, MINRES tracks GMRES to solver tolerance (per-frame max diff 0 → ~2e-4 over 200
steps — chaotic amplification of the ~1e-6 solve tolerance, not a physics difference). Adopted
in `init.cpp` (`KSPSetType(ksp, KSPMINRES)`); override with `-ksp_type gmres` if ever needed.

**Consequence for the remaining plan:** MINRES eliminates the GMRES `Vec` orthogonalization
kernels — the very ones that "MPI to thread the Vec kernels" targeted. The remaining solve cost
is now the dense arrowhead **`MatMult`** (~59% of KSPSolve, a BLAS-2 gemv on `Mcm_block`), which
threaded **OpenBLAS** (Lever 2) accelerates directly. So MPI is no longer the obvious next step;
the OpenBLAS relink is.

## Lever 2 result: OpenBLAS helps (serial), but threading the arrowhead does NOT

PETSc was rebuilt against threaded OpenBLAS in a **separate, non-destructive arch**
`arch-linux-openblas` (the `f2cblas` `arch-linux-c-opt` is untouched as a fallback;
`petsc/reconfigure-openblas.py`). All numbers below are MINRES, 5000 dumbbells, 300 steps,
Release (`-O3`):

| Config | total (s) | KSPSolve (s) | MatMult (s) | mob (s) |
|---|---|---|---|---|
| f2cblas, 1 thread | 36.0 | 31.4 | 18.6 | 2.13 |
| **OpenBLAS, 1 thread** | **23.6** | **19.9** | **13.6** | 2.06 |
| f2cblas, `mob` at 8 OMP threads | 42.1 | 30.4 | 18.1 | **9.23** |
| OpenBLAS, 8 BLAS threads | **> 115 (timeout)** | — | — | — |

- **OpenBLAS single-threaded is ~34% faster than f2cblas** (23.6 vs 36.0 s) — its optimized
  `dgemv` beats the reference C-translated BLAS even at one thread. **This is the win.**
- **Threading is a net loss for this O(Nm) arrowhead.** (a) The `mob` assembly is only ~5% of
  runtime and an OpenMP `parallel for` ran ~4× *slower* at 8 threads (per-thread
  `boost::multi_array` allocation contention across the ~4 mob calls/step) — so the `mob`
  threading was **reverted**. (b) Threading the dense `MatMult` is worse: it is a *skinny*
  11 × 3·10⁵ gemv called ~30k times; an OpenBLAS thread sweep gave ~nothing at 2–4 threads and
  blew up at 8 (per-call dispatch overhead ≫ the tiny work). This matches the roadmap's caveat
  that the arrowhead matvec is memory-bound, not a threading target.

## Combined result and recommended config

**GMRES + f2cblas (original) 53.6 s → MINRES + OpenBLAS (1 thread) 23.6 s: ~56% faster**, all
serial. Recommended production stack for the `mm_HI=False` arrowhead regime:

- **MINRES** (in `init.cpp`; symmetric saddle system).
- **OpenBLAS** PETSc/SLEPc arch, run with **`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`**
  (threading the skinny gemv hurts). To build/run against it, point at the arch and keep the
  `f2cblas` dir off `LD_LIBRARY_PATH`:
  ```bash
  export PKG_CONFIG_PATH=/home/takalab/Software/petsc/arch-linux-openblas/lib/pkgconfig:\
  /home/takalab/Software/slepc/arch-linux-openblas/lib/pkgconfig
  export LD_LIBRARY_PATH=/home/takalab/Software/petsc/arch-linux-openblas/lib:\
  /home/takalab/Software/slepc/arch-linux-openblas/lib   # NOT the arch-linux-c-opt dirs
  pip install .    # or: cmake --build build_openblas
  ```
- **Not adopted:** warm-start (net loss), OpenMP threading of the assembly/solve (net loss for
  this problem size). The `find_package(OpenMP)` hook remains in CMake, harmless and unused,
  for future larger-Nm experiments.

Validation: MINRES tracks GMRES to solver tolerance (identical ICs) and the bond/trap Boltzmann
oracles are statistically identical between the two (e.g. trap σ-ratio 2.91 vs 2.87 — the offset
is the known coarse-`dt` integrator heating, present in both, not a solver effect).

## Note: latent clean-build bug fixed
`arrays::Pinv` was declared `extern` but its definition was commented out; the old `build/`
only linked via a stale pre-comment object. Any clean build (`pip install .`) would fail with
`undefined symbol: arrays::Pinv`. Restored the definition in `src/arrays.cpp` (the referencing
`initialize_Pinv()` is otherwise-unused scaffolding).

## Reproduce
```bash
cd build && cmake --build . -j8
OMP_NUM_THREADS=1 POLYSTOKES_LOGVIEW=1 PYTHONPATH=build python scratch_profile.py 5000 300
```
