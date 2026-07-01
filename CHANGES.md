# Changes — Brownian motion feature work

This log records the changes made to get the Brownian-motion branch building and
running end-to-end (`test_sim.py` completing with the `kT > 0` path active).
Grouped by area; file/line references are approximate.

## Build system

- **`CMakeLists.txt`**
  - Added **SLEPc** as a hard dependency (`pkg_check_modules(SLEPc REQUIRED
    IMPORTED_TARGET slepc)`) and linked `PkgConfig::SLEPc`. SLEPc provides the
    matrix-function (`MFN`) solver used for the matrix square root of the
    mobility (needed for Brownian slip velocities).
  - Added the new sources `src/drift.cpp` and `src/slip_vel.cpp` to `SOURCES`
    (they existed but were never compiled).
- **`INSTALL.md`** — documented the SLEPc build/configure step, the extra
  `PKG_CONFIG_PATH`/`LD_LIBRARY_PATH` entries, and the optimized-arch rebuild.

## Headers / declarations

- **`Stokes.h`**
  - `#include <random>` (for `std::mt19937_64` / `std::normal_distribution`).
  - Fixed `void RHS(bool drift=false)` (was missing the parameter type).
  - Declared `init_random`, `sample_slip_vel`, `solve_slip_vel` (defined but
    never declared).
- **`params.h`** — added `double kT;` to `ParticleInfo` (`particle_info()`
  already assigned it).
- **`arrays.h` / `arrays.cpp`**
  - Added the global `Vec W` (declared/defined alongside `X`/`rhs`).
  - Uncommented the `initialize_M` declaration; renamed the definition
    `intiialize_W` → `initialize_W` (matched its declaration/caller).

## Initialization wiring

- **`init.cpp`**
  - `PetscInitialize` → **`SlepcInitialize`** (SLEPc's `FN`/`MFN` classes
    require it).
  - Added the missing `initialize_M(consts.nm3nc11)` call in `alok_arrays`
    (`M` was null → `MatSetValues(M,...)` aborted).
  - Called `init_square_root_solver()` on the `kT > 0` path (`mfn` was never
    created → `MFNSolve` failed).
  - `init_random` now seeds the **member** RNG (`rng.seed(seed)`) instead of
    shadowing it with locals.
  - Typo fixes: `FNUSER` → `FNSQRT`, `CHERRV` → `CHKERRV`, `err` → `ierr`,
    `inititalize_fb` → `initialize_fb`.
- **`run.cpp`** — `PetscFinalize` → **`SlepcFinalize`** in the signal handler.

## Mobility assembly (the non-SPD `M` bug)

- **`mob.cpp`** — the grand mobility `M` is assembled then copied into the
  top-left block of the saddle matrix `A` (`INSERT_VALUES`). Several mobility
  contributions were mistakenly written to `A` instead of `M`, so the `M→A`
  copy clobbered them and left `M` missing its self/AA blocks — making `M`
  non-positive-definite and the matrix square root (`FNSQRT`) fail with a
  negative eigenvalue. Rerouted all mobility writes to `M`:
  - `fill_self`: colloid self translation (`mob_a`), rotation (`mob_c`), and
    stresslet (`mob_m`) blocks → `M`.
  - `mob`: monomer–monomer (AA) pair `mob_a` terms → `M`.
  - The only remaining `A` write in `mob()` is the intended `M→A` block copy.

## RHS / drift / slip velocities

- **`rhs.cpp`**
  - `RHS(bool drift)` definition signature (was `RHS(drift=false)`).
  - Declared the local `PetscInt fdriftindices[nm3nc6]` and removed a bogus,
    out-of-bounds `ishift` read.
  - Fixed swapped `VecSetValues` arguments (indices vs. values order).
- **`drift.cpp`**
  - Added `using namespace arrays;`.
  - `check_dist():` → `check_dist();`; `RHS(drift=true)` → `RHS(true)`.
  - `udrift` → the existing `arrays::drift` array (qualified to avoid the
    `drift()` method-name clash).
  - Position-update loops bounded by `consts.n3` (not `nm3nc6`) — `x` is sized
    `n3`, so the old bound wrote out of bounds (segfault). Matches `traj.cpp`.
  - `memcpy` uses `consts.PetscScalarSize` (which is already the full
    `nm3nc6`-vector byte size, per `Stokes.cpp`), matching `vel.cpp`; added the
    balancing `VecRestoreArrayRead`.
- **`slip_vel.cpp`**
  - Copy the sampled `std::vector` `fb` into the `Vec W` (was a `fb`/`W`/
    `std_vec` mixup).
  - `consts.dt` → `timeinfo.dt` (and added the missing `;`).
  - `VecAXPY(W, scale, W)` → `VecScale(W, scale)`.
  - Replaced `SETERRQ` (returns a value, illegal in a `void` function) with a
    `PetscPrintf` + `return`.

## Tooling

- **`make_gsd.py`** — migrated the trajectory writer to the modern `gsd` API
  (tested with `gsd 5.0.1`): `gsd.hoomd.Snapshot()` → `gsd.hoomd.Frame()` and
  `open(..., mode='wb')` → `mode='w'`. (Downgrading `gsd` to a 2.x with the old
  API conflicts with the env's NumPy 2.x, so the script was updated instead.)

## Open items (physics, not mechanics — left for review)

- The `beta_inv` factor on the monomer self-mobility but not on colloid-self /
  pair terms — verify this is intended now that all blocks live in `M`.
- In the BB stresslet coupling, the `gt` 2–1 term flips sign while the `ht`
  2–1 term does not (`M` is symmetric either way) — a tensor-parity question.
- `test_sim.py` completing is a smoke test only; the Brownian drift/fluctuation
  statistics still need validation against theory.
