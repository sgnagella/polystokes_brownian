# PolyStokes — Installation (verified on WSL2 / Ubuntu 24.04)

This document records a complete, working installation of PolyStokes from
scratch, including the PETSc build it depends on. It reflects exactly what was
done on this machine; adjust paths and versions to match your system.

## Environment used

| Component   | Value                                              |
|-------------|----------------------------------------------------|
| OS          | Ubuntu 24.04 on WSL2 (Linux 6.18 kernel)           |
| Compiler    | gcc / g++ 12.4.0 (system, `/usr/bin`)              |
| Fortran     | **none** — `gfortran` not installed                |
| CMake       | 4.1.4                                              |
| pkg-config  | 0.29.2 (installed via conda-forge)                 |
| Boost       | 1.83 headers (system, `/usr/include/boost`)        |
| Python      | 3.11.15 in a dedicated conda env `polystokes`      |
| PETSc       | 3.25.2 (release), at `/home/snagella/Software/petsc` |
| SLEPc       | 3.25.1, at `/home/snagella/Software/slepc`         |

---

## 1. Install PETSc

PETSc source lives at `/home/snagella/Software/petsc` (the `release` branch,
version 3.25.2).

### 1.1 Configure

There is **no Fortran compiler** on this machine, so PETSc is configured with
Fortran disabled. Because the reference Netlib BLAS/LAPACK requires Fortran,
PETSc downloads the C-translated `f2cblaslapack` instead. No system MPI was
present, so MPICH is downloaded and built by PETSc.

```bash
cd /home/snagella/Software/petsc
./configure PETSC_ARCH=arch-linux-c-debug \
  --with-cc=gcc \
  --with-cxx=g++ \
  --with-fc=0 \
  --download-f2cblaslapack \
  --download-mpich
```

Configuration option rationale:

| Option                      | Why                                                              |
|-----------------------------|------------------------------------------------------------------|
| `--with-fc=0`               | No Fortran compiler; build without Fortran support.              |
| `--download-f2cblaslapack`  | No system BLAS/LAPACK; reference Netlib build needs Fortran, so use the C-translated version. |
| `--download-mpich`          | No system MPI found; build MPICH (v5.0.0) for parallel support.  |
| `PETSC_ARCH=arch-linux-c-debug` | Debug build (`-O0 -g3`). See the note at the end about an optimized arch. |

> Note: `--download-f2cblaslapack` is fetched from a remote server and can hit a
> transient network timeout. If the download fails, simply re-run the identical
> `./configure` command — already-built dependencies (e.g. MPICH) are cached and
> are not rebuilt.

This produces an in-place (non-prefix) build with:
- Scalar type: real, double precision
- 4-byte integers
- Shared libraries: yes
- Single library: yes

### 1.2 Build and verify

```bash
make PETSC_DIR=/home/snagella/Software/petsc PETSC_ARCH=arch-linux-c-debug all
make PETSC_DIR=/home/snagella/Software/petsc PETSC_ARCH=arch-linux-c-debug check
```

A successful `check` runs `snes/tutorials/ex19` on 1 and 2 MPI processes.

After building, PETSc exposes a `pkg-config` file at
`$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig/petsc.pc`, which the PolyStokes build uses
to locate PETSc.

---

## 1b. Install SLEPc

The Brownian-motion feature needs the matrix square root of the mobility, which
PolyStokes computes with **SLEPc** (the eigenproblem/`SVD`/matrix-function
companion to PETSc). `CMakeLists.txt` declares it as a hard dependency:

```cmake
pkg_check_modules(SLEPc REQUIRED IMPORTED_TARGET slepc)
```

so the build will not configure unless `pkg-config` can find SLEPc.

SLEPc is built *against the PETSc arch from section 1* — it reuses the same
`PETSC_DIR`/`PETSC_ARCH`, compilers, and MPICH. SLEPc source lives at
`/home/snagella/Software/slepc` (version 3.25.1, matching PETSc 3.25.x).

### 1b.1 Configure and build

```bash
export PETSC_DIR=/home/snagella/Software/petsc
export PETSC_ARCH=arch-linux-c-debug
export SLEPC_DIR=/home/snagella/Software/slepc

cd $SLEPC_DIR
./configure
make SLEPC_DIR=$SLEPC_DIR PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH
make SLEPC_DIR=$SLEPC_DIR PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check
```

`./configure` takes no extra options here — it picks up everything (BLAS/LAPACK,
MPICH, Fortran-disabled settings) from the already-configured PETSc via
`PETSC_DIR`/`PETSC_ARCH`.

After building, SLEPc exposes its own `pkg-config` file at
`$SLEPC_DIR/$PETSC_ARCH/lib/pkgconfig/slepc.pc`, which the PolyStokes build uses
to locate SLEPc.

> Note: SLEPc must be rebuilt for each PETSc arch you use. If you add an
> optimized PETSc arch (see the note at the end), re-run SLEPc's `configure`/
> `make` with that `PETSC_ARCH` before rebuilding PolyStokes against it.

---

## 2. Prepare the `polystokes` conda environment

PolyStokes is installed into a dedicated conda env. On this machine the env
existed but was empty, so Python and dependencies were installed into it.

```bash
# Make conda usable in a non-interactive shell
source /home/snagella/miniconda3/etc/profile.d/conda.sh

# Python 3.11 + runtime/build tools.
#   numpy, matplotlib  -> required by test_sim.py at runtime
#   pkg-config         -> required by CMake to find PETSc
conda install -y -n polystokes -c conda-forge \
  python=3.11 pip numpy matplotlib pkg-config

conda activate polystokes
```

### 2.1 Boost (build dependency)

The C++ sources include `boost/multi_array.hpp` (header-only Boost, used in
`src/multi_arrays.h` and `src/mob.cpp`). Boost is **not** declared in
`CMakeLists.txt`, so its headers must be discoverable on the compiler's default
include path. Installed system-wide:

```bash
sudo apt update && sudo apt install libboost-all-dev   # provides Boost 1.83 in /usr/include/boost
```

> The build links with the **system** `/usr/bin/g++`, which does not search the
> conda env's include directory automatically. Installing Boost system-wide
> (into `/usr/include`) is the simplest way to make the header discoverable.
> See "Portability note" below.

---

## 3. Build and install PolyStokes

PolyStokes uses `scikit-build-core` + CMake + pybind11. CMake finds PETSc and
SLEPc via `pkg-config`, so `PKG_CONFIG_PATH` must point at **both** the PETSc and
SLEPc arch directories built above.

```bash
source /home/snagella/miniconda3/etc/profile.d/conda.sh
conda activate polystokes

# Point pkg-config at the PETSc and SLEPc builds
export PETSC_DIR=/home/snagella/Software/petsc
export PETSC_ARCH=arch-linux-c-debug
export SLEPC_DIR=/home/snagella/Software/slepc
export PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$SLEPC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH

# Sanity check — should print 3.25.2 and 3.25.1
pkg-config --modversion petsc
pkg-config --modversion slepc

# Build and install (build isolation pulls scikit-build-core, pybind11,
# setuptools, wheel automatically from pyproject.toml)
cd /home/snagella/Projects/SD/polystokes_public_repo
pip install .
```

This compiles the C++ sources and installs the `PolyStokes` extension module
(`PolyStokes.cpython-311-x86_64-linux-gnu.so`) into the env's `site-packages`.

> Note: the `CMakeLists.txt` hardcodes `PETSC_DIR`/`PETSC_ARCH` to a cluster
> path, but those variables are unused — PETSc and SLEPc are located purely
> through `pkg-config` and `PKG_CONFIG_PATH`. Only `PKG_CONFIG_PATH` matters here.

---

## 4. Runtime environment

The extension links PETSc's and SLEPc's **shared** libraries, so
`LD_LIBRARY_PATH` must include **both** arch `lib` directories whenever
PolyStokes is imported. Otherwise `import PolyStokes` fails to find `libpetsc.so`
or `libslepc.so`.

```bash
conda activate polystokes
export PETSC_DIR=/home/snagella/Software/petsc
export PETSC_ARCH=arch-linux-c-debug
export SLEPC_DIR=/home/snagella/Software/slepc
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$SLEPC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
```

To make this automatic on every `conda activate polystokes`, create an
activation hook. Include `PKG_CONFIG_PATH` as well so the hook also covers
**rebuilds** — with it set, `pip install .` finds PETSc/SLEPc via `pkg-config`
without any manual exports (see section 3), so the same hook serves both running
and rebuilding:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/polystokes_env.sh" <<'EOF'
export PETSC_DIR=/home/snagella/Software/petsc
export PETSC_ARCH=arch-linux-c-debug
export SLEPC_DIR=/home/snagella/Software/slepc
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$SLEPC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$SLEPC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH
EOF
```

With this hook in place, both building and running reduce to:

```bash
conda activate polystokes
cd /home/snagella/Projects/SD/polystokes_brownian
pip install .          # rebuild — PETSc/SLEPc found via PKG_CONFIG_PATH
python test_sim.py     # run — shared libs found via LD_LIBRARY_PATH
```

> Note: the hook hardcodes `arch-linux-c-debug`. If you build the optimized arch
> (see the note at the end), update `PETSC_ARCH` in the hook — both the `lib` and
> `pkgconfig` paths derive from it, so that one line is the only change.

---

## 5. Verify with the test simulation

```bash
conda activate polystokes
export PETSC_DIR=/home/snagella/Software/petsc
export PETSC_ARCH=arch-linux-c-debug
export SLEPC_DIR=/home/snagella/Software/slepc
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$SLEPC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
export MPLBACKEND=Agg   # headless plotting (WSL has no display)

cd /home/snagella/Projects/SD/polystokes_public_repo
python test_sim.py
```

Expected: the solver initializes, integrates from `Time 0.001` to `Time 0.191`
(`tmax = 0.2`), and prints `Cleanup complete`. Outputs produced:

- `figures/initial_config.png` — initial configuration plot (`check_conf=True`)
- `data/beta_0.1/config/config_*.txt` — trajectory frames
- `params.pkl` — simulation parameters

---

## Notes and caveats

- **Debug vs optimized PETSc.** This build uses `arch-linux-c-debug`
  (`-O0 -g3`), good for development but slow. The project README assumes
  `arch-linux-c-opt`. For production runs, configure a second optimized arch
  and rebuild PolyStokes against it:
  ```bash
  cd /home/snagella/Software/petsc
  ./configure PETSC_ARCH=arch-linux-c-opt --with-cc=gcc --with-cxx=g++ \
    --with-fc=0 --download-f2cblaslapack --download-mpich --with-debugging=0
  make PETSC_ARCH=arch-linux-c-opt all
  # rebuild SLEPc against the optimized arch as well
  cd /home/snagella/Software/slepc
  PETSC_ARCH=arch-linux-c-opt ./configure
  make PETSC_ARCH=arch-linux-c-opt
  # then re-export PETSC_ARCH/PKG_CONFIG_PATH/LD_LIBRARY_PATH for arch-linux-c-opt
  # (covering both PETSc and SLEPc) and re-run `pip install .`
  ```

- **Portability note (Boost).** Boost is not declared in `CMakeLists.txt`; the
  build succeeds only because Boost headers are on the default system include
  path (`/usr/include`). On a machine where Boost lives elsewhere (e.g. an HPC
  cluster module), add a `find_package(Boost REQUIRED)` and a corresponding
  `target_include_directories`/`include_directories` entry, or pass the include
  path explicitly (e.g. `export CPATH=/path/to/boost/include`).
