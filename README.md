<div align="center">
  <img src="renderings/Winding_1_.gif" alt="Polymer winding simulation" width="600"/>
  <br/>
  <em>Two optically trapped colloidal probes winding polymer chains in a Stokesian dynamics simulation. Yellow lines highlight bond connections aomng monomeric beads and do not have physical volume. </em>
</div>

---

# PolyStokes

A C++ Stokesian dynamics simulator with Python bindings for modeling the hydrodynamic interactions of colloidal particles and flexible polymer chains at low Reynolds number.

**Physics:** Solves the grand mobility problem — including far-field many-body hydrodynamic interactions (HI) and near-field lubrication corrections — using a saddle-point formulation solved via [PETSc](https://petsc.org/). Polymer chains use FENE or harmonic spring models. Colloidal probes are driven by a moving harmonic trap.

## Features

- Full many-body hydrodynamic interactions
- FENE and harmonic spring bond models
- Configurable HI: monomer-monomer, inter-chain only, or none
- Moving optical trap with configurable on/off timing

## Requirements

- C++11 compiler (GCC or Clang)
- [CMake](https://cmake.org/) >= 3.15
- [PETSc](https://petsc.org/) (with `pkg-config` support)
- [pybind11](https://pybind11.readthedocs.io/) >= 2.10
- [scikit-build-core](https://scikit-build-core.readthedocs.io/) >= 0.5
- Python >= 3.8

## Installation

### 1. Install PETSc

Follow the [PETSc installation guide](https://petsc.org/release/install/). After building, set the environment variables:

```bash
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-linux-c-opt   # or your chosen arch
export PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH
```

### 2. Install Python dependencies

```bash
pip install pybind11 scikit-build-core setuptools wheel
```

### 3. Build and install PolyStokes

```bash
pip install .
```

This invokes CMake via `scikit-build-core`, compiles the C++ sources, and installs the `PolyStokes` Python extension module.

## Usage

Configure and run a simulation from Python:

```python
import numpy as np
import PolyStokes

sim = PolyStokes.PolyStokes(
    dt=0.001,
    samplerate=10,
    tmax=2.0,
    output_dir="data/run1",
    mm_HI=False,     # monomer-monomer hydrodynamic interactions
    chain_HI=True,   # inter-chain HI only
    fene=False,      # use harmonic springs
    record_forces=True,
)

sim.particle_info(Np, Nc, Nm, Npoly, Nmono_per_chain, beta, kbond, r0, Lmax, tau)
sim.trap_info(ktrap=500, tstart=0.0, trun=0.5)
sim.initial_configuration(conf)  # (Np, 3) numpy array of initial positions
sim.run()
```

See [`test_sim.py`](test_sim.py) for a complete example with 11 polymer chains and 2 trapped colloidal probes.

### Visualization

Convert output to GSD format for OVITO:

```bash
python make_gsd.py
```

## Project Structure

```
src/           C++ source and headers
test_sim.py    Example simulation script
make_gsd.py    Convert output trajectories to GSD format
```
