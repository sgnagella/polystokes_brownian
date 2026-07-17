#!/usr/bin/env python
"""Stage-2a MPI scaffold regression: a short deterministic sim (fixed IC seed) that writes
config frames. Run serial, -n1, -n2 and diff the output — they must be identical while data
is replicated. Usage: python scratch_mpi_check.py <outdir>
"""
import os, sys, shutil
import numpy as np
import PolyStokes
from test_sim_dumbbells_thermal import sample_initial_config

outdir = sys.argv[1]
N_dumbbell, nsteps, beta, dt = 100, 100, 0.001, 1e-5
box = [10, 10, 10]
r0 = 2.0 ** (1.0 / 6.0) * (2.0 * beta) + 0.03
Lmax, tau, kT, epsilon, ktrap, kbond = 1.5 * r0, 1000, 1.0, 5.0, 1.0, 1.0
mono_cut = 2.0 ** (1.0 / 6.0) * (2.0 * beta)
N_mono, N_trapped = 2, 1
N_mono_total = N_dumbbell * N_mono
Np = N_mono_total + N_trapped
conf = sample_initial_config(N_dumbbell, r0, kbond, kT, box=box, min_sep=mono_cut, seed=42)

cfg = os.path.join(outdir, "config")
# Only rank 0 should manage the output dir; use PETSc-free rank detection via env set by mpiexec.
rank = int(os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
if rank == 0:
    if os.path.exists(cfg):
        shutil.rmtree(cfg)
    os.makedirs(cfg, exist_ok=True)

sim = PolyStokes.PolyStokes(dt, 20, nsteps * dt, cfg, mm_HI=False, chain_HI=False,
                            fene=False, record_forces=False, tether=False,
                            mono_ev=False, box=box)
sim.particle_info(kT, epsilon, Np, N_trapped, N_mono_total, N_dumbbell, N_mono,
                  beta, kbond, r0, Lmax, tau)
sim.trap_info(ktrap, 0.0, nsteps * dt, weaken_trap=-1)
sim.initial_configuration(conf.flatten())
sim.run()
