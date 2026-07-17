#!/usr/bin/env python
"""Stage-0 profiling driver: run a moderate-Nm dumbbell case for a few hundred
steps with PETSc -log_view enabled (PetscLogView is called at the end of run()).

Usage:
    PYTHONPATH=build python scratch_profile.py [N_dumbbell] [nsteps]
"""
import os, sys, shutil
import numpy as np
import PolyStokes
from test_sim_dumbbells_thermal import sample_initial_config, build_bond_ids

N_dumbbell = int(sys.argv[1]) if len(sys.argv) > 1 else 5000   # 10k monomers
nsteps     = int(sys.argv[2]) if len(sys.argv) > 2 else 300

beta = 0.001
dt = 1e-5
samplerate = 100000              # effectively never sample (avoid I/O in the timed run)
tmax = nsteps * dt
box = [10, 10, 10]
kbond = 1.0

N_mono = 2
N_trapped = 1
N_mono_total = N_dumbbell * N_mono
Np = N_mono_total + N_trapped
r0 = 2.0 ** (1.0 / 6.0) * (2.0 * beta) + 0.03
Lmax = 1.5 * r0
tau = 1000
kT = 1.0
epsilon = 5.0
ktrap = 1.0
fene = False
mono_cut = 2.0 ** (1.0 / 6.0) * (2.0 * beta)

conf = sample_initial_config(N_dumbbell, r0, kbond, kT, box=box, min_sep=mono_cut)

data_dir = f"data/profile_N_{N_dumbbell}/config"
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir, exist_ok=True)

print(f"[profile] N_dumbbell={N_dumbbell} Nm={N_mono_total} nsteps={nsteps} "
      f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','unset')}")

sim = PolyStokes.PolyStokes(
    dt, samplerate, tmax, data_dir,
    mm_HI=False, chain_HI=False, fene=fene, record_forces=False,
    tether=False, mono_ev=False, box=box,
)
sim.particle_info(kT, epsilon, Np, N_trapped, N_mono_total,
                  N_dumbbell, N_mono, beta, kbond, r0, Lmax, tau)
sim.trap_info(ktrap, 0.0, tmax, weaken_trap=-1)
sim.initial_configuration(conf.flatten())
sim.run()
