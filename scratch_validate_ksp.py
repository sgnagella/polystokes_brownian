#!/usr/bin/env python
"""Validate that KSPMINRES reproduces KSPGMRES physics. Runs the same short sim
(same seed 12345) under each KSP type via PETSC_OPTIONS and compares trajectories.

Usage: PYTHONPATH=build python scratch_validate_ksp.py
  (runs both internally by re-exec with different PETSC_OPTIONS)
"""
import os, sys, shutil, subprocess
import numpy as np

N_dumbbell = 200
nsteps = 200
beta = 0.001
dt = 1e-5
box = [10, 10, 10]


def run(nthreads, outdir, ksp_type="minres"):
    env = dict(os.environ)
    env["PETSC_OPTIONS"] = f"-ksp_type {ksp_type}"
    env["OMP_NUM_THREADS"] = str(nthreads)
    env["OPENBLAS_NUM_THREADS"] = str(nthreads)
    env["PYTHONPATH"] = "build"
    env["POLYSTOKES_VALIDATE_OUT"] = outdir
    subprocess.run([sys.executable, __file__, "child"], env=env, check=True)


def child():
    outdir = os.environ["POLYSTOKES_VALIDATE_OUT"]
    import PolyStokes
    from test_sim_dumbbells_thermal import sample_initial_config

    r0 = 2.0 ** (1.0 / 6.0) * (2.0 * beta) + 0.03
    Lmax, tau, kT, epsilon, ktrap, kbond = 1.5 * r0, 1000, 1.0, 5.0, 1.0, 1.0
    mono_cut = 2.0 ** (1.0 / 6.0) * (2.0 * beta)
    N_mono, N_trapped = 2, 1
    N_mono_total = N_dumbbell * N_mono
    Np = N_mono_total + N_trapped
    # Fixed seed so BOTH KSP runs start from the identical initial configuration
    # (the C++ Brownian RNG is already hardcoded to 12345), making the runs differ ONLY
    # by the linear solver -- so early-frame agreement isolates solver correctness.
    conf = sample_initial_config(N_dumbbell, r0, kbond, kT, box=box, min_sep=mono_cut, seed=42)

    cfg = os.path.join(outdir, "config")
    if os.path.exists(cfg):
        shutil.rmtree(cfg)
    os.makedirs(cfg, exist_ok=True)

    sim = PolyStokes.PolyStokes(dt, 10, nsteps * dt, cfg, mm_HI=False, chain_HI=False,
                                fene=False, record_forces=False, tether=False,
                                mono_ev=False, box=box)
    sim.particle_info(kT, epsilon, Np, N_trapped, N_mono_total, N_dumbbell, N_mono,
                      beta, kbond, r0, Lmax, tau)
    sim.trap_info(ktrap, 0.0, nsteps * dt, weaken_trap=-1)
    sim.initial_configuration(conf.flatten())
    sim.run()


def read_last_config(cfg):
    from test_sim_dumbbells_thermal import read_trajectory
    times, traj = read_trajectory(cfg)
    return times, traj


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child()
        sys.exit(0)

    base = "data/validate_ksp"
    # Compare OMP_NUM_THREADS=1 vs 8 (same MINRES solver, same IC seed). The mob() AB loop
    # writes disjoint columns, so threaded output must match serial ~bitwise.
    run(1, f"{base}/t1")
    run(8, f"{base}/t8")

    tg, trg = read_last_config(f"{base}/t1/config")
    tm, trm = read_last_config(f"{base}/t8/config")
    trg, trm = np.asarray(trg), np.asarray(trm)
    print(f"\n=== 1-thread vs 8-thread trajectory comparison ({N_dumbbell} dumbbells, {nsteps} steps) ===")
    print(f"frames: gmres {trg.shape}, minres {trm.shape}")
    n = min(len(trg), len(trm))
    # Minimum-image the difference: free dumbbells diffuse across box boundaries, so raw
    # coords can differ by ~L between runs for a physically identical (wrapped) position.
    L = np.asarray(box, dtype=float)
    d = trg[:n] - trm[:n]
    d = d - L * np.round(d / L)          # nearest-image separation
    diff = np.abs(d)
    print(f"[min-image]  max |pos diff|  = {diff.max():.3e}")
    print(f"[min-image]  mean |pos diff| = {diff.mean():.3e}")
    print(f"[min-image]  per-frame max:", np.round(np.abs(d).reshape(n, -1).max(axis=1), 4))
