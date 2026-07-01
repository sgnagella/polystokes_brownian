"""
Purely-thermal Brownian-dynamics test with hydrodynamic interactions (HI).

Setup:
  - 40 free 2-bead dumbbells (harmonic bond, no colloid tether)
  - 1 colloid trapped at the origin (provides HI; not a driving force on the beads)
  - monomer-monomer HI off (mm_HI=False): the truncated far-field grand mobility
    is not positive-definite for many close monomers, so sqrt(M) fails with it on
  - No external driving on the dumbbells

Diagnostic:
  With only conservative bond/WCA forces and thermal noise, the bonds fluctuate
  but must not drift. We sample every 1/dt steps and check that the mean bond
  displacement (per-sample increment of the bond vector, and of the bond length)
  is statistically consistent with zero. A nonzero mean would signal a spurious
  systematic force from the integrator (e.g. an unbalanced thermal-drift term).
"""

import numpy as np
import os
import pickle
import shutil
import PolyStokes


# ----------------------------------------------------------------------------
# Configuration builders
# ----------------------------------------------------------------------------
def build_initial_config(N_dumbbell, r0, colloid_radius=1.0):
    """Place N_dumbbell dumbbells on a grid in the xy-plane, clear of the
    central trapped colloid. Returns a (Np, 3) array; the last row is the colloid."""
    spacing = 1.6                              # inter-dumbbell spacing (>> dumbbell size)
    clearance = colloid_radius + 1.5           # keep beads well outside the colloid
    n_side = 15                                # grid large enough to yield >= N_dumbbell sites
    offs = (np.arange(n_side) - n_side // 2) * spacing

    sites = []
    for cx in offs:
        for cy in offs:
            r = np.hypot(cx, cy)
            if r >= clearance:
                sites.append((r, cx, cy))
    sites.sort()                               # nearest (still-cleared) sites first
    assert len(sites) >= N_dumbbell, "grid too small for requested dumbbell count"
    sites = sites[:N_dumbbell]

    Np = 2 * N_dumbbell + 1
    conf = np.zeros((Np, 3))
    for d, (_, cx, cy) in enumerate(sites):
        conf[2 * d]     = [cx - 0.5 * r0, cy, 0.0]   # bead 0 of dumbbell d
        conf[2 * d + 1] = [cx + 0.5 * r0, cy, 0.0]   # bead 1 of dumbbell d
    conf[-1] = [0.0, 0.0, 0.0]                       # trapped colloid at the origin
    return conf


def build_bond_ids(N_dumbbell):
    """Intra-dumbbell bonds: dumbbell d bonds beads (2d, 2d+1)."""
    bond_ids = np.zeros((2, N_dumbbell), dtype=int)
    for d in range(N_dumbbell):
        bond_ids[0, d] = 2 * d
        bond_ids[1, d] = 2 * d + 1
    return bond_ids


# ----------------------------------------------------------------------------
# Trajectory I/O and statistics
# ----------------------------------------------------------------------------
def read_trajectory(config_dir):
    """Read config_<t>_.txt frames (one particle position per line), sorted by time."""
    files = [f for f in os.listdir(config_dir)
             if f.startswith('config_') and f.endswith('.txt')]
    files.sort(key=lambda x: float(x.split('_')[1]))
    times = np.array([float(f.split('_')[1]) for f in files])
    frames = []
    for f in files:
        with open(os.path.join(config_dir, f)) as fh:
            frames.append([[float(v) for v in line.split()] for line in fh])
    return times, np.array(frames)


def equilibrium_length_stats(kbond, r0, kT):
    """Theoretical equilibrium bond-LENGTH distribution P(r) ~ r^2 exp(-V/kT),
    V = 1/2 kbond (r-r0)^2 (the r^2 is the 3D spherical-shell Jacobian).
    Returns (mean, mode, std). For a stiff spring these collapse onto r0."""
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # NumPy 2.0 renamed trapz
    hi = r0 + 8.0 * np.sqrt(kT / kbond) + 3.0
    r = np.linspace(0.0, hi, 40000)
    w = r**2 * np.exp(-0.5 * kbond * (r - r0)**2 / kT)
    w /= trapz(w, r)
    mean = trapz(r * w, r)
    mode = r[np.argmax(w)]
    std = np.sqrt(trapz((r - mean)**2 * w, r))
    return mean, mode, std


def bond_statistics(times, traj, N_dumbbell, kbond=1.0, r0=0.2, kT=1.0):
    """Statistics on the bond displacements for a purely thermal, undriven system.

    The convention-free 'zero on average' quantities are the mean bond VECTOR and
    the mean per-sample bond-vector increment d b_i(k) = b_i(t_{k+1}) - b_i(t_k),
    with b_i = x_{2i+1} - x_{2i}: both vanish at equilibrium (isotropy + no drift).
    The scalar length carries the r^2 shell factor, so its mean sits at/above r0;
    we compare it against the theoretical equilibrium rather than assuming r0.
    """
    n_frames = traj.shape[0]
    if n_frames < 2:
        print(f"Only {n_frames} frame(s) recorded; need >= 2 for bond-displacement "
              f"statistics. Increase tmax.")
        return

    b0 = traj[:, 0:2 * N_dumbbell:2, :]        # (frames, N_dumbbell, 3) bead 0 of each dumbbell
    b1 = traj[:, 1:2 * N_dumbbell:2, :]        # (frames, N_dumbbell, 3) bead 1 of each dumbbell
    bond_vec = b1 - b0                          # bond vectors
    bond_len = np.linalg.norm(bond_vec, axis=-1)

    dvec = np.diff(bond_vec, axis=0).reshape(-1, 3)   # bond-vector increments, pooled
    dlen = np.diff(bond_len, axis=0).reshape(-1)      # bond-length increments, pooled
    n = dvec.shape[0]

    # --- primary zero test: mean bond vector and mean bond-vector increment ---
    vec_flat = bond_vec.reshape(-1, 3)
    mean_vec = vec_flat.mean(axis=0)
    se_vec = vec_flat.std(axis=0, ddof=1) / np.sqrt(vec_flat.shape[0])
    mean_dvec = dvec.mean(axis=0)
    se_dvec = dvec.std(axis=0, ddof=1) / np.sqrt(n)

    # --- bond-length distribution vs theory ---
    th_mean, th_mode, th_std = equilibrium_length_stats(kbond, r0, kT)
    pcts = np.percentile(bond_len.reshape(-1), [5, 50, 95])

    def zline(name, m, s):
        z = m / s if s > 0 else np.nan
        flag = "OK" if abs(z) < 3 else "** nonzero **"
        return f"   {name} = {m:+.3e}  +/- {s:.3e}  (z = {z:+.2f})  {flag}"

    print("\n" + "=" * 70)
    print("BOND-DISPLACEMENT STATISTICS (purely thermal, undriven dumbbells)")
    print("=" * 70)
    print(f"samples (frames)   : {n_frames}   dumbbells: {N_dumbbell}   "
          f"increments pooled: {n}")
    print(f"time span          : {times[0]:.3f} -> {times[-1]:.3f}")
    print("-" * 70)
    print("Mean bond VECTOR (isotropy => 0 on average):")
    for a, m, s in zip("xyz", mean_vec, se_vec):
        print(zline(f"<b_{a}>", m, s))
    print("Mean per-sample bond-vector DISPLACEMENT (no drift => 0):")
    for a, m, s in zip("xyz", mean_dvec, se_dvec):
        print(zline(f"<db_{a}>", m, s))
    print("-" * 70)
    print("Bond LENGTH distribution  (length pdf ~ r^2 exp(-V/kT), so mean >= r0):")
    print(f"   simulated : mean={bond_len.mean():.4f}  std={bond_len.std():.4f}  "
          f"[5/50/95 %ile = {pcts[0]:.3f}/{pcts[1]:.3f}/{pcts[2]:.3f}]")
    print(f"   theory    : mean={th_mean:.4f}  mode={th_mode:.4f}  std={th_std:.4f}   (r0={r0})")
    print(f"   per-sample mean length: first={bond_len[0].mean():.4f}  "
          f"last={bond_len[-1].mean():.4f}  (stationary => no runaway)")
    print("=" * 70 + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(beta=0.1, dt=0.001, tmax=25.0, samplerate=None, run=True):
    if samplerate is None:
        samplerate = int(1 / dt)               # sample every 1/dt timesteps

    N_dumbbell = 50
    N_mono = 2                                  # 2-bead dumbbells
    N_trapped = 1                              # single trapped colloid
    N_mono_total = N_dumbbell * N_mono
    Np = N_mono_total + N_trapped

    # Bond / interaction parameters
    r0 = 2.0 * beta                            # bond rest length (small)
    kbond = 1.0
    Lmax = 10 * r0                             # unused (harmonic), kept for the API
    tau = 1000
    kT = 1.0
    epsilon = 1.0                              # WCA excluded-volume energy scale
    ktrap = 500                                # holds the colloid at the origin

    conf = build_initial_config(N_dumbbell, r0)
    bond_ids = build_bond_ids(N_dumbbell)

    data_save_dir = f'data/dumbbells_thermal_beta_{beta}'
    data_config_dir = os.path.join(data_save_dir, 'config')
    if run:
        if os.path.exists(data_config_dir):
            shutil.rmtree(data_config_dir)
        os.makedirs(data_config_dir, exist_ok=True)

        params = {
            "Np": Np, "beta": beta, "N_trapped": N_trapped,
            "N_poly": N_dumbbell, "N_mono": N_mono,
            "N_mono_total": N_mono_total, "bond_ids": bond_ids,
        }
        with open(os.path.join(data_save_dir, 'params.pkl'), 'wb') as f:
            pickle.dump(params, f)

        sim = PolyStokes.PolyStokes(
            dt, samplerate, tmax, data_config_dir,
            mm_HI=False,       # monomer-monomer HI off
            chain_HI=False,
            fene=False,        # harmonic bonds
            record_forces=False,
            tether=False,      # free dumbbells (NOT tethered to the colloid)
        )
        sim.particle_info(
            kT, epsilon, Np, N_trapped, N_mono_total,
            N_dumbbell, N_mono, beta, kbond, r0, Lmax, tau,
        )
        sim.trap_info(ktrap, 0.0, tmax, weaken_trap=-1)
        sim.initial_configuration(conf.flatten())
        sim.run()

    # Analyze bond displacements
    times, traj = read_trajectory(data_config_dir)
    bond_statistics(times, traj, N_dumbbell, kbond=kbond, r0=r0, kT=kT)


if __name__ == "__main__":
    dt = 1e-3
    samplerate = 1
    main(dt=dt, samplerate=samplerate, tmax=1.0)
