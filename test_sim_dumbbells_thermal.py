"""
Purely-thermal Brownian-dynamics test with hydrodynamic interactions (HI).

Setup:
  - 40 free 2-bead dumbbells (harmonic bond, no colloid tether)
  - 1 colloid held in a harmonic trap at the origin (also provides HI; exerts no
    driving force on the beads)
  - monomer-monomer HI off (mm_HI=False): the truncated far-field grand mobility
    is not positive-definite for many close monomers, so sqrt(M) fails with it on
  - No external driving on the dumbbells

Diagnostics:
  (1) Bonds: with only conservative bond/WCA forces and thermal noise, the bonds
      fluctuate but must not drift. We sample every 1/dt steps and check that the
      mean bond displacement (per-sample increment of the bond vector, and of the
      bond length) is statistically consistent with zero. A nonzero mean would
      signal a spurious systematic force from the integrator (e.g. an unbalanced
      thermal-drift term).
  (2) Trapped colloid: with a nonzero trap stiffness ktrap, the colloid's position
      must sample the Boltzmann distribution of the harmonic trap potential
      V = 1/2 ktrap |x - x_trap|^2. We check that each Cartesian displacement is
      Gaussian with variance kT/ktrap and that the radial displacement follows
      P(r) ~ r^2 exp(-ktrap r^2 / 2 kT) -- the r0 -> 0 analogue of the bond-length
      test. (The dumbbell tethers integrate out to a colloid-position-independent
      constant, so the colloid's marginal distribution is the pure trap Boltzmann.)
"""

import numpy as np
import os
import pickle
import shutil
import PolyStokes


# ----------------------------------------------------------------------------
# Configuration builders
# ----------------------------------------------------------------------------
def build_initial_config(N_dumbbell, r0, colloid_radius=1.0, box=None):
    """Place N_dumbbell dumbbells on a grid in the xy-plane, clear of the central
    trapped colloid. When `box=[Lx,Ly,Lz]` is given, every bead is placed strictly
    inside the periodic cell [-L/2, L/2) so nothing wraps on top of another particle
    at t=0. Returns a (Np, 3) array; the last row is the colloid."""
    clearance = colloid_radius + 1.5           # keep beads well outside the colloid
    half_bond = 0.5 * r0                        # dumbbell half-extent (oriented along x)

    def centered_grid(half_extent, spacing):
        n = int(np.floor(2.0 * half_extent / spacing)) + 1
        return (np.arange(n) - (n - 1) / 2.0) * spacing

    def cleared_sites(xmax, ymax, spacing):
        xs, ys = centered_grid(xmax, spacing), centered_grid(ymax, spacing)
        s = [(np.hypot(cx, cy), cx, cy) for cx in xs for cy in ys
             if np.hypot(cx, cy) >= clearance]
        s.sort()                               # nearest (still-cleared) sites first
        return s

    if box is not None:
        # Largest coordinate a dumbbell CENTER may take so its beads stay inside the
        # box; shrink the grid spacing until enough cleared sites fit.
        xmax = 0.5 * float(box[0]) - (half_bond + 0.05)
        ymax = 0.5 * float(box[1]) - (half_bond + 0.05)
        spacing = 1.6
        sites = cleared_sites(xmax, ymax, spacing)
        while len(sites) < N_dumbbell and spacing > 0.3:
            spacing *= 0.9
            sites = cleared_sites(xmax, ymax, spacing)
        assert len(sites) >= N_dumbbell, (
            f"cannot fit {N_dumbbell} dumbbells clear of the colloid inside box={box}")
    else:
        spacing, n_side = 1.6, 15
        offs = (np.arange(n_side) - n_side // 2) * spacing
        sites = [(np.hypot(cx, cy), cx, cy) for cx in offs for cy in offs
                 if np.hypot(cx, cy) >= clearance]
        sites.sort()
        assert len(sites) >= N_dumbbell, "grid too small for requested dumbbell count"

    sites = sites[:N_dumbbell]

    Np = 2 * N_dumbbell + 1
    conf = np.zeros((Np, 3))
    for d, (_, cx, cy) in enumerate(sites):
        conf[2 * d]     = [cx - half_bond, cy, 0.0]   # bead 0 of dumbbell d
        conf[2 * d + 1] = [cx + half_bond, cy, 0.0]   # bead 1 of dumbbell d
    conf[-1] = [0.0, 0.0, 0.0]                        # trapped colloid at the origin

    if box is not None:
        half = 0.5 * np.asarray(box, dtype=float)
        assert np.all(np.abs(conf) <= half + 1e-9), \
            "initial configuration falls outside the box"
    return conf


def sample_initial_config(N_dumbbell, r0, kbond, kT, colloid_radius=1.0, box=None,
                          buffer=0.5, min_sep=0.0, seed=None, max_tries=1000):
    """Randomly initialize N_dumbbell free 2-bead dumbbells around a central probe
    (colloid at the origin), equilibrium-consistent rather than grid-placed:

      - Placement: each dumbbell's CENTER OF MASS is drawn UNIFORMLY over the box
        volume (independent per-axis Uniform(-Lx/2, Lx/2) etc.) -- `box=[Lx,Ly,Lz]`
        is required.
      - Alignment: each dumbbell's bond direction is independently uniform on the
        sphere (isotropic orientation).
      - Bond vector: drawn from the Gaussian Boltzmann distribution of the harmonic
        spring linearized about r0: vec = r0*u_bond + xi, xi ~ N(0, kT/kbond, size=3)
        (each Cartesian component of the displacement from equilibrium is an
        independent Gaussian with the equipartition variance kT/kbond).

    A bead offset from a COM near a face can land outside the box; rather than
    rejecting, it is wrapped back into the primary cell using the same convention
    the C++ integrator applies at runtime (Box::wrap / Box::minimum_image, box.cpp:
    v -= L*round(v/L)). Each dumbbell is independently rejection-sampled (redrawing
    its COM and bond vector) until (a) both wrapped beads clear the colloid and (b)
    both beads are at least `min_sep` (minimum-image) from every bead of every
    PREVIOUSLY placed dumbbell -- i.e. no inter-dumbbell monomer overlap at t=0, so
    the monomer excluded-volume (WCA) force starts at zero between distinct
    dumbbells. Set `min_sep = 2^(1/6) * (2*beta)` (the monomer-monomer WCA cutoff)
    when running with mono_ev=True. The two beads of a dumbbell are NOT checked
    against each other -- they are the bonded pair and are allowed to sit at ~r0
    (which is inside the WCA core; the bond + WCA relax it at runtime). Since the
    colloid sits at the box center, a wrapped position's minimum-image distance to
    it is just its raw norm. Returns a (2*N_dumbbell + 1, 3) array; the last row is
    the colloid at the origin (matches build_initial_config's layout).
    """
    if box is None:
        raise ValueError("sample_initial_config requires box=[Lx,Ly,Lz] for uniform placement")

    rng = np.random.default_rng(seed)
    clearance = colloid_radius + buffer
    L = np.asarray(box, dtype=float)
    half = 0.5 * L
    sigma = np.sqrt(kT / kbond)

    def random_unit_vector():
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)

    def wrap(v):
        return v - L * np.round(v / L)     # matches Box::wrap (box.cpp)

    def min_image_gap(p, others):
        """Smallest minimum-image distance from point p (3,) to any row of others
        (M,3); +inf when others is empty."""
        if others.shape[0] == 0:
            return np.inf
        d = others - p
        d -= L * np.round(d / L)           # nearest image (matches Box::minimum_image)
        return np.sqrt((d * d).sum(axis=1)).min()

    beads = np.zeros((2 * N_dumbbell, 3))
    for d in range(N_dumbbell):
        placed = beads[:2 * d]             # beads of the already-accepted dumbbells
        for _ in range(max_tries):
            center = rng.uniform(-half, half)      # uniform over the box volume
            bond_vec = r0 * random_unit_vector() + rng.normal(scale=sigma, size=3)
            b0 = wrap(center - 0.5 * bond_vec)
            b1 = wrap(center + 0.5 * bond_vec)

            if np.linalg.norm(b0) < clearance or np.linalg.norm(b1) < clearance:
                continue                            # too close to the central colloid
            if min_sep > 0.0 and (min_image_gap(b0, placed) < min_sep or
                                  min_image_gap(b1, placed) < min_sep):
                continue                            # overlaps a previously placed monomer

            beads[2 * d]     = b0
            beads[2 * d + 1] = b1
            break
        else:
            raise RuntimeError(f"could not place dumbbell {d} without overlap after "
                               f"{max_tries} tries; check min_sep/clearance vs box size")

    conf = np.zeros((2 * N_dumbbell + 1, 3))
    conf[:2 * N_dumbbell] = beads
    conf[-1] = [0.0, 0.0, 0.0]                        # trapped colloid at the origin
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


def equilibrium_trap_stats(ktrap, kT):
    """Theoretical equilibrium statistics for a colloid in a 3D harmonic trap
    V = 1/2 ktrap |x - x_trap|^2.

    Each Cartesian displacement component is Gaussian with mean 0 and variance
    sigma^2 = kT/ktrap. The radial displacement r = |x - x_trap| then follows the
    Maxwell-Boltzmann form P(r) ~ r^2 exp(-r^2 / 2 sigma^2) (the r^2 is the
    spherical-shell Jacobian) -- exactly the r0 -> 0 limit of equilibrium_length_stats
    with spring constant ktrap. The radial moments have closed forms:
        <r>  = 2 sqrt(2/pi) sigma,  mode = sqrt(2) sigma,  std = sqrt(3 - 8/pi) sigma.
    Returns (sigma, r_mean, r_mode, r_std).
    """
    sigma = np.sqrt(kT / ktrap)
    r_mean = 2.0 * np.sqrt(2.0 / np.pi) * sigma
    r_mode = np.sqrt(2.0) * sigma
    r_std = np.sqrt(3.0 - 8.0 / np.pi) * sigma
    return sigma, r_mean, r_mode, r_std


def minimum_image(vec, box):
    """Minimum-image an array of separation vectors under an orthorhombic box.
    `box` is [Lx,Ly,Lz] (or None for no PBC). Matches the C++ Box convention."""
    if box is None:
        return vec
    L = np.asarray(box, dtype=float)
    return vec - L * np.round(vec / L)


def bond_statistics(times, traj, N_dumbbell, kbond=1.0, r0=0.2, kT=1.0, box=None):
    """Statistics on the bond displacements for a purely thermal, undriven system.

    The convention-free 'zero on average' quantities are the mean bond VECTOR and
    the mean per-sample bond-vector increment d b_i(k) = b_i(t_{k+1}) - b_i(t_k),
    with b_i = x_{2i+1} - x_{2i}: both vanish at equilibrium (isotropy + no drift).
    The scalar length carries the r^2 shell factor, so its mean sits at/above r0;
    we compare it against the theoretical equilibrium rather than assuming r0.

    Under PBC the output positions are wrapped, so bond vectors are taken to the
    minimum image (a bond straddling a boundary then has a physical, not ~L, length).
    """
    n_frames = traj.shape[0]
    if n_frames < 2:
        print(f"Only {n_frames} frame(s) recorded; need >= 2 for bond-displacement "
              f"statistics. Increase tmax.")
        return

    b0 = traj[:, 0:2 * N_dumbbell:2, :]        # (frames, N_dumbbell, 3) bead 0 of each dumbbell
    b1 = traj[:, 1:2 * N_dumbbell:2, :]        # (frames, N_dumbbell, 3) bead 1 of each dumbbell
    bond_vec = minimum_image(b1 - b0, box)     # bond vectors (nearest image under PBC)
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


def colloid_displacements(times, traj, N_trapped, box=None, drop_transient=0.2):
    """Displacements of the trapped colloid(s) from their trap centers.

    The colloids are the LAST N_trapped particles (see build_initial_config); the
    trap center of each is its initial position, which equals the t=0 frame -- this
    matches the C++ xpos0 convention (initial_configuration seeds xpos0). Positions
    are taken to the nearest image of the trap center (no-op when box is None), and
    the leading `drop_transient` fraction of the run is discarded so the colloid has
    relaxed from its exact-center start (r=0 carries no Boltzmann weight).
    Returns (disp, n_frames_used) with disp of shape (n_frames_used, N_trapped, 3).
    """
    pos = traj[:, -N_trapped:, :]                  # (frames, N_trapped, 3)
    centers = traj[0, -N_trapped:, :]              # trap centers = initial positions
    disp = minimum_image(pos - centers, box)
    if drop_transient > 0.0:
        t0 = times[0] + drop_transient * (times[-1] - times[0])
        keep = times >= t0
        disp = disp[keep]
    return disp, disp.shape[0]


def trap_statistics(times, traj, N_trapped, ktrap, kT=1.0, box=None,
                    drop_transient=0.2):
    """Verify the trapped colloid samples the harmonic-trap Boltzmann distribution.

    Checks, on the colloid displacement from its trap center:
      - each Cartesian component has mean 0 (z-test) and variance kT/ktrap;
      - the radial displacement r=|x-x_trap| matches P(r) ~ r^2 exp(-ktrap r^2/2kT).

    Note only N_trapped colloid(s) contribute, so the statistics are dominated by
    the number of (decorrelated) time frames rather than the particle count.
    """
    n_frames = traj.shape[0]
    if n_frames < 2:
        print(f"Only {n_frames} frame(s) recorded; need >= 2 for trap statistics. "
              f"Increase tmax.")
        return

    disp, n_used = colloid_displacements(times, traj, N_trapped, box, drop_transient)
    comps = disp.reshape(-1, 3)                     # (n_used*N_trapped, 3)
    r = np.linalg.norm(disp, axis=-1).reshape(-1)   # radial displacements, pooled
    n = comps.shape[0]

    sigma, r_mean, r_mode, r_std = equilibrium_trap_stats(ktrap, kT)

    # per-axis mean (should vanish) and its standard error
    mean_c = comps.mean(axis=0)
    se_c = comps.std(axis=0, ddof=1) / np.sqrt(n)
    # pooled per-component variance vs theory sigma^2 (all 3 axes pooled by isotropy)
    var_meas = comps.reshape(-1).var(ddof=1)
    sigma_meas = np.sqrt(var_meas)
    # SE of a variance ~ var*sqrt(2/(N-1)) for uncorrelated normals (optimistic here;
    # time-adjacent frames are correlated, so treat |z| as a guide, not a hard gate).
    Ncomp = comps.size
    se_var = var_meas * np.sqrt(2.0 / (Ncomp - 1))

    pcts = np.percentile(r, [5, 50, 95])

    def zline(name, m, s, target=0.0):
        z = (m - target) / s if s > 0 else np.nan
        flag = "OK" if abs(z) < 3 else "** off **"
        return f"   {name} = {m:+.3e}  +/- {s:.3e}  (z = {z:+.2f} vs {target:g})  {flag}"

    print("\n" + "=" * 70)
    print("TRAPPED-COLLOID BOLTZMANN STATISTICS (harmonic trap, ktrap = "
          f"{ktrap:g})")
    print("=" * 70)
    print(f"frames used        : {n_used}/{n_frames}   colloids: {N_trapped}   "
          f"component samples: {Ncomp}")
    print(f"time span (kept)    : {times[times >= times[0] + drop_transient*(times[-1]-times[0])][0]:.3f} "
          f"-> {times[-1]:.3f}")
    print("-" * 70)
    print("Mean displacement per axis (centered trap => 0):")
    for a, m, s in zip("xyz", mean_c, se_c):
        print(zline(f"<dx_{a}>", m, s))
    print("-" * 70)
    print("Per-component variance  (theory sigma^2 = kT/ktrap):")
    print(zline("Var[dx]", var_meas, se_var, target=sigma**2))
    print(f"   sigma: simulated={sigma_meas:.4f}   theory={sigma:.4f}   "
          f"(ratio {sigma_meas/sigma:.3f})")
    print("-" * 70)
    print("Radial displacement distribution  (pdf ~ r^2 exp(-ktrap r^2 / 2kT)):")
    print(f"   simulated : mean={r.mean():.4f}  std={r.std():.4f}  "
          f"[5/50/95 %ile = {pcts[0]:.4f}/{pcts[1]:.4f}/{pcts[2]:.4f}]")
    print(f"   theory    : mean={r_mean:.4f}  mode={r_mode:.4f}  std={r_std:.4f}")
    print("=" * 70 + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(beta=0.1, dt=0.001, tmax=25.0, samplerate=None, run=True, box=None,
         N_dumbbell=6000, kbond=1.0):
    if samplerate is None:
        samplerate = int(1 / dt)               # sample every 1/dt timesteps

    N_mono = 2                                  # 2-bead dumbbells
    N_trapped = 1                              # single trapped colloid
    N_mono_total = N_dumbbell * N_mono
    Np = N_mono_total + N_trapped

    # Bond / interaction parameters. r0 is set 0.01 above the monomer contact
    # diameter (2*beta) as a small regularization so bonded beads sit near/above the
    # monomer-monomer WCA core and are only weakly (ideally not) repelled by it.
    r0 = 2.0 * beta + 0.01                     # bond rest length
    Lmax = 1.5*r0                             # unused (harmonic), kept for the API
    tau = 1000
    kT = 1.0
    epsilon = 5.0                              # WCA excluded-volume energy scale
    ktrap = 10                                # harmonic trap holding the colloid at the origin
    fene = False
    # Monomer-monomer WCA cutoff (matches C++ rcuts[0] = 2^(1/6) * sigma[0],
    # sigma[0] = 2*beta): initialize distinct dumbbells at least this far apart so no
    # inter-dumbbell excluded-volume overlap exists at t=0.
    mono_cut = 2.0 ** (1.0 / 6.0) * (2.0 * beta)
    conf = sample_initial_config(N_dumbbell, r0, kbond, kT, box=box, min_sep=mono_cut)
    bond_ids = build_bond_ids(N_dumbbell)

    data_save_dir = f'data/test_dumbbells_thermal_beta_{beta}_fene_{fene}'
    data_config_dir = os.path.join(data_save_dir, 'config')
    if run:
        if os.path.exists(data_config_dir):
            shutil.rmtree(data_config_dir)
        os.makedirs(data_config_dir, exist_ok=True)

        params = {
            "Np": Np, "beta": beta, "N_trapped": N_trapped,
            "N_poly": N_dumbbell, "N_mono": N_mono,
            "N_mono_total": N_mono_total, "bond_ids": bond_ids,
            "box": box, "kT": kT, "kbond": kbond, "r0": r0, "ktrap": ktrap,
        }
        with open(os.path.join(data_save_dir, 'params.pkl'), 'wb') as f:
            pickle.dump(params, f)

        sim = PolyStokes.PolyStokes(
            dt, samplerate, tmax, data_config_dir,
            mm_HI=False,       # monomer-monomer HI off
            chain_HI=False,
            fene=fene,        # harmonic bonds
            record_forces=False,
            tether=False,      # free dumbbells (NOT tethered to the colloid)
            mono_ev=True,      # monomer-monomer excluded volume (WCA) on
            box=box,           # None => unbounded; [Lx,Ly,Lz] => periodic
        )
        sim.particle_info(
            kT, epsilon, Np, N_trapped, N_mono_total,
            N_dumbbell, N_mono, beta, kbond, r0, Lmax, tau,
        )
        sim.trap_info(ktrap, 0.0, tmax, weaken_trap=-1)
        sim.initial_configuration(conf.flatten())
        sim.run()

    # Analyze bond displacements (undriven dumbbells: no drift)
    times, traj = read_trajectory(data_config_dir)
    bond_statistics(times, traj, N_dumbbell, kbond=kbond, r0=r0, kT=kT, box=box)

    # Verify the trapped colloid samples the harmonic-trap Boltzmann distribution.
    # (For a histogram-vs-theory figure, run plot_trap_distribution.py afterward.)
    if ktrap > 0:
        trap_statistics(times, traj, N_trapped, ktrap, kT=kT, box=box)


if __name__ == "__main__":
    dt = 0.0001
    samplerate = 10; int(1/dt) # sample on brownian timescale of colloid
    beta = 0.1
    main(beta=beta, dt=dt, samplerate=samplerate, tmax=1.0, box=[50,50,50],
         N_dumbbell=10000, kbond=0.1)

