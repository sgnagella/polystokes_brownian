
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sphere_rotation_animation
import importlib
import shutil
importlib.reload(sphere_rotation_animation)
import PolyStokes

# Colloidal-brush test simulation.
# The polymer chains are grafted (tethered) to the large colloidal particles:
# each chain's inner-end monomer is bonded to its host colloid. The tether bonds
# themselves are built in the C++ (set_vars / nbonds); the geometry and the
# bond_ids here mirror that topology for placement and visualization.

def main(beta=1, check_conf=False):
    Np = 50
    N_trapped = 2          # the two large colloids (brush cores)
    N_poly = 24            # polymer chains (grafted to the colloids)
    N_mono = 2             # monomers per chain
    N_mono_total = N_poly * N_mono  
    assert Np == N_trapped + N_mono_total, 'Np must equal N_trapped + N_mono_total'

    # Chains split evenly across the colloids. MUST match the C++ assignment in
    # set_vars: host colloid of chain ii is Nm + ii // chains_per_colloid.
    assert N_poly % N_trapped == 0, 'chains must split evenly across the colloids'
    chains_per_colloid = N_poly // N_trapped   # 6 chains per colloid

    dt = 0.001
    samplerate = int(1/dt)   # output every step so the short test yields a multi-frame trajectory

    # Bond/spring parameters. A single global rest length r0 and stiffness kbond
    # apply to *all* bonds, i.e. both the intra-chain bonds and the colloid
    # tethers (so a tethered monomer rests r0 from its colloid centre).
    r0 = 3.0
    Lmax = 10*r0
    kbond = 20.0
    tau = 1000
    ktrap = 500   # harmonic trap holding each colloid at its initial position
    tstart = 0
    trun = 5.0
    kT = 1.0
    epsilon = 1.0   # WCA excluded-volume energy scale
    tmax = 5.0

    # ---- Geometry: two brush-coated colloids on the x-axis ----
    D = 24.0                                   # colloid centre-to-centre separation
    colloid_pos = np.array([[-0.5*D, 0.0, 0.0],
                            [ 0.5*D, 0.0, 0.0]])

    def chain_directions(n, phase=0.0):
        # n unit vectors evenly spaced in the xy-plane
        ang = phase + 2*np.pi*np.arange(n)/n
        return np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=1)

    def return_conf(Np):
        conf = np.zeros((Np, 3))
        # Monomers (indices 0 .. N_mono_total-1): chain ii grafted to colloid ii//cpc
        for ii in range(N_poly):
            c = ii // chains_per_colloid
            local = ii % chains_per_colloid
            # offset the second colloid's chains so the two brushes interleave
            phase = (np.pi / chains_per_colloid) * c
            u = chain_directions(chains_per_colloid, phase)[local]
            P = colloid_pos[c]
            for jj in range(N_mono):
                # monomer jj sits at radial distance r0*(jj+1); the inner monomer
                # (jj=0) is the tethered end, resting r0 from the colloid centre
                conf[ii*N_mono + jj] = P + u * (r0+1) * (jj + 1)
        # Colloids occupy the last N_trapped indices (Nm .. Np-1)
        for c in range(N_trapped):
            conf[N_mono_total + c] = colloid_pos[c]
        return conf

    def return_orientations(Np):
        orientations = np.zeros((Np, 3))
        orientations[:, 0] = 1.0
        return orientations

    def return_complete_conf(Np):
        return np.vstack((return_conf(Np), return_orientations(Np)))

    # ---- Bonds (for visualization; mirrors the C++ bond topology) ----
    n_linear = (N_mono - 1) * N_poly      # intra-chain bonds
    n_tether = N_poly                     # one tether per chain
    nbonds = n_linear + n_tether
    bond_ids = np.zeros((2, nbonds), dtype=int)
    kk = 0
    # linear intra-chain bonds
    for ii in range(N_poly):
        for jj in range(N_mono - 1):
            bond_ids[0, kk] = ii*N_mono + jj
            bond_ids[1, kk] = ii*N_mono + jj + 1
            kk += 1
    # tether bonds: inner-end monomer -> host colloid
    for ii in range(N_poly):
        host = N_mono_total + (ii // chains_per_colloid)
        bond_ids[0, kk] = ii*N_mono
        bond_ids[1, kk] = host
        kk += 1
    assert kk == nbonds

    # ---- Output directories ----
    data_save_dir = f'data/beta_{beta}'
    data_config_dir = os.path.join(data_save_dir, 'config')
    if os.path.exists(data_config_dir):
        shutil.rmtree(data_config_dir)
    os.makedirs(data_config_dir, exist_ok=True)

    conf = return_conf(Np).flatten()

    # Display the initialization
    if check_conf:
        conf_complete = return_complete_conf(Np)
        anim = sphere_rotation_animation.SphereRotationAnimation(Np, n_mono=N_mono_total, beta=beta)
        os.makedirs('figures', exist_ok=True)
        anim.plot_config(conf_complete, bond_ids=bond_ids.T, save='figures/initial_config_brushes.png')

    # Save the parameters
    params = {
        "Np": Np,
        "beta": beta,
        "N_trapped": N_trapped,
        "N_poly": N_poly,
        "N_mono": N_mono,
        "N_mono_total": N_mono_total,
        "bond_ids": bond_ids,
    }
    with open(os.path.join('params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # ---- Run the simulation ----
    sim = PolyStokes.PolyStokes(dt, samplerate, tmax, data_config_dir,
                                mm_HI=False, chain_HI=True, fene=False, record_forces=True,
                                tether=True)

    sim.particle_info(
        kT,
        epsilon,
        Np,
        N_trapped,
        N_mono_total,
        N_poly,
        N_mono,
        beta,
        kbond,
        r0,
        Lmax,
        tau
    )

    sim.trap_info(
        ktrap,
        tstart,
        trun,
        weaken_trap=-1
    )

    sim.initial_configuration(conf)
    sim.run()

    return

if __name__ == "__main__":
    beta = 0.1
    main(beta=beta, check_conf=True)
