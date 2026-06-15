
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

def main(beta=1, check_conf=False):
    Np = 35; 4
    N_trapped = 2 # number of particles trapped in the moving harmonic trap
    N_poly = 11 # number of polymer chains
    N_mono = 3; 2 # number of monomers per chain
    N_mono_total = N_poly * N_mono
    assert Np == N_trapped + N_mono_total, 'Number of particles does not match the number of trapped particles and polymer chains'

    dt = 0.001
    samplerate = 10
    
    params_sim = {
        "dt": dt,
        "samplerate": samplerate,
        "Nc": N_trapped,
        "Npoly": N_poly,
        "Nmono_per_chain": N_mono,
    }
        
    nbonds_per_poly = N_mono - 1        # Number of bonds per polymer chain
    nbonds = nbonds_per_poly * N_poly   # total number of bonds
    # Populate the bond id array (linear chain)
    bond_ids = np.zeros((2, nbonds_per_poly*N_poly), dtype=int)
    kk = 0
    for ii in range(N_poly):
        for jj in range(nbonds_per_poly):
            bond_ids[0, kk] = ii * N_mono + jj
            bond_ids[1, kk] = ii * N_mono + jj + 1
            kk += 1
    
    monomer_x0 = 2.8; 1.7; 2.8; 3.5
    r12 = 6; 3.5; 2.8; 7  # center-to-center distance between the two particles
    r0 = 2.8; 2.8; 2.01*beta; 2.8 # equilibrium bond length
    
    poly_pos = np.array([
        [monomer_x0, 0, 0],
        [r12+monomer_x0, 0, 0],
        [-r12+monomer_x0, 0, 0],
        [-2*r12+monomer_x0, 0, 0],
        [2*r12+monomer_x0, 0, 0], 
        [0, r12+monomer_x0, 0], 
        [0, -r12-monomer_x0, 0],
        [0, -2*r12-monomer_x0, 0],
        [0, -3*r12-monomer_x0, 0],
        [0, 2*r12+monomer_x0, 0],
        [r12/np.sqrt(2) - 0.15, r12/np.sqrt(2) - 0.15, 0]
    ])
    
    x_nonzero = np.nonzero(poly_pos[:-1,0])[0]
    y_nonzero = np.nonzero(poly_pos[:-1,1])[0]
    vert = np.concatenate([np.ones_like(x_nonzero), np.zeros_like(y_nonzero)]) # boolean array to indicate whether the polymer is oriented vertically or horizontally
    #
    assert poly_pos.shape[0] == N_poly, 'Number of polymer chains does not match the number of polymer chain positions'
    # Manually set the 1st polymer com to be horizontally oriented
    # vert = np.insert(vert, 0, 0)
    assert len(vert) == (N_poly-1), 'array size does not match number of polymer chains'

    def return_conf(Np):
        conf = np.zeros((Np,3))
        
        # Place the trapped particles
        conf[-2] = np.array([r12, 0, 0])
        
        # Place the polymer chains
        for ii in range(N_poly-1):
            com = poly_pos[ii]
            
            for jj in range(N_mono):
                # Place monomers along negative y-direction if vert is true
                # Otherwise, along positive x-direction if vert is false 
                conf[ii*N_mono+jj] = np.array([com[0] + r0*jj*(1-vert[ii]), com[1]-r0*jj*vert[ii], 0])
        
        # Handle the last polymer chain separately
        com = poly_pos[-1]
        disp = r0/np.sqrt(2)
        idx_poly = (N_poly-1)*N_mono
        conf[idx_poly] = np.array([com[0]-disp, com[1]-disp, 0])
        conf[idx_poly+1] = com
        conf[idx_poly+2] = np.array([com[0]+disp, com[1]+disp, 0])
        return conf
    
    def return_orientations(Np):
        orientations = np.zeros((Np,3))
        orientations[:-1, 0] = -1
        orientations[-1, 0] = 1     
        return orientations
    
    def return_complete_conf(Np):
        conf = return_conf(Np)
        orientations = return_orientations(Np)
        conf_complete = np.vstack((conf, orientations))
        return conf_complete
    
    # Save the initial configuration and bond ids
    data_save_dir = f'data/beta_{beta}'
    data_config_dir = os.path.join(data_save_dir, 'config')
    params_sim['output_dir'] = data_config_dir
    if os.path.exists(data_config_dir):
        shutil.rmtree(data_config_dir)
    os.makedirs(data_config_dir, exist_ok=True)
        
    # Write the particle positions to conf.in
    conf = return_conf(Np).flatten()
    
    # Display the initialization
    if check_conf: 
        conf_complete = return_complete_conf(Np)
        # Plot the initial configuration
        # print(f"Initial configuration: {conf_complete}")
        anim = sphere_rotation_animation.SphereRotationAnimation(Np, n_mono=N_mono_total, beta=beta)
        os.makedirs('figures', exist_ok=True)
        anim.plot_config(conf_complete, bond_ids=bond_ids.T, save='figures/initial_config.png')
    
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
    
    # with open(os.path.join(data_save_dir, 'params.pkl'), 'wb') as f:
    #     pickle.dump(params, f)
    
    with open(os.path.join('params.pkl'), 'wb') as f:
        pickle.dump(params, f)
    
    # Run the simulation
    tmax = 0.2; 0.1; 2.5; 0.25
    # r0 = 2.8
    Lmax = 10*r0; 2.0*r0; 1.1*r0
    kbond = 0.1; 10; Lmax**(-2) 
    tau = 1000
    ktrap = 500
    tstart = 0
    trun = 0.1;1.0; 0.50; 0.1; 0.25; 2.0; 0.2
    
    sim = PolyStokes.PolyStokes(dt, samplerate, tmax, data_config_dir, mm_HI=False, chain_HI=True, fene=False, record_forces=True)
    
    sim.particle_info(
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
    # exit()
    sim.run()
    
    return

if __name__ == "__main__":
    beta = 0.1; 0.3; 1; 0.5
    main(beta=beta, check_conf=True)
