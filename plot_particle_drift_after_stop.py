"""
    Script to analyze simulation results
    Plots the polar angle of the central particle as a function of time from its quaternion representation
    Plots dependence on the size parameter, beta
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import importlib

FONTSIZE = 16 

plt.rcParams.update({
    "pdf.fonttype":42, 
    "ps.fonttype":42,
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset":"custom",
    # "mathtext.rm": "sans", 
    "font.size": FONTSIZE,
    "axes.linewidth": 1.25, 
    "xtick.labelsize": FONTSIZE -2, 
    "ytick.labelsize": FONTSIZE -2,
    "legend.fontsize": FONTSIZE -3})

ASPECT_RATIO = 1.25 # width/height
MARKERSIZE = 5
CAPSIZE = 4
LINEWIDTH = 1.5
FIG_WIDTH = 5 # inches
FIG_HEIGHT = FIG_WIDTH/ASPECT_RATIO
MARKEREDGEWIDTH = 1

def main(T=1.0, save_to_beta_dir=False):
    """
        T indicates the period of revolution of the moving particle 
    """
    
    betas = [0.1]; [1, 0.3, 0.1]; [0.1,0.5,1]
    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()
    for beta in betas:
        save_dir = f'particle_drift_after_stop_test'
        fig_save_dir = f'figures/beta_{beta}'
        os.makedirs(fig_save_dir, exist_ok=True)
        
        params = pickle.load(open(f'params.pkl', 'rb'))
        N_trapped = 2 
        N_poly = params['N_poly']
        N_mono = params['N_mono']
        dt = 5e-4; 1e-4
        samplerate = 25; 100
        N_mono_total = N_poly * N_mono
        Np = N_trapped + N_mono_total
        beta = params['beta']
    
        positions_trap_large = np.load(f'{save_dir}/trajectory_trap_large.npy')
        positions_trap_small = np.load(f'{save_dir}/trajectory_trap_small.npy')
        
        # Extract the number of frames and particles
        Nframes, Nparticles, _ = positions_trap_large.shape
        
        # Plot the central particle azimuthal angle as a function of time
        times = np.arange(Nframes) * samplerate * dt
        
        mask_time = times >= T
        num_times = np.sum(mask_time)
        print("number of times aftero T:", num_times)
        
        assert Nparticles == Np, f"Number of particles in trajectory ({Nparticles}) does not match expected number of particles ({Np})"
    
        particle_pos_trap_large = positions_trap_large[mask_time, -2:]
        particle_pos_trap_small = positions_trap_small[mask_time, -2:]
        
        ax1.plot(times[mask_time], particle_pos_trap_large[:,-1,0], '-', lw=LINEWIDTH, label=r'$\beta = $' + f'{beta:.2f}' + ' (large trap)')
        ax1.plot(times[mask_time], particle_pos_trap_small[:,-1,0], '-', lw=LINEWIDTH, label=r'$\beta = $' + f'{beta:.2f}' + ' (small trap)')
        
        print("particle_pos_trap_large shape:", particle_pos_trap_large.shape)
        print("particle_pos_trap_small shape:", particle_pos_trap_small.shape)
        
        # Calculate the distance between the two particles
        r12_trap_large = particle_pos_trap_large[:,0,0] - particle_pos_trap_large[:,1,0]
        r12_trap_large = np.sqrt(r12_trap_large**2)
        r12_trap_small = particle_pos_trap_small[:,0,0] - particle_pos_trap_small[:,1,0]
        r12_trap_small = np.sqrt(r12_trap_small**2)
        
        # Plot the distance as a function of time
        ax.plot(times[mask_time], r12_trap_large, '-', lw=LINEWIDTH, label=r'$\beta = $' + f'{beta:.2f}' + ' (large trap)')
        ax.plot(times[mask_time], r12_trap_small, '-', lw=LINEWIDTH, label=r'$\beta = $' + f'{beta:.2f}' + ' (small trap)')
        
        # Save the data
        np.save(f'{save_dir}/particle_drift_after_stop_times.npy', times[mask_time])
        np.save(f'{save_dir}/particle_drift_after_stop_r12_trap_large.npy', r12_trap_large)
        np.save(f'{save_dir}/particle_drift_after_stop_r12_trap_small.npy', r12_trap_small)
        
        if save_to_beta_dir:
            fig_beta, ax_beta = plt.subplots()
            ax_beta.plot(times[mask_time], r12_trap_large, '-', lw=LINEWIDTH)
            # ax.set_xlim(T+0.2,T+0.75)
            # ax.set_ylim(2,6)
            ax_beta.set_xlabel(r'$t/T$', fontsize=FONTSIZE)
            ax_beta.set_ylabel(r'$r_{12} $', fontsize=FONTSIZE)
            fig_beta.savefig(f'{save_dir}/particle_drift_after_stop_vs_time.png', bbox_inches='tight', dpi=300)
        
    ax.set_xlabel(r'$t/T$', fontsize=FONTSIZE)
    ax.set_ylabel(r'$r_{12} $', fontsize=FONTSIZE)
    # ax.set_xlim(0, 0.75)
    ax.legend(loc='upper left', frameon=False)
    # fig.savefig('figures/central_particle_azimuth_vs_time_vary_beta.png', bbox_inches='tight', dpi=300)

    plt.show() 

    return

if __name__ == '__main__':
    main(save_to_beta_dir=True)