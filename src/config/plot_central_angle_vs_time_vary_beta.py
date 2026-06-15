"""
    Script to analyze simulation results from 3D_SD_make_trajectory.py
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

def main(T=2.0, save_to_beta_dir=False):
    """
        T indicates the period of revolution of the moving particle 
    """
    
    def quat2azimuth(q):
        """
        Convert quaternion to azimuthal angle (0 to 2pi)
        """
        q0, q1, q2, q3 = q
        phi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
        return phi
    
    betas = [0.1]; [1, 0.3, 0.1]; [0.1,0.5,1]
    fig, ax = plt.subplots()
    
    for beta in betas:
        os.chdir('config')
        save_dir = f'data/beta_{beta}'
        fig_save_dir = f'figures/beta_{beta}'
        os.makedirs(fig_save_dir, exist_ok=True)
        
        params = pickle.load(open(f'{save_dir}/params.pkl', 'rb'))
        N_trapped = params['N_trapped'] 
        N_poly = params['N_poly']
        N_mono = params['N_mono']
        dt = 1e-3; 1e-2; 
        # params['dt']
        samplerate = 10; 1;
        # params['samplerate']
        N_mono_total = N_poly * N_mono
        Np = N_trapped + N_mono_total
        beta = params['beta']
    
        # Load the orientations and trajectory
        orientations = np.load(f'{save_dir}/orientations.npy')
        positions = np.load(f'{save_dir}/trajectory.npy')
    
        # Extract the number of frames and particles
        Nframes, Nparticles, _ = orientations.shape
        assert Nparticles == Np, f"Number of particles in trajectory ({Nparticles}) does not match expected number of particles ({Np})"
    
        # Extract the central particle orientation
        central_particle_orientation = orientations[:, -1]
        central_particle_azimuth = np.zeros(Nframes)
        central_particle_orientations = np.zeros((Nframes, 1, 3))
        for i in range(Nframes):
            central_particle_azimuth[i] = quat2azimuth(central_particle_orientation[i].T)
            central_particle_orientations[i] = np.array([np.cos(central_particle_azimuth[i]), np.sin(central_particle_azimuth[i]), 0])
        
        # central_particle_orientations = np.array([np.cos(central_particle_azimuth), np.sin(central_particle_azimuth), np.zeros(Nframes)]).reshape(Nframes, 1, 3)
        central_particle_pos = positions[:, -1].reshape(Nframes, 1, 3)
        conf = np.concatenate((np.zeros((Nframes, 1, 3)), central_particle_orientations), axis=1)
        
        # # Animate the central particle trajectory   
        # anim = sphere_rotation_animation.SphereRotationAnimation(1)
        # anim.animate(conf, Nframes, dt=dt, samplerate=samplerate, save=False, plot_init_config_only=False, interval=75)
        
        # Plot the central particle azimuthal angle as a function of time
        times = np.arange(Nframes) * samplerate * dt
        ax.plot(times, central_particle_azimuth, '-', lw=LINEWIDTH, label=r'$\beta = $' + f'{beta:.2f}')
        
        # Print the % recovery of the angle from maximum displacement by the final timestep recorded
        t_end_record = 1; 0.6
        tol = 1e-6
        stop_displacement = float( central_particle_azimuth[np.abs(times - T)<tol] )
        print(stop_displacement)
        # np.max(central_particle_azimuth[times<t_end_record])
        tend_idx = -1; 
        print(central_particle_azimuth[-1])
        recovery = 100 * float((central_particle_azimuth[tend_idx] - stop_displacement)/stop_displacement )
        print(f'For beta = {beta:.2f}, the central particle azimuthal angle recovered {recovery:.2f}% of the maximum displacement')

        # Save the data
        np.save(f'{save_dir}/central_particle_azimuth_times_T_{T}.npy', times)
        np.save(f'{save_dir}/central_particle_azimuth_T_{T}.npy', central_particle_azimuth)
        
        if save_to_beta_dir:
            fig_beta, ax_beta = plt.subplots()
            ax_beta.plot(times, central_particle_azimuth, '-', lw=LINEWIDTH)
            ax_beta.set_xlabel(r'$t/T$', fontsize=FONTSIZE)
            ax_beta.set_ylabel(r'$\theta $' + ' [rad]', fontsize=FONTSIZE)
            ax_beta.set_xlim(0, 0.75)
            fig_beta.savefig(f'{save_dir}/central_particle_azimuth_vs_time.png', bbox_inches='tight', dpi=300)
        
    ax.set_xlabel(r'$t/T$', fontsize=FONTSIZE)
    ax.set_ylabel(r'$\theta $' + ' [rad]', fontsize=FONTSIZE)
    # ax.set_xlim(0, 0.75)
    ax.legend(loc='upper left', frameon=False)
    fig.savefig('figures/central_particle_azimuth_vs_time_vary_beta.png', bbox_inches='tight', dpi=300)
    
    # Save the data
    np.save(f'{save_dir}/central_particle_azimuth_vs_time.npy', central_particle_azimuth)
    np.save(f'{save_dir}/times.npy', times)
    plt.show() 

    return

if __name__ == '__main__':
    main(save_to_beta_dir=True)