"""
    Script to analyze simulation results from 3D_SD_make_trajectory.py
    Plots the relaxation of the central particle's polar angle after the moving particle stops
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

def main():
    
    betas = [0.1]; [1, 0.3, 0.1]; [0.1,0.5,1]
    beta = 0.1
    periods = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    fig, ax = plt.subplots()
    
    os.chdir('config')
    save_dir = f'data/beta_{beta}'
    
    cmap = cm.get_cmap('plasma', len(periods))
    disps = []
    
    for idx, T in enumerate(periods):
        base = 'central_particle_azimuth'
        times_file = f'{save_dir}/{base}_times_T_{T}.npy'
        data_file = f'{save_dir}/{base}_T_{T}.npy'
        times = np.load(times_file)
        angles = np.load(data_file)

        times_relax = times[times>T]
        times_relax -= times_relax[0]
        angles_relax = angles[times>T]
        angles_relax -= angles_relax[0]
        
        th01 = float( angles_relax[np.abs(times_relax-0.01)<1e-6] )
        print(th01)
        disp = angles_relax - th01
        disps.append( disp[-1] ) 
                
        ax.plot(
            times_relax, 
            angles_relax, 
            '-', 
            lw=LINEWIDTH,
            label=f'T={T}', 
            color=cmap(idx)
        )
        
    ax.set_xlabel(r'$t/T$', fontsize=FONTSIZE)
    ax.set_ylabel(r'$\theta $' + ' [rad]', fontsize=FONTSIZE)
    # ax.set_xlim(0, 0.75)
    ax.legend(loc='upper left', fontsize=11, frameon=False)
    
    disps = np.asarray(disps)
    fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax1.plot(
        periods, 
        disps, 
        'ok', 
        ms=MARKERSIZE, 
        mfc='none', 
        markeredgewidth=MARKEREDGEWIDTH, 
    )
    
    ax1.set_xlabel('Winding Time')
    ax1.set_ylabel(r'$\theta $ ' + 'Recovery after ' r't=0.5')
    
    
    fig.savefig('figures/central_particle_azimuth_relax_vary_period.png', bbox_inches='tight', dpi=300)
    fig1.savefig('figures/central_particle_azimuth_recovery_vs_winding.png', bbox_inches='tight', dpi=300)
    plt.show() 

    return

if __name__ == '__main__':
    main()