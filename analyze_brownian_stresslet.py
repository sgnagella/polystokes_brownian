import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os

def read_files(files):
        traj = []
        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
                particle_data = []
                for line in lines:
                    particle_data.append([float(i) for i in line.split()])
                traj.append(particle_data)

        return traj

beta = 0.1
fene = False

def main(): 
    data_dir = f'data/dumbbells_thermal_beta_{beta}_fene_{False}'
    with open(os.path.join(data_dir, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)

    data_config_dir = os.path.join(data_dir, 'config')
    os.chdir(data_config_dir)
    print(f"Current working directory: {os.getcwd()}")

    files_config = [f for f in os.listdir() if os.path.isfile(f)]
    files_brownian = [f for f in files_config if 'brownian_stresslet' in f]

    # Sort the files by the number in the filename
    sort_func = lambda x: float(x.split('_')[2])
    files_brownian.sort(key=sort_func)

    # Read the files and store the data in a numpy array
    stresslet = np.array(read_files(files_brownian)).reshape(len(files_brownian), 5)

    print(f"Brownian stresslet data with shape {stresslet.shape} loaded from {data_config_dir}")

    # Stresslets were computed as 
    # S1 = Sxx - Szz
    # S2 = Sxy = Syx
    # S3 = Sxz = Szx
    # S4 = Syz = Szy
    # S5 = Syy - Szz

    # Plot the time-series of stresslet components
    plt.figure(figsize=(10, 6))
    plt.plot(stresslet[:, 0], label='S1 = Sxx - Szz')
    # plt.plot(stresslet[:, 1], label='S2 = Sxy = Syx')
    # plt.plot(stresslet[:, 2], label='S3 = Sxz = Szx')
    # plt.plot(stresslet[:, 3], label='S4 = Syz = Szy')
    plt.plot(stresslet[:, 4], label='S5 = Syy - Szz')
    
    plt.xlabel('Time step')
    plt.ylabel('Stresslet components')
    plt.title('Brownian Stresslet Components Time Series')
    plt.legend()
    # plt.savefig(os.path.join(data_dir, 'brownian_stresslet_time_series.png'))
    plt.show()

    mean_isotropic = np.mean(stresslet[:, 0] + stresslet[:, 4]) / 2
    print(f"Mean isotropic stresslet: {mean_isotropic}")

    mean_shear = np.mean(stresslet[:, 1:4], axis=0)
    print(f"Mean shear stresslet components: {mean_shear}")
    return

if __name__ == "__main__":
    main()