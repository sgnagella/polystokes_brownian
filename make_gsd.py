import os
import numpy as np
import pickle
import gsd.hoomd

def main(beta=1):
    
    data_save_dir = f'data/beta_{beta}'
    with open(os.path.join(data_save_dir, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)

    data_config_dir = os.path.join(data_save_dir, 'config')
    os.chdir(data_config_dir)
    print(f"Current working directory: {os.getcwd()}")
    files_config = [f for f in os.listdir() if os.path.isfile(f)]
    files_positions = [f for f in files_config if 'config' in f]
    files_quats = [f for f in files_config if 'quats' in f]
    
    # Print the keys
    for key in params.keys():
        print(key)
        
    Np = params['Np']
    beta = params['beta']
    N_poly = params['N_poly']
    N_mono = params['N_mono']
    N_mono_total = N_poly * N_mono
    bond_ids  = np.asarray(params['bond_ids']).T
    
    # Sort the files by the number in the filename
    sort_func = lambda x: float(x.split('_')[1])
    files_positions.sort(key=sort_func)
    files_quats.sort(key=sort_func)
    
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
    
    # for file in files_config:
    #     with open(file, 'r') as f:
    #         lines = f.readlines()
    #         particle_pos = []
    #         for line in lines:
    #             particle_pos.append([float(i) for i in line.split()])
    #         traj.append(particle_pos)
            
    # for file in files_quats:
    #     with open(file, 'r') as f:
    #         lines = f.readlines()
    #         particle_quat = []
    #         for line in lines:
    #             particle_quat.append([float(i) for i in line.split()])
    #         quats.append(particle_quat)
    
    traj = read_files(files_positions)
    quats = read_files(files_quats)
                
    trajectory = np.array(traj)
    quats = np.array(quats)
    __, n_quats, __ = quats.shape
    pad_no = Np - n_quats
    if pad_no > 0:
        pad_quat = np.zeros((trajectory.shape[0], pad_no, 4))
        pad_quat[:,:,0] = 1
        quats = np.concatenate([pad_quat, quats], axis=1)
        
    n_frames, n_particles, __ = trajectory.shape
    print(f"Trajectory shape: {trajectory.shape}")
    
    # Save the particle positions and quaternions to npy files
    np.save(f"../trajectory.npy", trajectory)
    np.save(f"../orientations.npy", quats)
    
    output_file = f"../trajectory.gsd"
    diameters = 2*np.ones(n_particles)
    typeids = np.zeros(n_particles, dtype=int)
    # types = ['A'] * n_particles # GSD < 2.9.0
    types = ['A', 'B'] # GSD > 2.9.0
    
    diameters[:N_mono_total] *= beta
    # types[:N_mono_total] = ['B'] * N_mono_total # GSD < 2.9.0
    typeids[:N_mono_total] = 1
    
    L = 60
    assert 0.5*L > np.max(np.abs(trajectory)), f"L/2={0.5*L} is smaller than the maximum particle position {np.max(np.abs(trajectory))}"
    
    box_vector = [L, L, L, 0, 0, 0]

    with gsd.hoomd.open(name=output_file, mode='wb') as traj:
        for frame_idx in range(n_frames):
            snap = gsd.hoomd.Snapshot()
            snap.configuration.box = box_vector
            snap.particles.N = n_particles
            snap.particles.position = trajectory[frame_idx]
            snap.particles.orientation = quats[frame_idx]
            snap.particles.diameter = diameters
            snap.particles.types = types
            snap.particles.typeid = typeids
            snap.bonds.N = bond_ids.shape[0]
            snap.bonds.group = bond_ids
            snap.bonds.types = ["bond"]
            traj.append(snap)
    
    print(f"Trajectory saved to {os.path.abspath(output_file)}")
    return

if __name__ == '__main__':
    beta = 0.1
    main(beta=beta)