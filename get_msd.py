import numpy as np 
import matplotlib.pyplot as plt
import gsd.hoomd

def getMSDTensor(r):
    """
    Computes the msd tensor for a time series of positions r(t) of a given particle.
    r: ndarray (Nframes, 3)
    """
    # Uses window averaging
    shifts = np.arange(len(r))
    msds = np.zeros((shifts.size, 6))

    for i, shift in enumerate(shifts):
        progress = (i+1)/len(shifts)*100
        if i % 100 == 0:
            print(f"Computing MSD tensor: {progress:.2f}% complete")
        diffs = r[:-shift if shift else None] - r[shift:]
        diffs_x = diffs[:, 0]
        diffs_y = diffs[:, 1]
        diffs_z = diffs[:, 2]
        msds[i,0] = np.mean(diffs_x**2)
        msds[i,1] = np.mean(diffs_y**2)
        msds[i,2] = np.mean(diffs_z**2)
        msds[i,3] = np.mean(diffs_x*diffs_y)
        msds[i,4] = np.mean(diffs_x*diffs_z)
        msds[i,5] = np.mean(diffs_y*diffs_z)
    return msds

beta = 0.01
fene = False
input_file = f"data/test_dumbbells_thermal_beta_{beta}_fene_{fene}/trajectory.gsd"

def main(): 
    # Load the trajectory of the last particle
    with gsd.hoomd.open(name=input_file, mode='r') as traj:
        traj_colloid = np.array([frame.particles.position[-1] for frame in traj])

    print(f"Trajectory of last particle with shape {traj_colloid.shape} loaded from {input_file}")

    # Compute the MSD tensor
    msd_tensor = getMSDTensor(traj_colloid)

    # Save the MSD tensor to a file
    output_file = f"data/test_dumbbells_thermal_beta_{beta}_fene_{fene}/msd_tensor.npy"
    np.save(output_file, msd_tensor)
    print(f"MSD tensor saved to {output_file}")

    # Plot the MSD tensor components
    plt.figure(figsize=(10, 6))
    plt.plot(msd_tensor[:, 0], label='MSD_xx')
    plt.plot(msd_tensor[:, 1], label='MSD_yy')
    plt.plot(msd_tensor[:, 2], label='MSD_zz')
    # plt.plot(msd_tensor[:, 3], label='MSD_xy')
    # plt.plot(msd_tensor[:, 4], label='MSD_xz')
    # plt.plot(msd_tensor[:, 5], label='MSD_yz')
    plt.xlabel('Time lag')
    plt.ylabel('MSD tensor components')
    plt.title('Mean Squared Displacement Tensor Components')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(f"data/test_dumbbells_thermal_beta_{beta}_fene_{fene}/msd_tensor_plot.png")
    plt.show()
    return


if __name__ == "__main__":
    main()
