import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import os

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

class SphereRotationAnimation: 
    def __init__(self, n_particles, n_mono=0, beta=1):
        """
        A class to generate spheres with half-filled circles and rotating them
        """ 
        self.Np = n_particles
        self.Nm = n_mono # number of monomers
        self.beta = beta # size ratio monomers to colloids
        self.twopi = 2 * np.pi

    def plot_config(self, conf: np.ndarray, bond_ids=None, save=None): 
        """
        Given a configuration of particles, plot the configuration

        Parameters:
        conf: np.ndarray (2*Np, 3): Configuration of particles
        bond_ids: np.ndarray (N_bonds, 2): Array of bond ids. Default is None
        save: str: Path to save the plot. Default is None
        """

        assert conf.shape[0] == 2 * self.Np, "Error: Number of particles in configuration does not match the number of particles in the class"
        assert conf.shape[1] == 3, "Error: Number of dimensions in configuration does not match the number of dimensions in the class"

        # Get maximum position (x, y) in the configuration array
        max_pos = np.max(np.abs(conf[:, :2]))
        lim = max_pos + 5

        circles = []
        points = []
        center_pts = []

        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        for ii in range(self.Np): 
            # Create circular outlines
            x, y = conf[ii, 0], conf[ii, 1]
            radius = self.beta if ii < self.Nm else 1
            circle = patches.Circle((x, y), radius, edgecolor='black', facecolor='none', linewidth=2)
            ax.add_patch(circle)
            circles.append(circle)
            
            # Plot a line pointing from the center of the circle to denote orientation
            orientation = radius*conf[ii + self.Np, :2]
            center_pt, = ax.plot([x], [y], 'ko', markersize=3, linestyle='None', zorder=100)
            pt, = ax.plot([x, x + orientation[0]], [y, y + orientation[1]], 'k-', linewidth=2)
            points.append(pt)
            center_pts.append(center_pt)
        
        # Draw lines connecting bonded particles
        if bond_ids is not None:
            for bond in bond_ids:
                x1, y1 = conf[bond[0], :2]
                x2, y2 = conf[bond[1], :2]
                ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, zorder=0)
                
        if save is not None:
            plt.savefig(save, dpi=200, bbox_inches='tight')
            
        plt.show()

        return

    def animate(self, conf: np.ndarray, n_frames, dt, samplerate, save=None, plot_init_config_only=False, bond_ids=None, interval=300): 
        """
        Given a configuration of particles, animate the rotation of spheres
        
        Parameters:
        conf: np.ndarray (n_frames, 2*Np, 3): Time-dependent configuration of particles
        """

        assert conf.shape[1] == 2 * self.Np, "Error: Number of particles in configuration does not match the number of particles in the class"
        assert conf.shape[0] == n_frames, "Error: Number of frames in configuration does not match the number of frames in the class"
        assert conf.shape[2] == 3, "Error: Number of dimensions in configuration does not match the number of dimensions in the class"
        
        # Get maximum position (x, y) in the configuration array
        max_pos = np.max(np.abs(conf[:, :2, :2]))
        lim = max_pos + 5

        circles = []
        points = []
        init_points = []
        center_pts = []

        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Create a text object for the timer
        timer_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top', fontsize=12, color='black')

        for ii in range(self.Np): 
            # Create circular outlines
            x, y = conf[0, ii, 0], conf[0, ii, 1]
            # print(f"Initial position of particle {ii}: {x, y}")
            radius = self.beta if ii < self.Nm else 1
            orientation = radius*conf[0, ii + self.Np, :2]
            circle = patches.Circle((x, y), radius, edgecolor='black', facecolor='none', linewidth=2)
            ax.add_patch(circle)
            circles.append(circle)
            
            # Plot a line pointing from the center of the circle to denote orientation
            center_pt, = ax.plot([x], [y], 'ko', markersize=3, linestyle='None', zorder=100)
            init_pt, = ax.plot([x, x + orientation[0]], [y, y + orientation[1]], 'r-', linewidth=2, zorder=0)
            pt, = ax.plot([x, x + orientation[0]], [y, y + orientation[1]], 'k-', linewidth=2)
            points.append(pt)
            init_points.append(init_pt)
            center_pts.append(center_pt)
            
        # Draw lines connecting bonded particles
        bond_lines = []
        if bond_ids is not None:
            for bond in bond_ids:
                x1, y1 = conf[0, bond[0], :2]
                x2, y2 = conf[0, bond[1], :2]
                bond_line, = ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, zorder=0)
                bond_lines.append(bond_line)
                
        if plot_init_config_only:
            plt.show()
            return

        # Update function for animation
        def update(frame):
            for ii in range(self.Np):
                # Update the position to move the circles
                x = conf[frame, ii, 0]
                y = conf[frame, ii, 1]
                radius = self.beta if ii < self.Nm else 1
                orientation = radius*conf[frame, ii + self.Np, :2]
                circles[ii].set_center((x, y))

                # Update the center points
                center_pts[ii].set_data([x], [y])

                # Update the points denoting orientation
                points[ii].set_data([x, x + orientation[0]], [y, y + orientation[1]])
                

                # Plot the initial orientation
                init_orientation = conf[0, ii + self.Np, :2]
                init_points[ii].set_data([x, x + init_orientation[0]], [y, y + init_orientation[1]])
            
            # Update bond lines
            if bond_ids is not None:
                for jj, bond in enumerate(bond_ids):
                    x1, y1 = conf[frame, bond[0], :2]
                    x2, y2 = conf[frame, bond[1], :2]
                    bond_lines[jj].set_data([x1, x2], [y1, y2])
                        
            # Update the timer
            time = round(frame * samplerate * dt, 2)
            timer_text.set_text(r'$\tau = {}$'.format(time))
            
            return circles + points + init_points + center_pts + bond_lines + [timer_text]

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)

        # Keep the animation reference alive
        self.ani = ani
        if save is not None:
            print('Saving animation...')
            if os.path.exists('figures'):   
                ani.save(f'figures/{save}', writer='pillow', dpi=200)
            else:
                ani.save(save, writer='pillow', dpi=200)

        plt.show()
