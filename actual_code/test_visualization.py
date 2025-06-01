import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


class MatrixAnimatorVoxels:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.colorbar = None
        self.colormap = matplotlib.colormaps['viridis']
        self.initialized = False

    def update(self, matrix_3d, threshold=1, index=0):
        self.ax.cla()  # clear the previous voxels

        # Create a mask for voxels to display (non-zero or above a threshold)
        mask = matrix_3d > threshold

        if not mask.any():
            # No voxels to display, just clear and pause
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
            return

        # Map normalized voxel values to colors
        colors = np.zeros((*matrix_3d.shape, 4))  # RGBA array
        colors[mask] = self.colormap(matrix_3d[mask])

        # Plot voxels with facecolors
        self.ax.voxels(mask, facecolors=colors, edgecolor='k')

        # Set limits
        self.ax.set_xlim(0, matrix_3d.shape[0])
        self.ax.set_ylim(0, matrix_3d.shape[1])
        self.ax.set_zlim(0, matrix_3d.shape[2])

        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'3D Voxel Visualization {index}')

        # Add or update colorbar only once
        if self.colorbar is None:
            sm = cm.ScalarMappable(cmap=self.colormap)
            sm.set_array([])
            self.colorbar = self.fig.colorbar(sm, ax=self.ax, pad=0.1)
            self.colorbar.set_label('Voxel Value')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


def visualize_voxel_space(voxel_space):
    # Threshold to select voxels
    mask = voxel_space > 1

    # Normalize values for colormap
    norm = Normalize(vmin=voxel_space[mask].min(), vmax=voxel_space[mask].max())
    colormap = matplotlib.colormaps['viridis'] # Or cm.get_cmap('viridis')

    # Create RGBA color array
    colored_voxels = np.zeros((*voxel_space.shape, 4))  # (X, Y, Z, 4)
    colored_voxels[mask] = colormap(norm(voxel_space[mask]))

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(mask, facecolors=colored_voxels, edgecolor='k')

    ax.set_xlim([0, voxel_space.shape[0]])
    ax.set_ylim([0, voxel_space.shape[1]])
    ax.set_zlim([0, voxel_space.shape[2]])

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Needed only to silence warnings
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Voxel Value')

    plt.show()


# Example usage:
# if __name__ == "__main__":
#     animator = MatrixAnimatorVoxels()
#     shape = (20, 20, 20)
#
#     for _ in range(50):
#         # Random 3D matrix with some values
#         data = np.random.rand(*shape)
#         animator.update(data)