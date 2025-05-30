import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def visualize_voxels(voxel_data: list[list[list]]):
    data = np.array(voxel_data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(data, facecolors='red', edgecolor='k')
    plt.show()

def animate_voxels(voxel_sequence: list[list[list[list]]], interval=500):
    voxel_sequence = [np.array(frame) for frame in voxel_sequence]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_index):
        ax.clear()
        ax.voxels(voxel_sequence[frame_index], facecolors='red', edgecolor='k')
        ax.set_title(f"Frame {frame_index + 1}")
        ax.set_xlim(0, voxel_sequence[0].shape[2])
        ax.set_ylim(0, voxel_sequence[0].shape[1])
        ax.set_zlim(0, voxel_sequence[0].shape[0])

    ani = FuncAnimation(fig, update, frames=len(voxel_sequence), interval=interval, repeat=True)
    plt.show()

def random_frame(shape = (20, 20, 20), percernt_of_ones: float = 1.0):
    size = np.prod(shape)
    liczba_jedynek = int(size * percernt_of_ones/100.0)

    frame = np.zeros(size, dtype=int)

    ids_ones = np.random.choice(size, liczba_jedynek, replace=False)
    frame[ids_ones] = 1

    frame = frame.reshape(shape)
    return frame

if __name__ == "__main__":
    #przykładowa wizualizacja i animacja
    frame = random_frame()

    sequence = [
        frame,
        np.roll(frame, shift=1, axis=2).tolist(), #ta klatka to po prostu frame, ale przesunięta
        np.roll(frame, shift=2, axis=2).tolist(),
        np.roll(frame, shift=3, axis=2).tolist(),
        np.roll(frame, shift=4, axis=2).tolist(),
        np.roll(frame, shift=5, axis=2).tolist(),
        np.roll(frame, shift=6, axis=2).tolist()
    ]

    visualize_voxels(frame)
    animate_voxels(sequence, interval=500)