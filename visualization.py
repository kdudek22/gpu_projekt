import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pickle


def visualize_voxels_plt(voxel_data: list[list[list]]):
    data = np.array(voxel_data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(data, facecolors='red', edgecolor='k')
    plt.show()

def animate_voxels_plt(voxel_sequence: list[list[list[list]]], interval=0.03):
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

def visualize_voxels_pv(voxel_data):
    data = np.array(voxel_data, dtype=np.uint8)
    
    grid = pv.ImageData(dimensions=data.shape)
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    grid.point_data["values"] = data.flatten(order="F")  # fortran order!

    # PyVista automatycznie renderuje voxele, gdzie wartości są !=0
    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="values", cmap="coolwarm", opacity="linear")
    plotter.show()

def random_frame(shape = (20, 20, 20), percent_of_ones: float = 1.0):
    size = np.prod(shape)
    liczba_jedynek = int(size * percent_of_ones/100.0)

    frame = np.zeros(size, dtype=int)

    ids_ones = np.random.choice(size, liczba_jedynek, replace=False)
    frame[ids_ones] = 1

    frame = frame.reshape(shape)
    return frame

def animate_voxels_pv(voxel_sequence, filepath="animation.gif", interval=0.3):
    voxel_sequence = [np.array(frame, dtype=np.uint8) for frame in voxel_sequence]

    shape = voxel_sequence[0].shape

    grid = pv.ImageData(dimensions=shape)
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)

    plotter = pv.Plotter()
    plotter.open_gif(filepath)

    # Dodajemy pierwszy frame (inaczej pusty)
    actor = plotter.add_volume(grid, scalars=voxel_sequence[0].flatten(order="F"),
                            cmap="coolwarm", opacity="linear")

    # animacja
    for i, frame_data in enumerate(voxel_sequence):
        grid.point_data["values"] = frame_data.flatten(order="F")
        actor.mapper.scalar_range = (0, 1)  # skalowanie
        plotter.add_text(f"Frame {i+1}", position="upper_left", font_size=10, color="black")
        plotter.write_frame()  # Zapisz frame do GIFa
        plotter.clear_actors()  # Usuń stare
        actor = plotter.add_volume(grid, scalars=frame_data.flatten(order="F"),
                                cmap="coolwarm", opacity="linear")

    plotter.close()


if __name__ == "__main__":
    with open('actual_code/res_arr.pkl', 'rb') as f:
        loaded_array = pickle.load(f)

    frame = random_frame(shape=(100, 100, 100), percent_of_ones=0.01)

    sequence = [
        frame,
        np.roll(frame, shift=1, axis=2).tolist(), #ta klatka to po prostu frame, ale przesunięta
        np.roll(frame, shift=2, axis=2).tolist(),
        np.roll(frame, shift=3, axis=2).tolist(),
        np.roll(frame, shift=4, axis=2).tolist(),
        np.roll(frame, shift=5, axis=2).tolist(),
        np.roll(frame, shift=6, axis=2).tolist()
    ]

    folder_path = os.path.join(os.getcwd(), "animations")
    animations_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    animate_voxels_plt(loaded_array)