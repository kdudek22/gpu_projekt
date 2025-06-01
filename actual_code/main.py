import pycuda.driver as cuda
import pycuda.autoinit  # if you dont import this it breaks :)
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.cm as cm
import matplotlib.colors as colors

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


if __name__ == "__main__":

    with open("kernel.cu", "r") as f:
        kernel_code = f.read()

    mod = SourceModule(kernel_code)
    process_image = mod.get_function("process_image")

    image_1 = cv2.imread("../recordings/4/frame_diffs/cam_1/image_150.png", cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread("../recordings/4/frame_diffs/cam_2/image_150.png", cv2.IMREAD_GRAYSCALE)
    # constant memory ?
    camera_1_data = [1000, 1, 150, 0, -60, 0, 60]
    camera_2_data = [0, 1, 0, -10, 90, 0, 60]

    joined_camera_data = np.array(camera_1_data + camera_2_data).astype(np.int32)

    height, width = image_1.shape

    joined_images = np.vstack([image_1, image_2]).astype(np.int32)

    joined_images = joined_images.reshape(-1)


    img_gpu = cuda.mem_alloc(joined_images.nbytes)
    cuda.memcpy_htod(img_gpu, joined_images)

    camera_data_gpu = cuda.mem_alloc(joined_camera_data.nbytes)
    cuda.memcpy_htod(camera_data_gpu, joined_camera_data)

    # VOXEL SPACE
    voxel_space_x_dim, voxel_space_y_dim, voxel_space_z_dim = 25, 25, 25

    voxel_space = np.zeros(shape=(voxel_space_x_dim, voxel_space_y_dim, voxel_space_z_dim)).astype(np.int32)
    voxel_space_gpu = cuda.mem_alloc(voxel_space.nbytes)
    cuda.memcpy_htod(voxel_space_gpu, voxel_space)
    voxel_space_unit = 40

    # KERNEL LAUNCH SETTINGS
    block_size = (16, 16, 1)  # block size, can update as long as TPD < 1024
    grid_size = ((width + 15)//16, (height + 15)//16, 2)  # the z index makes us able to pass multiple images

    process_image(img_gpu, np.int32(width), np.int32(height), camera_data_gpu, voxel_space_gpu,
                  np.int32(voxel_space_x_dim), np.int32(voxel_space_y_dim), np.int32(voxel_space_z_dim),
                  np.int32(voxel_space_unit), block=block_size, grid=grid_size)

    # Copy result back
    result = np.empty_like(voxel_space)
    cuda.memcpy_dtoh(result, voxel_space_gpu)

    # Threshold to select voxels
    mask = result > 1

    # Normalize values for colormap
    norm = Normalize(vmin=result[mask].min(), vmax=result[mask].max())
    colormap = matplotlib.colormaps['viridis'] # Or cm.get_cmap('viridis')

    # Create RGBA color array
    colored_voxels = np.zeros((*result.shape, 4))  # (X, Y, Z, 4)
    colored_voxels[mask] = colormap(norm(result[mask]))

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(mask, facecolors=colored_voxels, edgecolor='k')

    ax.set_xlim([0, result.shape[0]])
    ax.set_ylim([0, result.shape[1]])
    ax.set_zlim([0, result.shape[2]])

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Needed only to silence warnings
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Voxel Value')

    plt.show()