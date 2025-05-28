import pycuda.driver as cuda
import pycuda.autoinit  # if you dont import this it breaks :)
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


if __name__ == "__main__":

    with open("kernel.cu", "r") as f:
        kernel_code = f.read()

    mod = SourceModule(kernel_code)
    process_image = mod.get_function("process_image")

    image_1 = cv2.imread("cam_1_30.png", cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread("cam_2_030.png", cv2.IMREAD_GRAYSCALE)
    # constant memory ?
    camera_1_data = [1000, 1, 0, 0, -90, 0, 60]
    camera_2_data = [0, 1, 0, 0, 90, 0, 60]

    joined_camera_data = np.array(camera_1_data + camera_2_data).astype(np.int32)

    width, height = image_1.shape

    joined_images = np.stack([image_1, image_2], axis=0)
    flattened_images = joined_images.reshape(-1).astype(np.int32)

    joined_images = joined_images.reshape((height * 2, width))

    img_gpu = cuda.mem_alloc(joined_images.nbytes)
    cuda.memcpy_htod(img_gpu, joined_images)

    camera_data_gpu = cuda.mem_alloc(joined_camera_data.nbytes)
    cuda.memcpy_htod(camera_data_gpu, joined_camera_data)

    # VOXEL SPACE
    voxel_space_x_dim, voxel_space_y_dim, voxel_space_z_dim = 1000, 1000, 1000

    voxel_space = np.zeros(shape=(voxel_space_x_dim, voxel_space_y_dim, voxel_space_z_dim)).astype(np.int32)
    voxel_space_gpu = cuda.mem_alloc(voxel_space.nbytes)
    cuda.memcpy_htod(voxel_space_gpu, voxel_space)
    voxel_space_unit = 1

    # KERNEL LAUNCH SETTINGS
    block_size = (16, 16, 1)  # block size, can update as long as TPD < 1024
    grid_size = ((width + 15)//16, (height + 15)//16, 2)  # the z index makes us able to pass multiple images

    process_image(img_gpu, np.int32(width), np.int32(height), camera_data_gpu, voxel_space_gpu,
                  np.int32(voxel_space_x_dim), np.int32(voxel_space_y_dim), np.int32(voxel_space_z_dim),
                  np.int32(voxel_space_unit), block=block_size, grid=grid_size)

    # Copy result back
    result = np.empty_like(voxel_space)
    cuda.memcpy_dtoh(result, voxel_space_gpu)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get coordinates of non-zero values
    x, y, z = np.nonzero(result > 0.5)

    ax.scatter(x, y, z)
    ax.set_xlim([0, voxel_space_x_dim])
    ax.set_ylim([0, voxel_space_y_dim])
    ax.set_zlim([0, voxel_space_z_dim])
    plt.show()
