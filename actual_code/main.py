import pycuda.driver as cuda
import pycuda.autoinit  # if you dont import this it breaks :)
from pycuda.compiler import SourceModule
import numpy as np
import cv2
from test_visualization import MatrixAnimatorVoxels


with open("kernel.cu", "r") as f:
    kernel_code = f.read()

mod = SourceModule(kernel_code)
process_image = mod.get_function("process_image")


def resize_frame(frame, amount: float):
    return cv2.resize(frame, (int(frame.shape[1] * amount), int(frame.shape[0] * amount)))


def get_voxel_space_from_images(image_1, image_2, camera_data_1, camera_data_2):
    height, width = image_1.shape

    joined_images = np.vstack([image_1, image_2]).astype(np.int32)
    joined_images = joined_images.reshape(-1)

    img_gpu = cuda.mem_alloc(joined_images.nbytes)
    cuda.memcpy_htod(img_gpu, joined_images)

    # CAMERA POSITION/ROTATION/FOV DATA
    joined_camera_data = np.array(camera_data_1 + camera_data_2).astype(np.int32)

    camera_data_gpu = cuda.mem_alloc(joined_camera_data.nbytes)
    cuda.memcpy_htod(camera_data_gpu, joined_camera_data)

    # VOXEL SPACE
    voxel_space_x_dim, voxel_space_y_dim, voxel_space_z_dim = 25, 25, 25
    voxel_space_unit = 40  # one voxel space cube is 40x40x40 in 'world' dimensions
    voxel_space = np.zeros(shape=(voxel_space_x_dim, voxel_space_y_dim, voxel_space_z_dim)).astype(
        np.int32)  # initialize the voxel space as a 3d matrix filled with 0

    voxel_space_gpu = cuda.mem_alloc(voxel_space.nbytes)
    cuda.memcpy_htod(voxel_space_gpu, voxel_space)

    # KERNEL LAUNCH SETTINGS
    block_size = (16, 16, 1)  # block size, can update as long as TPD < 1024
    grid_size = ((width + 15) // 16, (height + 15) // 16, 2)  # the z index makes us able to pass multiple images

    process_image(img_gpu, np.int32(width), np.int32(height), camera_data_gpu, voxel_space_gpu,
                  np.int32(voxel_space_x_dim), np.int32(voxel_space_y_dim), np.int32(voxel_space_z_dim),
                  np.int32(voxel_space_unit), block=block_size, grid=grid_size)

    # Copy result back
    result = np.empty_like(voxel_space)
    cuda.memcpy_dtoh(result, voxel_space_gpu)

    return result


if __name__ == "__main__":
    visualization = MatrixAnimatorVoxels()

    camera_1_data = [1000, 1, 150, 0, -65, 0, 60]
    camera_2_data = [0, 1, 0, 0, 60, 0, 60]

    cap_1 = cv2.VideoCapture("../recordings/5/cam_1.mp4")
    cap_2 = cv2.VideoCapture("../recordings/5/cam_2.mp4")

    ret_1, prev_1 = cap_1.read()
    ret_2, prev_2 = cap_2.read()

    if not ret_1 or not ret_2:
        exit(-1)

    frame = 0
    while True:

        ret_1, frame_1 = cap_1.read()
        ret_2, frame_2 = cap_2.read()

        if not ret_1 or not ret_2:
            break

        diff_1 = cv2.absdiff(cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_1, cv2.COLOR_BGR2GRAY))
        diff_2 = cv2.absdiff(cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_2, cv2.COLOR_BGR2GRAY))

        prev_1, prev_2 = frame_1, frame_2

        cv2.imshow("diff_1", resize_frame(diff_1, 0.5))
        cv2.imshow("diff_2", resize_frame(diff_2, 0.5))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        result = get_voxel_space_from_images(diff_1, diff_2, camera_1_data, camera_2_data)

        # result = result > 1

        visualization.update(result, threshold=1, index=frame)
        frame += 1

    cap_1.release()
    cap_2.release()
    cv2.destroyAllWindows()
