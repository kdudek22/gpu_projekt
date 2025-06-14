import numpy as np
import cv2
import time
from flask import Flask, jsonify, request
import threading
import matplotlib.pyplot as plt


app = Flask(__name__)
result = []


def resize_image(image, amount: float):
    return cv2.resize(image, (int(image.shape[1] * amount), int(image.shape[0] * amount)))


def run_cuda():
    # this is the way this has to be done, as we have to set up cuda in the same thread the kernel invocation is done
    import pycuda.driver as cuda
    import pycuda.autoinit  # if you dont import this it breaks :)
    from pycuda.compiler import SourceModule

    with open("kernel.cu", "r") as f:
        kernel_code = f.read()

    mod = SourceModule(kernel_code)
    process_image = mod.get_function("process_image")

    def get_voxel_space_from_images(images: list[np.array], camera_data: list[int]):
        # all images must have the same dimensions: width and height
        height, width = images[0].shape

        # IMAGES
        joined_images = np.vstack([i for i in images]).astype(np.int32).reshape(-1)

        img_gpu = cuda.mem_alloc(joined_images.nbytes)
        cuda.memcpy_htod(img_gpu, joined_images)

        # CAMERA DATA
        joined_camera_data = np.array(camera_data).astype(np.int32)

        camera_data_gpu = cuda.mem_alloc(joined_camera_data.nbytes)
        cuda.memcpy_htod(camera_data_gpu, joined_camera_data)

        # VOXEL SPACE
        voxel_space_x_dim, voxel_space_y_dim, voxel_space_z_dim = 25, 25, 25
        voxel_space_unit = 40  # one voxel space cube is 40x40x40 in 'world' dimensions
        voxel_space = np.zeros(shape=(voxel_space_x_dim, voxel_space_y_dim, voxel_space_z_dim)).astype(np.int32)

        voxel_space_gpu = cuda.mem_alloc(voxel_space.nbytes)
        cuda.memcpy_htod(voxel_space_gpu, voxel_space)

        block_size = (16, 16, 1)  # block size, can update as long as TPD < 1024
        grid_size = (
        (width + 15) // 16, (height + 15) // 16, len(images))  # the z index makes us able to pass multiple images

        process_image(img_gpu, np.int32(width), np.int32(height), camera_data_gpu, voxel_space_gpu,
                      np.int32(voxel_space_x_dim), np.int32(voxel_space_y_dim), np.int32(voxel_space_z_dim),
                      np.int32(voxel_space_unit), block=block_size, grid=grid_size)

        # Copy result back
        result = np.empty_like(voxel_space)
        cuda.memcpy_dtoh(result, voxel_space_gpu)

        return result

    def process_images():
        camera_1_data = [500, 100, 0, 0, 0, 0, 60]
        camera_2_data = [500, 100, 1000, 0, 180, 0, 60]
        camera_3_data = [500, 1000, 500, 90, 0, 0, 60]

        cap_1 = cv2.VideoCapture("../recordings/11/cam_1.mp4")
        cap_2 = cv2.VideoCapture("../recordings/11/cam_2.mp4")
        cap_3 = cv2.VideoCapture("../recordings/11/cam_3.mp4")

        ret_1, frame_1 = cap_1.read()
        ret_2, frame_2 = cap_2.read()
        ret_3, frame_3 = cap_3.read()

        if not ret_1 or not ret_2 or not ret_3:
            exit(-1)

        while True:
            ret_1, frame_1 = cap_1.read()
            ret_2, frame_2 = cap_2.read()
            ret_3, frame_3 = cap_3.read()

            if not ret_1 or not ret_2 or not ret_3:
                break

            frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
            frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
            frame_3 = cv2.cvtColor(frame_3, cv2.COLOR_BGR2GRAY)

            global result
            result = get_voxel_space_from_images([frame_1, frame_2, frame_3], camera_1_data + camera_2_data + camera_3_data)

            cv2.imshow("cam_1", resize_image(frame_1, 0.5))
            cv2.imshow("cam_2", resize_image(frame_2, 0.5))
            cv2.imshow("cam_3", resize_image(frame_3, 0.5))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap_1.release()
        cap_2.release()
        cap_3.release()

    while True:
        process_images()


@app.route("/test")
def test_view():
    return "test works :)"


@app.route("/voxel_space")
def get_voxel_space():
    threshold = int(request.args.get("threshold", 0))
    x, y, z = np.where(result > threshold)
    values = result[x, y, z]

    voxel_data = [{"x": int(x_), "y": int(y_), "z": int(z_), "value": int(val)} for x_, y_, z_, val in zip(x, y, z, values)]

    return jsonify(voxel_data)


if __name__ == "__main__":
    t = threading.Thread(target=run_cuda)
    t.start()
    app.run(debug=True, use_reloader=False)
