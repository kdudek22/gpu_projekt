import numpy as np
import cv2
from flask import Flask, jsonify, request
import threading
import json


app = Flask(__name__)
result = []


def resize_image(image, amount: float):
    return cv2.resize(image, (int(image.shape[1] * amount), int(image.shape[0] * amount)))


def get_camera_data(file_path: str) -> list[int]:
    with open(file_path, "r") as f:
        content = json.load(f)

    return [value for cam in content.values() for value in cam.values()]


def run_cuda(num_cameras: int):
    print("Running Setup")
    # this is the way this has to be done, as we have to set up cuda in the same thread the kernel invocation is done
    import pycuda.driver as cuda
    import pycuda.autoinit  # if you dont import this it breaks :)
    from pycuda.compiler import SourceModule

    with open("kernel.cu", "r") as f:
        kernel_code = f.read()

    mod = SourceModule(kernel_code)
    process_image = mod.get_function("process_image")

    i = 0

    def get_voxel_space_from_images(images: list[np.array], camera_data: list[int]):
        nonlocal i
        i += 1

        if(i == 100):
            x  = 123

        print(i)

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
        camera_data = get_camera_data("../recordings/11/locations.json")

        caps = [cv2.VideoCapture(f"../recordings/11/cam_{i + 1}.mp4") for i in range(num_cameras)]

        rets, frames = zip(*[cap.read() for cap in caps])

        if not all(rets):
            exit(-1)

        while True:
            rets, frames = zip(*[cap.read() for cap in caps])

            if not all(rets):
                return

            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

            global result
            result = get_voxel_space_from_images(frames, camera_data)

            # displaying the video in real time as the algorith is running
            for i, frame in enumerate(frames):
                cv2.namedWindow(f'cam_{i + 1}', cv2.WINDOW_NORMAL)
                cv2.imshow(f"cam_{i + 1}", resize_image(frame, 0.5))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return -1

        for cap in caps:
            cap.release()

        cv2.destroyAllWindows()

    while True:
        print("Running main loop again...")
        res = process_images()

        if res == -1:
            break


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
    # run_cuda(4)
    t = threading.Thread(target=run_cuda, args=(4,))
    t.start()
    app.run(debug=True, use_reloader=False)
