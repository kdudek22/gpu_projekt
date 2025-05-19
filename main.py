import cv2
from utils import get_rotation_matrix, create_camera_frustum, create_visualization_lines, Camera
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


if __name__ == "__main__":
    camera_1 = Camera([2000, 1, 0], [-25, -90, 0], 60)
    image_1 = cv2.imread("newest_cam_1/image_0330.png", cv2.IMREAD_GRAYSCALE)
    line_set_1 = create_visualization_lines(image_1, camera_1)

    camera_2 = Camera([-2000, 1, 0], [-25, 90, 0], 60)
    image_2 = cv2.imread("newest_cam_2/image_0330.png", cv2.IMREAD_GRAYSCALE)
    line_set_2 = create_visualization_lines(image_2, camera_2)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([line_set_1, line_set_2, axis])
