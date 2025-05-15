import cv2
from utils import get_rotation_matrix, create_camera_frustum, create_visualization_lines, Camera
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d




if __name__ == "__main__":

    camera_1 = Camera([-1000, 1, 1000], [-20, 160, 0], 60)
    image_1 = cv2.imread("images/image_300.jpg", cv2.IMREAD_GRAYSCALE)
    line_set_1 = create_visualization_lines(image_1, camera_1)

    camera_2 = Camera([2000, 1, 0], [-25, -90, 0], 60)
    image_2 = cv2.imread("images_2/image_450.jpg", cv2.IMREAD_GRAYSCALE)
    line_set_2 = create_visualization_lines(image_2, camera_2)

    min_corner = np.array([-100, -100, -100])
    max_corner = np.array([100, 100, 100])

    # Create the axis-aligned bounding box
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
    aabb.color = (1, 0, 0)  # red

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5,  # length of each axis arrow
        origin=[0, 0, 0]  # position of the origin
    )


    # Visualize
    o3d.visualization.draw_geometries([line_set_1, line_set_2, aabb, axis])
