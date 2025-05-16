import numpy as np
import open3d as o3d


class Camera:
    def __init__(self, position: [int, int, int], rotation: [int, int, int], fov: int):
        # coordinates [x, y, z]
        self.position = position
        # camera fov
        self.fov = fov
        # rotation in degrees [x, y, z]
        self.rotation = rotation


def get_rotation_matrix(x_deg, y_deg, z_deg):
    # Convert degrees to radians
    x = np.radians(x_deg)
    y = np.radians(y_deg)
    z = np.radians(z_deg)

    # Rotation matrix around the x-axis
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])

    # Rotation matrix around the y-axis
    ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])

    # Rotation matrix around the z-axis
    rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])

    # Unity uses ZXY order: R = Ry * Rx * Rz (applied in that order)
    return ry @ rx @ rz


def create_camera_frustum(pos, rot=np.eye(3), img_w=640, img_h=480, fov_deg=60, scale=0.2, color=[0, 0, 0]):
    """
    Create a LineSet representing a camera frustum.
    Parameters:
        pos: (list of 3) Camera position [x, y, z]
        rot: (3x3 np.array) Camera rotation matrix (world_from_camera)
    """
    # Convert FOV to focal length
    fov_rad = np.deg2rad(fov_deg)
    focal = (img_w / 2) / np.tan(fov_rad / 2)

    # Image plane corners in camera space
    cx, cy = img_w / 2, img_h / 2
    z = 1.0
    x0 = (-cx) / focal
    x1 = (img_w - cx) / focal
    y0 = -(cy) / focal
    y1 = (img_h - cy) / focal

    # Define frustum points in camera space
    corners = np.array([
        [0, 0, 0],  # camera center
        [x0, y0, z],
        [x1, y0, z],
        [x1, y1, z],
        [x0, y1, z],
    ]) * scale

    # Transform to world coordinates
    corners = (rot @ corners.T).T + np.array(pos)

    # Define lines between points to form the frustum
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    colors = [color for _ in lines]

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(corners)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)

    return frustum


def create_visualization_lines(image, camera: Camera, threshold=10, ray_length=10000):
    image_height, image_width = image.shape
    focal_length = (image_width / 2) / np.tan(np.deg2rad(camera.fov) / 2)

    points = []
    colors = []

    for v in range(image_height):
        for u in range(image_width):
            if image[v, u] > threshold:
                # Camera space
                x = (u - image_width / 2)
                y = -(v - image_height / 2)
                z = focal_length
                dir_cam = np.array([x, y, z])
                dir_cam = dir_cam / np.linalg.norm(dir_cam)

                # World space
                dir_world = get_rotation_matrix(*camera.rotation) @ dir_cam
                end_point = camera.position + dir_world * ray_length

                points.append(camera.position)
                points.append(end_point)

    lines = [[i, i + 1] for i in range(0, len(points), 2)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set
