import numpy as np
import open3d

"""
This script is for visualizing camera poses and bounding boxes.  
"""

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_K = np.array([[969.7, 0, 960],
                      [0, 969.7, 540],
                      [0, 0, 1]])

def draw_camera(K, R, t, w, h,
                scale=1, color=[0.8, 0.2, 0.8]):
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5 * scale)
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines))
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]

def draw_box(min_bound, max_bound, color):
    x1, y1, z1 = min_bound
    x2, y2, z2 = max_bound
    points = [
        [x1, y1, z1],
        [x2, y1, z1],
        [x1, y2, z1],
        [x2, y2, z1],
        [x1, y1, z2],
        [x2, y1, z2],
        [x1, y2, z2],
        [x2, y2, z2]
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set

class Visualizer:
    def __init__(self):
        self.cameras = []
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()
    
    def add_cameras(self, c2ws, K=DEFAULT_K, w=DEFAULT_WIDTH, h=DEFAULT_HEIGHT, 
                    scale=1, color=[0.8, 0.2, 0.8]):
        for c2w in c2ws:
            R = c2w[:3, :3]
            t = c2w[:3, 3].flatten()
            cam_model = draw_camera(K, R, t, w, h, scale, color)
            for i in cam_model:
                self.__vis.add_geometry(i)
    
    def add_boxes(self, boxes, color=(1,0,0)):
        N = len(boxes)
        for i in range(N):
            line_set = draw_box(boxes[i][0], boxes[i][1], color)
            self.__vis.add_geometry(line_set)

    def add_axis(self, scale=1):
        axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size = scale)
        self.__vis.add_geometry(axis)
    
    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()

