import numpy as np
import open3d as o3d


class Sphere(object):

    def __init__(self, center: np.ndarray, radius: float, res: int = 40, color: np.ndarray = np.array([0, 0, 0])):
        self.center = center
        self.radius = radius
        self.res = res
        self.color = color

    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, res: int = 40,
                 color: np.ndarray = np.array([0, 0, 0])):
        A = np.array([[v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]],
                      [v3[0] - v2[0], v3[1] - v2[1], v3[2] - v2[2]],
                      [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]]])

        b = 0.5 * np.array([np.dot(v2, v2) - np.dot(v1, v1),
                            np.dot(v3, v3) - np.dot(v2, v2),
                            np.dot(v3, v3) - np.dot(v1, v1)])

        # Check if the matrix is singular
        if np.linalg.det(A) == 0:
            # If the matrix is singular, use the least squares to solve the system
            self.center = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            # If the matrix is not singular, use the normal method
            self.center = np.linalg.solve(A, b)

        self.radius = np.sqrt(np.dot(v1 - self.center, v1 - self.center))
        self.res = res
        self.color = color

    def contains(self, v: np.ndarray):
        d = v - self.center
        return np.dot(d, d) < self.radius * self.radius

    @property
    def center3d(self):
        return np.array([self.center[0], self.center[1], self.center[2]])

    @property
    def as_o3d_line_set(self):
        theta = np.linspace(0, 2 * np.pi, self.res)
        phi = np.linspace(0, np.pi, self.res)

        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)

        points = np.array([x, y, z]).T

        lines = np.stack((np.arange(self.res - 1), np.arange(self.res - 1) + 1)).T

        lines = np.concatenate((lines, np.flip(lines, -1)), 0)

        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector2iVector(lines)
        ).translate(self.center3d).paint_uniform_color(self.color)

    @property
    def as_o3d_mesh(self):
        samples = np.linspace(0, 2 * np.pi, self.res)
        points = np.array([
            self.radius * np.cos(samples),
            self.radius * np.sin(samples),
            np.zeros(self.res),
        ]).T

        c = np.array([[0, 0, 0.5]])
        points = np.concatenate((c, points))
        triangles = np.stack((np.arange(self.res - 1) + 1, np.zeros(self.res - 1), np.arange(self.res - 1) + 2)).T
        triangles = np.concatenate((triangles, np.flip(triangles, -1)), 0)

        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector3iVector(triangles)
        ).translate(self.center3d).paint_uniform_color(self.color)
