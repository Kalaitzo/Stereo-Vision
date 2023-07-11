import math
import numpy as np
import open3d as o3d
import utility as util
from edge import Edge
from sphere import Sphere
from point import Point

dims = lambda x: len(x.shape)
di = lambda x, i: x.shape[i]


class Triangle3D(object):

    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, color: np.ndarray = np.array([0, 0, 0])):
        assert dims(v1) == dims(v2) == dims(v3) == 1, "Vertices must be one-dimensional arrays"
        assert di(v1, 0) == di(v2, 0) == di(v3, 0) == 3, "Vertices must have a shape of (3,)"

        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

        self.e1 = Edge(v1, v3)
        self.e2 = Edge(v1, v2)
        self.e3 = Edge(v2, v3)

        self.color = color

    def __init__(self, vertices: list[Point], color: np.ndarray = np.array([0, 0, 0])):
        # assert dims(vertices) == 2, "Vertex array must be two-dimensional"
        # assert di(vertices, 0) == di(vertices, 1) == 3, "Vertex array must have a shape of (3, 3)"

        self.v1 = vertices[0].coords
        self.v2 = vertices[1].coords
        self.v3 = vertices[2].coords

        self.e1 = Edge(vertices[0], vertices[2])
        self.e2 = Edge(vertices[0], vertices[1])
        self.e3 = Edge(vertices[1], vertices[2])

        self.p1 = vertices[0]
        self.p2 = vertices[1]
        self.p3 = vertices[2]

        self.color = color

    def contains(self, v):
        # Computer the 3 areas that are created by the point and the vertices
        a1 = Triangle3D(self.v1, self.v2, v).area
        a2 = Triangle3D(self.v2, self.v3, v).area
        a3 = Triangle3D(self.v3, self.v1, v).area

        # If the sum of the areas is equal to the area of the triangle, the point is inside
        return np.isclose(a1 + a2 + a3, self.area)

    def set_color(self, color: np.ndarray):
        self.color = color

    @property
    def getPoints(self):
        return [self.v1, self.v2, self.v3]

    @property
    def min_max_angle(self):
        edges = self.edges

        angle1 = np.arccos(
            np.dot(edges[0], edges[2]) / (np.linalg.norm(edges[0]) * np.linalg.norm(edges[2]))
        ) * 180 / np.pi
        angle2 = np.arccos(
            np.dot(edges[0], edges[1]) / (np.linalg.norm(edges[0]) * np.linalg.norm(edges[1]))
        ) * 180 / np.pi
        angle3 = np.arccos(
            np.dot(edges[1], edges[2]) / (np.linalg.norm(edges[1]) * np.linalg.norm(edges[2]))
        ) * 180 / np.pi

        return min(angle1, angle2, angle3), max(angle1, angle2, angle3)

    @property
    def in_circle_radius(self):
        # Get the length of the edges of the triangle
        p1 = Point(self.v1[0], self.v1[1], self.v1[2], 0)
        p2 = Point(self.v2[0], self.v2[1], self.v2[2], 1)
        p3 = Point(self.v3[0], self.v3[1], self.v3[2], 2)

        e1_length = util.points_distance(p1, p3)
        e2_length = util.points_distance(p1, p2)
        e3_length = util.points_distance(p2, p3)

        # Calculate the semi-perimeter
        s = (e1_length + e2_length + e3_length) / 2

        # Calculate the radius of the in_circle
        radius = math.sqrt(((s - e1_length) * (s - e2_length) * (s - e3_length)) / s)
        return radius

    @property
    def area(self):
        return 0.5 * np.linalg.norm(np.cross(self.v2 - self.v1, self.v3 - self.v1))

    @property
    def edges(self):
        return np.array([self.e1, self.e2, self.e3])

    @property
    def centroid(self):
        return (self.v1 + self.v2 + self.v3) / 3

    @property
    def circumSphere(self):
        return Sphere(self.v1, self.v2, self.v3)

    @property
    def as_o3d_line_set(self):
        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(np.array([self.v1, self.v2, self.v3])),
            o3d.utility.Vector2iVector(np.array([[0, 1], [1, 2], [2, 0]]))
        ).paint_uniform_color(self.color)

    @property
    def as_o3d_mesh(self):
        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.array([self.v1, self.v2, self.v3])),
            o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
        ).paint_uniform_color(self.color)

    @property
    def as_o3d_mesh_fb(self):
        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.array([self.v1, self.v2, self.v3])),
            o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 1]]))
        ).paint_uniform_color(self.color)
