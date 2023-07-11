import numpy as np
import utility as util


class Point(object):
    """A point in 3D space"""
    def __init__(self, x, y, z, id, normal=None, is_used=False):
        self.x = np.float32(x)
        self.y = np.float32(y)
        self.z = np.float32(z)
        self.id = id
        self.normal = normal
        self.is_used = is_used

    @property
    def coords(self):
        return np.array([self.x, self.y, self.z])
