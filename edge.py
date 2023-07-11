import utility as util


class Edge:
    """An edge in 3D space"""
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.edge_in_triangles = 0
        self.color = []

    @property
    def length(self):
        return util.points_distance(self.p1, self.p2)
