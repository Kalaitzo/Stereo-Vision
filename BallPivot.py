import numpy as np
import utility as util
from point import Point
from edge import Edge
from tqdm import tqdm
from triangle3D import Triangle3D
from kd_tree import kd_tree

INF = np.inf


class BallPivotAlgorithm:
    def __init__(self, points, radius):
        self.first_free_point_ind = 0
        self.num_points_tried_to_seed_from = 0
        self.points = points
        self.radius = radius
        self.num_free_points = len(points)
        self.kdtree = kd_tree(points)
        self.triangles = []
        self.times_tried_to_create_mesh = 0
        self.used_points = []

    def create_mesh(self, first_point_ind: int) -> list:
        with tqdm(total=250) as pbar:
            while True:
                # Find a seed triangle
                seed_triangle = self.find_seed_triangle(first_point_ind)

                # If there is no seed triangle, return
                if seed_triangle is None:
                    print("No seed triangle found")
                    break

                next_seed_triangle = []
                # Pivot the ball around the seed triangle
                while seed_triangle is not None and next_seed_triangle not in self.triangles:
                    next_seed_triangle = self.pivot_ball(seed_triangle)
                    seed_triangle = next_seed_triangle

                self.times_tried_to_create_mesh += 1
                pbar.update(1)

                # If there are no more free getPoints, return
                if self.times_tried_to_create_mesh > 250:
                    print("Tried to create mesh 250 times")
                    break

                # Get a random free point that is not in the triangles list
                for tri in self.triangles:
                    for ind in tri:
                        self.used_points.append(ind)

                self.used_points = list(set(self.used_points))

                # Check if all the points are used
                if len(self.used_points) == len(self.points):
                    print("All points are used")
                    break

                first_point_ind = np.random.choice([i for i in range(len(self.points))
                                                    if i not in self.used_points])

        return self.triangles

    def find_seed_triangle(self, first_point_ind: int) -> Triangle3D:
        # Get the first point
        p1 = Point(self.points[first_point_ind][0],
                   self.points[first_point_ind][1],
                   self.points[first_point_ind][2],
                   first_point_ind)

        # Find the getPoints in radius 2r from the first point
        p1_neighbors_ind = self.kdtree.find_points_in_radius(self.points,
                                                             first_point_ind,
                                                             self.radius)

        # If there are no getPoints in radius r from the first point
        if len(p1_neighbors_ind) == 0:
            return None

        # Get the getPoints in radius 2r to calculate the distances
        p1_neighbors = [Point(self.points[p][0], self.points[p][1], self.points[p][2], p) for p in p1_neighbors_ind]

        # Calculate the distances from the first point to the neighbors
        p1_neighbors_dist = [util.points_distance(p1, p2) for p2 in p1_neighbors]

        # Sort the indices of the neighbors by distance
        p1_neighbor_points_ind_sorted = [p for _, p in sorted(zip(p1_neighbors_dist, p1_neighbors_ind))]

        # Limit the number of getPoints to try to seed from to 5
        LIMIT = 5
        p1_neighbor_points_ind_sorted = p1_neighbor_points_ind_sorted[:LIMIT]

        # The second point of the triangle will be one of the neighbors of the first point
        for second_point_ind in p1_neighbor_points_ind_sorted:
            p2 = Point(self.points[second_point_ind][0],
                       self.points[second_point_ind][1],
                       self.points[second_point_ind][2],
                       second_point_ind)

            # Check if the second point isn't the same as the first point
            if p2.id == p1.id:
                continue

            # Check if the second point is used
            if p2.is_used:
                continue

            # Find the getPoints that are in radius r from the second point
            p2_neighbors_ind = self.kdtree.find_points_in_radius(self.points,
                                                                 second_point_ind,
                                                                 self.radius)

            # If there are no getPoints in radius r from the second point, continue
            if len(p2_neighbors_ind) == 0:
                continue

            # Get the getPoints in radius 2r from the second point
            p2_neighbors = [Point(self.points[p][0], self.points[p][1], self.points[p][2], p) for p in p2_neighbors_ind]

            # Calculate the distances from the second point to the neighbors
            p2_neighbors_dist = [util.points_distance(p2, p3) for p3 in p2_neighbors]

            # Sort the indices of the neighbors by distance
            p2_neighbor_points_ind_sorted = [p for _, p in sorted(zip(p2_neighbors_dist, p2_neighbors_ind))]

            # Limit the number of getPoints to try to seed from to 5
            p2_neighbor_points_ind_sorted = p2_neighbor_points_ind_sorted[:LIMIT]

            # The third point should be both in the neighbors of the first point and the second point
            third_point_candidates_ind = np.intersect1d(p1_neighbor_points_ind_sorted,
                                                        p2_neighbor_points_ind_sorted)

            # If there are no candidates, continue
            if len(third_point_candidates_ind) == 0:
                continue

            # Get the candidates
            third_point_candidates = [Point(self.points[p][0],
                                            self.points[p][1],
                                            self.points[p][2],
                                            p) for p in third_point_candidates_ind]

            # Calculate the distances from the third point candidates to the first point
            third_point_candidates_dist = [util.points_distance(p1, p3) for p3 in third_point_candidates]

            # Sort the indices of the candidates by distance
            third_point_candidates_ind_sorted = [p for _, p in sorted(zip(third_point_candidates_dist,
                                                                          third_point_candidates_ind))]

            # Limit the number of getPoints to try to seed from to 5
            third_point_candidates_ind_sorted = third_point_candidates_ind_sorted[:LIMIT]

            # The third point should be both in the neighbors of the first point and the second point
            for third_point_ind in third_point_candidates_ind_sorted:
                p3 = Point(self.points[third_point_ind][0],
                           self.points[third_point_ind][1],
                           self.points[third_point_ind][2],
                           third_point_ind)

                # Check if the third point isn't the same as the first point or the second point
                if p3.id == p1.id or p3.id == p2.id:
                    continue

                # Check if the third point is used
                if p3.is_used:
                    continue

                points = [p1, p2, p3]

                # Create the triangle
                triangle = Triangle3D(points)

                # Check if the triangle is valid
                triangle_is_valid = self.check_triangle(triangle)

                if triangle_is_valid:
                    # Return the triangle
                    # Set the p1, p2, p3 as used
                    p1.is_used = True
                    p2.is_used = True
                    p3.is_used = True

                    # Add the triangle to the list of triangles
                    self.triangles.append([first_point_ind, second_point_ind, third_point_ind])

                    return triangle

        # If no triangle was found, call the function again with the next point
        # Go a random index of the list of points to try to seed from tha is not used
        first_point_ind = np.random.choice([i for i in range(len(self.points))
                                            if i not in self.used_points])

        # If there is no point to try to seed from, return None
        if first_point_ind is None:
            return None

        return self.find_seed_triangle(first_point_ind)

    def pivot_ball(self, seed_triangle: Triangle3D):
        # Try to pivot the ball around each edge of the triangle
        for edge in seed_triangle.edges:
            p1 = edge.p1
            p2 = edge.p2

            # Find the getPoints in radius r from each point of the edge
            p1_neighbors_ind = self.kdtree.find_points_in_radius(self.points,
                                                                 p1.id,
                                                                 self.radius)

            p2_neighbors_ind = self.kdtree.find_points_in_radius(self.points,
                                                                 p2.id,
                                                                 self.radius)

            # If there are no getPoints in radius r from each point of the edge, continue
            if len(p1_neighbors_ind) == 0 or len(p2_neighbors_ind) == 0:
                continue

            # Find the neighbors that are in radius r from both getPoints of the edge
            neighbors_ind = np.intersect1d(p1_neighbors_ind, p2_neighbors_ind)

            # If there are no neighbors, continue
            if len(neighbors_ind) == 0:
                continue

            # Get the neighbors
            neighbors = [Point(self.points[p][0], self.points[p][1], self.points[p][2], p) for p in neighbors_ind]

            # Calculate the distances from the neighbors to the edge
            neighbors_dist = [util.point_to_edge_distance(p3, edge) for p3 in neighbors]

            # Sort the indices of the neighbors by distance
            neighbors_ind_sorted = [p for _, p in sorted(zip(neighbors_dist, neighbors_ind))]

            LIMIT = 5
            # Iterate through the neighbors
            for neighbor_ind in neighbors_ind_sorted[:LIMIT]:
                # Get the neighbor
                p3 = Point(self.points[neighbor_ind][0],
                           self.points[neighbor_ind][1],
                           self.points[neighbor_ind][2],
                           neighbor_ind,
                           is_used=True)

                # Check if the neighbor is the same as the first point or the second point
                if p3.id == p1.id or p3.id == p2.id:
                    continue

                # Check if the triangle is already inserted in the list of triangles
                if [p1.id, p2.id, p3.id] in self.triangles or [p2.id, p1.id, p3.id] in self.triangles or \
                   [p1.id, p3.id, p2.id] in self.triangles or [p2.id, p3.id, p1.id] in self.triangles or \
                   [p3.id, p1.id, p2.id] in self.triangles or [p3.id, p2.id, p1.id] in self.triangles:
                    continue

                # Crete the new triangle
                new_triangle = Triangle3D([p1, p2, p3])

                # Check if the triangle is valid
                triangle_is_valid = self.check_triangle(new_triangle)

                if triangle_is_valid:
                    # Return the triangle
                    # Set the p1, p2, p3 as used
                    p1.is_used = True
                    p2.is_used = True
                    p3.is_used = True

                    # Add the triangle to the list of triangles
                    self.triangles.append([p1.id, p2.id, p3.id])

                    return new_triangle

        # If no triangle was found, the ball can pivot around the triangle
        return None

    def check_triangle(self, triangle: Triangle3D):
        # Check if the triangle is valid by checking if the sphere touches all three getPoints of the triangle. For that
        # to happen the radius of the circle in the triangle should be less than the radius of the sphere
        if triangle.in_circle_radius > self.radius:
            return False

        return True

