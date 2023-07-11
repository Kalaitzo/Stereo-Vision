import random
import numpy as np


def ransac(points, threshold=0.05, iterations=1000):
    """
    RANSAC algorithm
    :param points: The coordinates of the points of the point cloud
    :param threshold: The threshold to consider a point as an inlier
    :param iterations: The number of iterations
    :return: Return the inliers and the coefficients of the best fitting plane
    """
    # Set an empty list to store the inliers
    inliers = []

    # Set an empty list to store the coefficients of the plane
    coefficients = []

    # Get the number of points
    n_points = len(points)

    # Initialize a variable to iterate over the number of iterations
    i = 1

    # Iterate over the number of iterations
    while i < iterations:
        # Step 1: Find a plane with 3 random points
        # Get 3 random indices
        idx_sample = random.sample(range(n_points), 3)

        # Get the points
        plane_points = points[idx_sample]

        # Define the two vectors of the plane
        v1 = plane_points[1] - plane_points[0]
        v2 = plane_points[2] - plane_points[0]

        # Define the normal vector of the plane
        normal = np.cross(v1, v2)

        # Define the plane following the equation ax + by + cz + d = 0
        # normal = [a, b, c]
        a, b, c = normal / np.linalg.norm(normal)

        # d = -(ax0+by0+cz0)
        d = -np.dot(normal, plane_points[0])

        # Step 2: Get the distances of all the points to the plane
        # The distance of a point from a plane ax+by+cz+d=0 is given by the formula |ax0+by0+cz0+d|/sqrt(a^2+b^2+c^2)
        distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

        # Step 3: Get the inliers candidates
        inliers_candidates = np.where(np.abs(distances) < threshold)[0]

        # Step 4: Check if the number of inliers candidates is greater than the number of inliers
        # If the number of inliers candidates is greater than the number of inliers, update the inliers and store the
        # coefficients of the plane
        if len(inliers_candidates) > len(inliers):
            # Update the inliers
            inliers = inliers_candidates

            # Store the coefficient
            coefficients = [a, b, c, d]

        # Update the number of iterations
        i += 1

    # Return the inliers and the coefficients of the plane
    return inliers, coefficients
