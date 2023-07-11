from cmath import inf
import numpy as np
import heapq


class kd_node:

    def __init__(self, index, left_child, right_child):
        self.index = index
        self.left_child = left_child
        self.right_child = right_child

    def depth_first_search(self):
        indices = depth_first_search(self, indices=[])
        return indices


class kd_tree:

    def __init__(self, points: np.array):
        self.root = self.build_kd_tree(points)

    def build_kd_tree(self, points: np.array):
        root = build_kd_tree(points, dim=points.shape[1], indices=np.arange(len(points)), level=0)
        return root

    def get_nodes_of_level(self, level):
        nodes = get_nodes_of_level(self.root, level, nodes=[])
        return nodes

    def find_nearest_neighbor(self, points, id):
        dstar, istar = find_nearest_neighbor(points, id, dim=points.shape[1], root=self.root, level=0, istar=-1, dstar=inf)
        return dstar, istar

    def find_points_in_radius(self, points, id, radius):
        indices = find_points_in_radius(points, id, radius, dim=points.shape[1], root=self.root, level=0, indices=[])
        return indices

    def find_k_nearest_neighbors(self, points, id, K):
        heap, _ = find_k_nearest_neighbors(points, id, K + 1, dim=points.shape[1], root=self.root, level=0, heap=[],
                                           dstar=inf)
        indices = []
        distances = []
        while len(heap) > 0:
            distance, node = heapq.heappop(heap)
            indices.append(node)
            distances.append(abs(distance))

        return distances[::-1], indices[::-1]


def build_kd_tree(points: np.array, dim, indices, level):
    # If there are no points, return
    if len(indices) == 0:
        return

    # Get the axis to split on
    axis = level % dim

    # The order of the points along the axis
    order = np.argsort(points[indices, axis])

    # The indices of the points sorted along the axis
    sorted_indices = indices[order]

    # The median index
    median_index = (len(indices) - 1) // 2

    # The index of the new root
    root_index = sorted_indices[median_index]

    # The indices of the left child
    left_child_indices = sorted_indices[:median_index]

    # The indices of the right child
    right_child_indices = sorted_indices[median_index + 1:]

    # The left child
    left_child = build_kd_tree(points, dim, left_child_indices, level + 1)

    # The right child
    right_child = build_kd_tree(points, dim, right_child_indices, level + 1)

    # Return the root
    return kd_node(index=root_index, left_child=left_child, right_child=right_child)


def get_nodes_of_level(root: kd_node, level, nodes):
    # If the level is 0, add the root to the list of nodes
    if level == 0:
        nodes.append(root)
    else:
        if root.left_child:
            # Get the nodes of the left child
            nodes = get_nodes_of_level(root.left_child, level - 1, nodes)
        if root.right_child:
            # Get the nodes of the right child
            nodes = get_nodes_of_level(root.right_child, level - 1, nodes)

    # Return the nodes
    return nodes


def depth_first_search(root: kd_node, indices):
    if root.left_child:
        indices = depth_first_search(root.left_child, indices)
    if root.right_child:
        indices = depth_first_search(root.right_child, indices)

    indices.append(root.index)

    return indices


def find_nearest_neighbor(points: np.array, id, dim, root: kd_node, level, dstar, istar):
    # Find the axis
    axis = level % dim

    # Get the distance between the point and the root along the axis
    d_ = points[id, axis] - points[root.index, axis]

    # Check if the point is on the left or right of the root
    is_on_left = d_ < 0

    if is_on_left:
        # Check if the left child exists
        if root.left_child:
            # Find the nearest neighbor in the left child
            dstar, istar = find_nearest_neighbor(points, id, dim, root.left_child, level + 1, dstar, istar)

        # Check if the distance between the point and the root is less than the current minimum distance
        if d_ ** 2 < dstar ** 2:
            # Check if the right child exists
            if root.right_child:
                # Find the nearest neighbor in the right child
                dstar, istar = find_nearest_neighbor(points, id, dim, root.right_child, level + 1, dstar, istar)

    else:
        if root.right_child:
            # Find the nearest neighbor in the right child
            dstar, istar = find_nearest_neighbor(points, id, dim, root.right_child, level + 1, dstar, istar)

        # Check if the distance between the point and the root is less than the current minimum distance
        if d_ ** 2 < dstar ** 2:
            # Check if the left child exists
            if root.left_child:
                # Find the nearest neighbor in the left child
                dstar, istar = find_nearest_neighbor(points, id, dim, root.left_child, level + 1, dstar, istar)

    # Check if the point is the root
    if root.index == id:
        pass
    else:
        # Get the distance between the point and the root
        d = np.linalg.norm(points[id] - points[root.index])

        # If the distance is less than the current minimum distance, update the minimum distance
        if d < dstar:
            dstar = d
            istar = root.index

    return dstar, istar


def find_points_in_radius(points: np.array, id, radius, dim, root: kd_node, level, indices):
    # Find the axis
    axis = level % dim

    # Get the distance between the point and the root along the axis
    d_ = points[id, axis] - points[root.index, axis]

    # Check if the point is on the left or right of the root
    is_on_left = d_ < 0

    if is_on_left:
        # Check if the left child exists
        if root.left_child:
            # Find the nearest neighbor in the left child
            find_points_in_radius(points, id, radius, dim, root.left_child, level + 1, indices)

        # Check if the distance between the point and the root is less than the current minimum distance
        if d_ ** 2 < radius ** 2:
            # Check if the right child exists
            if root.right_child:
                # Find the nearest neighbor in the right child
                find_points_in_radius(points, id, radius, dim, root.right_child, level + 1, indices)
    else:
        if root.right_child:
            # Find the nearest neighbor in the right child
            find_points_in_radius(points, id, radius, dim, root.right_child, level + 1, indices)

        # Check if the distance between the point and the root is less than the current minimum distance
        if d_ ** 2 < radius ** 2:
            # Check if the left child exists
            if root.left_child:
                # Find the nearest neighbor in the left child
                find_points_in_radius(points, id, radius, dim, root.left_child, level + 1, indices)

    # Check if the point is not the root
    if root.index != id:
        # Get the distance between the point and the root
        d = np.linalg.norm(points[id] - points[root.index])

        # If the distance is less than the current minimum distance, update the minimum distance
        if d < radius:
            indices.append(root.index)

    return indices


# Homework: Find the k nearest neighbours using a max heap
def find_k_nearest_neighbors(points: np.array, id, K, dim, root: kd_node, level, heap, dstar):
    # Find the axis
    axis = level % dim

    # Get the distance between the point and the root along the axis
    d_ = points[id, axis] - points[root.index, axis]

    # Check if the point is on the left or right of the root
    is_on_left = d_ < 0

    if is_on_left:
        # Check if the left child exists
        if root.left_child:
            # Find the nearest neighbor in the left child
            heap, dstar = find_k_nearest_neighbors(points, id, K, dim, root.left_child, level + 1, heap, dstar)

        # Check if the distance between the point and the root is less than the current minimum distance
        if d_ ** 2 < dstar ** 2:
            # Check if the right child exists
            if root.right_child:
                # Find the nearest neighbor in the right child
                heap, dstar = find_k_nearest_neighbors(points, id, K, dim, root.right_child, level + 1, heap, dstar)

    else:
        if root.right_child:
            # Find the nearest neighbor in the right child
            heap, dstar = find_k_nearest_neighbors(points, id, K, dim, root.right_child, level + 1, heap, dstar)

        # Check if the distance between the point and the root is less than the current minimum distance
        if d_ ** 2 < dstar ** 2:
            # Check if the left child exists
            if root.left_child:
                # Find the nearest neighbor in the left child
                heap, dstar = find_k_nearest_neighbors(points, id, K, dim, root.left_child, level + 1, heap, dstar)

    # Check if the point is not the root
    if root.index != id:
        # Get the distance between the point and the root
        d = np.sqrt(np.sum(np.square(points[root.index, :] - points[id, :])))

        # If the distance is less than the current minimum distance, update the minimum distance
        if len(heap) < K-1:
            heapq.heappush(heap, (-d, root.index))
        else:
            # If the heap is full replace the maximum element if the distance is less than the maximum distance
            if d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, root.index))
                dstar = -heap[0][0]

    return heap, dstar
