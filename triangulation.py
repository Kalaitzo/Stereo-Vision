import numpy as np
import open3d as o3d
import utility as util
from kd_tree import kd_tree
from BallPivot import BallPivotAlgorithm

# Load the file with the data
data = np.loadtxt('./data/pointclouds/sample_w_normals.xyz', skiprows=1)

# Create a point cloud object and assign the data to it
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6]/255)
pcd.normals = o3d.utility.Vector3dVector(data[:, 6:9])

# Take 10000 random points from the point cloud
points = np.asarray(pcd.points)[np.random.choice(np.asarray(pcd.points).shape[0], 10000, replace=False)]

# Change the points of the point cloud to the random points
pcd.points = o3d.utility.Vector3dVector(points)

# Get teh average distance between the getPoints
# Create a kdtree from the getPoints
kdtree = kd_tree(points=points)

indices, distances = util.find_k_nearest_neighbors_with_kd_tree(points, kdtree, 7)

average_distances = np.mean(distances, axis=0)

mean_distance = np.mean(average_distances)

# Create a BallPivot object with the getPoints and the mean distance multiplied by 2
ball_pivot = BallPivotAlgorithm(points, mean_distance * 2)
# Begin the algorithm at a random point
triangles = ball_pivot.create_mesh(np.random.randint(0, len(points)))

# Find the amount of triangles
num_triangles = len(ball_pivot.triangles)

print('Done creating mesh with {} triangles'.format(num_triangles))

# Create a mesh object
mesh = o3d.geometry.TriangleMesh()

# Assign the vertices and triangles to the mesh
mesh.vertices = o3d.utility.Vector3dVector(points)
mesh.triangles = o3d.utility.Vector3iVector(triangles)

# Recenter the mesh
mesh.translate(-mesh.get_center())

# Recenter the point cloud
pcd.translate(-pcd.get_center())

# Pain the mesh red
mesh.paint_uniform_color([1, 0, 0])

# Draw the mesh
o3d.visualization.draw_geometries([mesh, pcd])
