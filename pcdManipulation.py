import numpy as np
import open3d as o3d


# Function that creates a point cloud from a file (supports .ply, .xyz and .txt)
def createPointCloudFromFile(filename: str, folder: str = "data/pointclouds/"):
    """
    Create a point cloud from a file
    :param filename: The name of the file
    :param folder: The folder of the file
    :return: A point cloud
    """
    # Check the format of the file
    if filename.endswith(".ply"):
        # Import the point cloud
        pcd = o3d.io.read_point_cloud(folder + filename)
        # Return the point cloud
        return pcd
    elif filename.endswith(".xyz"):
        # Import the point cloud
        pcd = o3d.io.read_point_cloud(folder + filename, format='xyz')
        # Return the point cloud
        return pcd
    elif filename.endswith(".txt"):
        # Get the data from the txt file
        data = np.loadtxt(folder + filename, delimiter=",", skiprows=1)
        # Get the x, y and z coordinates
        xyz = data[:, :3]
        # Create a point cloud
        pcd = o3d.geometry.PointCloud()
        # Add the coordinates to the point cloud
        pcd.points = o3d.utility.Vector3dVector(xyz)
        # Re-center the point cloud
        pcd = pcd.translate(-pcd.get_center())
        # Return the point cloud
        return pcd
    else:
        # Print an error message
        print("File format not supported")
        # Return None
        return None


# Function that down-samples a point cloud
def downSamplePointcloud(pcd: o3d.geometry.PointCloud):
    """
    Down-sample a point cloud
    :param pcd: The point cloud
    :return: The down-sampled point cloud
    """
    # Down sample the point cloud
    down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # Return the down-sampled point cloud
    return down_pcd


# Function that re-centers a point cloud
def recenterPointCloud(pcd: o3d.geometry.PointCloud):
    """
    Re-center a point cloud
    :param pcd: The point cloud
    :return: The re-centered point cloud
    """
    # Get the center of the point cloud
    center = pcd.get_center()

    # Get the points
    points = np.asarray(pcd.points)

    # Re-center the point cloud
    points = points - center

    # Create a new point cloud
    new_pcd = o3d.geometry.PointCloud()

    # Set the points
    new_pcd.points = o3d.utility.Vector3dVector(points)

    # Return the re-centered point cloud
    return new_pcd


# Function that finds the AABB of a point cloud
def getAABB(obj: o3d.geometry.PointCloud, color: np.ndarray = None):
    """
    Get the axis-aligned bounding box of a point cloud
    :param color: The color of the bounding box
    :param obj: The object we want to get the bounding box of
    :return: The axis-aligned bounding box
    """
    # Get the points of the point cloud
    points = np.asarray(obj.points)

    # Get the minimum and maximum points
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)

    # Create the bounding box
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_point, max_bound=max_point)

    # Paint the bounding box
    bounding_box.color = obj.colors[0] if color is None else color

    # Return the bounding box
    return bounding_box


# Function that creates a pointcloud from the outliers
def createOutlierPointCloud(points: np.ndarray, inlier_index: np.ndarray):
    """
    Create a point cloud from the outliers
    :param points: The points of the point cloud
    :param inlier_index: The inlier index
    :return: The outlier point cloud
    """
    # Create a mask for the outliers
    outliers = np.ones(points.shape[0], dtype=bool)

    # Get the invert of inlier_index in order to get the outliers
    outliers[inlier_index] = False

    # Get the outliers (points)
    outliers = points[outliers]

    # Create a point cloud
    outlier_pcd = o3d.geometry.PointCloud()

    # Set the points
    outlier_pcd.points = o3d.utility.Vector3dVector(outliers)

    # Return the outlier point cloud
    return outlier_pcd


# Function that creates a pointcloud from the inliers
def createInlierPointCloud(points: np.ndarray, inlier_index: np.ndarray):
    """
    Create a point cloud from the inliers
    :param points: The points of the point cloud
    :param inlier_index: The inlier index
    :return: The inlier point cloud
    """
    # Get the inliers (points)
    inliers = points[inlier_index]

    # Create a point cloud
    inlier_pcd = o3d.geometry.PointCloud()

    # Set the points
    inlier_pcd.points = o3d.utility.Vector3dVector(inliers)

    # Return the inlier point cloud
    return inlier_pcd


# Function that creates a point cloud object from vertices and colors
def createPointCloudFromVertices(vertices: np.ndarray, colors: np.ndarray):
    """
    Create a point cloud object from vertices and colors
    :param vertices: The vertices of the point cloud
    :param colors: The colors of the point cloud
    :return: The point cloud object
    """
    # Create a point cloud
    pcd = o3d.geometry.PointCloud()

    # Set the vertices
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # Set the colors
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Return the point cloud
    return pcd
