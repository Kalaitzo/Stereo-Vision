import cv2
import pcds
import time
import math
import stereoPairs
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from kd_tree import kd_tree
from KittiReconstruction import loadCalib
from scipy.ndimage import convolve
from sklearn.cluster import DBSCAN
from matplotlib.patches import Rectangle


# Function that displays an image with a rectangle on top of it
def disp_image_and_rectangle(img, rect_start, template_rows, template_cols):
    """
    Display the image and the rectangle on top of it
    :param img: The image to display
    :param rect_start: The top left corner of the rectangle
    :param template_rows: The number of rows of the rectangle
    :param template_cols: The number of columns of the rectangle
    :return: Display the image and the rectangle on top of it
    """
    # Display the original image
    plt.imshow(img, cmap='gray')

    # Get the current reference
    ax = plt.gca()

    # Create a Rectangle patch
    rect = Rectangle(rect_start, template_cols, template_rows, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


# Function that visualizes the stereo frames
def visualize_stereo_frames(left_frame: np.ndarray, right_frame: np.ndarray):
    """
    Visualize the stereo frames
    :param left_frame: The left frame
    :param right_frame: The right frame
    :return: Visualize the stereo frames
    """

    # Concatenate the left and right frames horizontally
    concatenated_frames = np.hstack((left_frame, right_frame))

    # Convert to RGB if the input is in BGR format
    if left_frame.ndim == 3 and left_frame.shape[2] == 3:
        concatenated_frames = cv2.cvtColor(concatenated_frames, cv2.COLOR_BGR2RGB)

    # Display the concatenated frames using Matplotlib
    plt.figure(figsize=(12, 5))
    plt.imshow(concatenated_frames, cmap='gray' if left_frame.ndim == 2 else None)
    plt.title('Stereo Frames')
    plt.axis('off')
    plt.show()


# Function that visualizes a disparity map
def visualize_disparity_map(disparity_map: np.ndarray, cmap: str = 'jet', title: str = 'Disparity Map'):
    """
    Visualize the disparity map
    :param title:
    :param disparity_map: The disparity map
    :param cmap: The color map to use
    :return: Visualize the disparity map
    """
    # Normalize the disparity map for visualization
    # normalized_disparity = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min())
    normalized_disparity = disparity_map

    # Display the disparity map using Matplotlib
    plt.figure(figsize=(12, 5))
    plt.imshow(normalized_disparity, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    plt.show()


# Function that visualizes the point cloud
def visualize_point_cloud(vertices: np.ndarray, colors: np.ndarray):
    """
    Visualize the point cloud
    :param vertices: The vertices of the point cloud
    :param colors: The colors of the point cloud
    :return: Visualize the point cloud
    """
    # Create a point cloud object
    point_cloud = o3d.geometry.PointCloud()

    # Set the vertices and the colors
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])


# Function that calculates a disparity map from a stereo pair using block matching algorithm
def stereo2disparity(left_frame: np.ndarray, right_frame: np.ndarray, block_size: int, previous_blocks_num: int):
    """
    Stereo to disparity
    :param left_frame: The left frame
    :param right_frame: The right frame
    :param block_size: The block size
    :param previous_blocks_num: The number of previous blocks
    :return: The disparity map
    """
    # Get the shape of the left frame
    rows, cols = left_frame.shape

    # Initialize the disparity map
    disparity_map = np.zeros_like(left_frame, dtype=np.float32)

    # Iterate over the left frame
    for row in tqdm(range(rows)):
        for col in range(cols):
            # Get the template
            template = left_frame[row: row + block_size, col: col + block_size]

            # Set the start of the columns of the search space as a parameter
            column_start = max(0, col - previous_blocks_num * block_size)
            # Set the end of the columns of the search space as a parameter
            column_end = col + block_size

            # Get the search space
            search_space = right_frame[row: row + block_size, column_start: column_end]

            # Do the template matching
            res = cv2.matchTemplate(search_space, template, cv2.TM_SQDIFF)

            # find the location of the min value
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Get the disparity
            disparity = col - (column_start + min_loc[0])

            # Update the disparity map
            disparity_map[row, col] = disparity

    return disparity_map


# Function that calculate a point cloud from a disparity map using the pinhole camera model
def disparity2pointcloud(rows: int,
                         columns: int,
                         disparity: np.ndarray,
                         baseline: float,
                         left_frame_color: np.ndarray,
                         field_of_view: float = 1.2,
                         scale: int = 40):
    """
    Disparity to point cloud
    :return: The vertices and the colors of the point cloud
    """
    # Get the center of the image
    u0 = columns / 2
    v0 = rows / 2

    # Calculate the focal length
    focal_length = 1.0 / (2.0 * np.tan(field_of_view / 2.0))

    # Initialize the vertices and the color lists for the point cloud
    vertices = []
    colors = []

    # Iterate over the disparity map and calculate the point cloud
    for i in range(columns):
        for j in range(rows):
            # Normalize the disparity in order to be in the range [0, 1]
            normalized_disparity = disparity[j, i] / columns

            # If the normalized disparity is 0, then skip this point
            if normalized_disparity == 0:
                continue

            # Get the color from the left frame to color the point cloud
            color = left_frame_color[j, i]

            # Calculate the depth
            depth = focal_length * baseline / normalized_disparity

            # Calculate the x, y, z coordinates
            # Calculate the x coordinate
            x = ((i - u0) / columns) * (depth / focal_length)

            # Calculate the y coordinate
            y = -((j - v0) / rows) * (depth / focal_length)

            # The z coordinate is the depth
            z = depth

            if z > 0:
                # Append the vertices and the colors
                vertices.append([scale * x, scale * y, -scale * z])
                colors.append(np.array(color) / 255.0)

    vertices = np.stack(vertices)
    colors = np.stack(colors)
    return vertices, colors


# Function that de-noises the point cloud using DBSCAN
def denoiseWithClustering(vertices: np.ndarray, colors: np.ndarray, eps: float, samples: int = 10):
    """
    Denoise the point cloud using DBSCAN
    :param vertices: The vertices of the point cloud
    :param colors: The colors of the point cloud
    :param eps: The epsilon parameter of DBSCAN (maximum distance between two samples)
    :param samples: The minimum number of samples in a neighborhood for a point to be considered as a core point
    :return: Return the de-noised point cloud
    """
    # Create the DBSCAN object
    labels = DBSCAN(eps=eps, min_samples=samples).fit(vertices)

    # Get the labels
    labels = labels.labels_

    # Get the unique labels
    vertices = vertices[labels > -1]

    # Get the unique colors
    colors = colors[labels > -1]

    return vertices, colors


# Function that transforms a rgb image to grayscale
def rgb2gray(rgb: np.ndarray):
    """
    Convert an RGB image to grayscale.

    Args:
        rgb (np.ndarray): RGB image.

    Returns:
        np.ndarray: Grayscale image.
    """
    return rgb.mean(axis=2)


# Function that downscales an image
def downscale(image: np.ndarray):
    """
    Downscale an image

    Args:
        image (np.ndarray): Image to downscale.

    Returns:
        np.ndarray: Downscaled image.
    """

    # Check if the image is grayscale or RGB
    if len(image.shape) == 2:
        # Grayscale image
        # Create the kernel
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]]) / 16

        # Downscale the image
        image = convolve(image, kernel)
        image = image[::2, ::2]

    elif len(image.shape) == 3:
        # RGB image
        # Create the kernel
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]]) / 16

        # Downscale the image
        image = convolve(image, kernel[:, :, None])
        image = image[::2, ::2, :]

    else:
        raise ValueError("The image must be grayscale or RGB")

    return image


# Function that applies gaussian smoothing to an image
def gaussianSmoothing(image: np.ndarray):
    """
    Gaussian smoothing
    :param image: The disparity map
    :return: The smoothed disparity map
    """
    # Create the kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16

    # Convolve the disparity map with the kernel
    smoothed_disparity_map = convolve(image, kernel)

    return smoothed_disparity_map


# Function that finds the nearest neighbor of a point with a kd-tree
def find_nearest_neighbor_with_kd_tree(points: np.ndarray, kdtree: kd_tree):
    """
    Find the nearest neighbor of a point with a kd tree
    :param points: The coordinates of the points of the point cloud
    :param kdtree: The kd tree
    :return: The distance and the index of the nearest neighbor
    """
    # Get the number of points
    N = points.shape[0]

    # Get a random index
    ind = np.random.randint(N)

    # Find the nearest neighbor with the kd tree
    start_time = time.time()
    best_distance, best_index = kdtree.find_nearest_neighbor(points, ind)
    print("Time to find the nearest neighbour with the kd tree", time.time() - start_time, "seconds")

    # Return the distance and the index of the nearest neighbor
    return best_distance, best_index


# Function that finds the k nearest neighbors of a point with a kd-tree
def find_k_nearest_neighbors_with_kd_tree(points: np.ndarray, kdtree: kd_tree, k: int = 8):
    """
    Find the k nearest neighbors for all the points in the dataset
    :param points: The coordinates of the points of the point cloud
    :param kdtree: The kd tree
    :param k: The number of neighbors
    :return: Return the indices and the distances of the k nearest neighbors
    """
    # Get the number of points
    N = points.shape[0]

    # Set a numpy array to store the indices of the k nearest neighbors
    indices = np.zeros((N, k)).astype('int32')

    # Set a numpy array to store the distances of the k nearest neighbors
    distances = np.zeros((N, k))

    # Find the k nearest neighbors for all the points with the kd tree
    start_time = time.time()
    for i in tqdm(range(N)):
        # Find the k nearest neighbors with the kd tree
        k_distances, k_indices = kdtree.find_k_nearest_neighbors(points, i, k)

        # Store the indices
        indices[i] = k_indices
        # Store the distances
        distances[i] = k_distances

    # Print the time
    print("Time to find the k nearest neighbours for all the points with the kd tree", time.time() - start_time,
          "seconds")

    # Return the indices and the distances
    return indices, distances


# Function that finds which dataset the user has chosen depending on the artificial point cloud choice
def getDataForUsersPointCloudChoice(artificial_pointcloud_choice: str):
    """
    Find which dataset the user has chosen depending on the artificial point cloud choice
    :param artificial_pointcloud_choice: The choice of the user
    :return: The pointcloud, the distance threshold, the epsilon and the minimum number of samples for the DBSCAN
    and a variable tha says if the pointcloud should be re-centered or not
    """
    # Check if the user has chosen the first dataset
    if artificial_pointcloud_choice == "1":
        # The user has chosen the playground dataset
        distance_threshold = 0.15
        epsilon = 2
        min_samples = 100
        recenter = True
        return pcds.the_play_ground, distance_threshold, epsilon, min_samples, recenter

    # Check if the user has chosen the second dataset
    elif artificial_pointcloud_choice == "2":
        # The user has chosen the researcher desk
        distance_threshold = 0.01
        epsilon = 0.01
        min_samples = 5
        recenter = False
        return pcds.the_researcher_desk, distance_threshold, epsilon, min_samples, recenter

    # Check if the user has chosen the third dataset
    elif artificial_pointcloud_choice == "3":
        # The use has chosen the adas lidar dataset
        distance_threshold = 0.7
        epsilon = 2
        min_samples = 20
        recenter = True
        return pcds.the_adas_lidar, distance_threshold, epsilon, min_samples, recenter

    # Check if the user has chosen the fourth dataset
    elif artificial_pointcloud_choice == "4":
        # The user has chosen the kitchen dataset (with walls)
        distance_threshold = 0.05
        epsilon = 0.05
        min_samples = 5
        recenter = False
        return pcds.tls_kitchen, distance_threshold, epsilon, min_samples, recenter

    # Check if the user has chosen the fifth dataset
    elif artificial_pointcloud_choice == "5":
        # The user has chosen the kitchen dataset (without walls)
        distance_threshold = 0.01
        epsilon = 0.04
        min_samples = 20
        recenter = True
        return pcds.tls_kitchen_sample, distance_threshold, epsilon, min_samples, recenter

    # Else the choice is not valid
    else:
        return False, False, False, False, False


def getDataForUsersStereoImagesChoice(stereo_images_choice: str):
    """
    Find which dataset the user has chosen depending on the stereo images choice
    :param stereo_images_choice: The choice of the user
    :return:
    """
    data_path = "data/"
    # Check if the user has chosen the first dataset
    if stereo_images_choice == "1":
        # The user has chosen the first dataset of the kitty dataset
        # Load the images in grayscale
        left_image = cv2.imread(data_path + stereoPairs.leftImageFolder + stereoPairs.kitty_78, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(data_path + stereoPairs.rightImageFolder + stereoPairs.kitty_78, cv2.IMREAD_GRAYSCALE)
        # Load only the left image in color
        left_image_color = cv2.imread(data_path + stereoPairs.leftImageFolder + stereoPairs.kitty_78)
        # Set the parameters for the disparity map
        window_size = 8
        previous_windows_num = 8
        # Set the parameters for the reconstruction
        baseline = np.array([0.54, 0, 0])
        # Set the parameters for the fov (we don't have the fov)
        fov = 0.0
        # Set the parameters for RANSAC
        d_threshold = 0.15
        # Set the parameters for de-noising (with clustering DBSCAN)
        eps = 0.1
        min_samples = 10
        # Set the parameters for the clustering DBSCAN
        clustering_eps = 0.2
        clustering_min_samples = 100
        # Set a boolean to say if the choice is from the kitti dataset
        usingKitti = True
        # Get the calibration matrices
        cam1, cam2 = loadCalib(data_path + stereoPairs.calibFolder + stereoPairs.calib_78)
        return left_image, right_image, left_image_color, window_size, previous_windows_num, baseline, fov, \
            d_threshold, eps, min_samples, clustering_eps, clustering_min_samples, usingKitti, cam1, cam2
    elif stereo_images_choice == "2":
        # The user has chosen the second dataset of the kitty dataset
        # Load the images in grayscale
        left_image = cv2.imread(data_path + stereoPairs.leftImageFolder + stereoPairs.kitty_89, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(data_path + stereoPairs.rightImageFolder + stereoPairs.kitty_89, cv2.IMREAD_GRAYSCALE)
        # Load only the left image in color
        left_image_color = cv2.imread(data_path + stereoPairs.leftImageFolder + stereoPairs.kitty_89)
        # Set the parameters for the disparity map
        window_size = 8
        previous_windows_num = 8
        # Set the parameters for the reconstruction
        baseline = np.array([0.54, 0, 0])
        # Set the parameters for the fov (we don't have the fov)
        fov = 0.0
        # Set the parameters for RANSAC
        d_threshold = 0.2
        # Set the parameters for de-noising (with clustering DBSCAN)
        eps = 0.1
        min_samples = 10
        # Set the parameters for the clustering DBSCAN
        clustering_eps = 0.5
        clustering_min_samples = 100
        # Set a boolean to say if the choice is from the kitti dataset
        usingKitti = True
        # Get the calibration matrices
        cam1, cam2 = loadCalib(data_path + stereoPairs.calibFolder + stereoPairs.calib_89)
        return left_image, right_image, left_image_color, window_size, previous_windows_num, baseline, fov, \
            d_threshold, eps, min_samples, clustering_eps, clustering_min_samples, usingKitti, cam1, cam2
    elif stereo_images_choice == "3":
        # The user has chosen the third dataset of the kitty dataset
        # Load the images in grayscale
        left_image = cv2.imread(data_path + stereoPairs.leftImageFolder + stereoPairs.kitty_102, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(data_path + stereoPairs.rightImageFolder + stereoPairs.kitty_102, cv2.IMREAD_GRAYSCALE)
        # Load only the left image in color
        left_image_color = cv2.imread(data_path + stereoPairs.leftImageFolder + stereoPairs.kitty_102)
        # Set the parameters for the disparity map
        window_size = 8
        previous_windows_num = 8
        # Set the parameters for the reconstruction
        baseline = np.array([0.54, 0, 0])
        # Set the parameters for the fov (we don't have the fov)
        fov = 0.0
        # Set the parameters for RANSAC
        d_threshold = 0.2
        # Set the parameters for de-noising (with clustering DBSCAN)
        eps = 0.1
        min_samples = 10
        # Set the parameters for the clustering DBSCAN
        clustering_eps = 0.4
        clustering_min_samples = 100
        # Set a boolean to say if the choice is from the kitti dataset
        usingKitti = True
        # Get the calibration matrices
        cam1, cam2 = loadCalib(data_path + stereoPairs.calibFolder + stereoPairs.calib_102)
        return left_image, right_image, left_image_color, window_size, previous_windows_num, baseline, fov, \
            d_threshold, eps, min_samples, clustering_eps, clustering_min_samples, usingKitti, cam1, cam2
    elif stereo_images_choice == "4":
        # The user has chosen the first dataset of the kitty dataset
        # Load the images in grayscale
        left_image = cv2.imread(data_path + stereoPairs.leftImageFolder + stereoPairs.storage_left,
                                cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(data_path + stereoPairs.rightImageFolder + stereoPairs.storage_right,
                                 cv2.IMREAD_GRAYSCALE)
        # Load only the left image in color
        left_image_color = cv2.imread(data_path + stereoPairs.leftImageFolder + stereoPairs.storage_left)
        # Set the parameters for the disparity map
        window_size = 8
        previous_windows_num = 5
        # Set the parameters for the reconstruction
        baseline = 0.2
        fov = 1.2
        # Set the parameters for RANSAC
        d_threshold = 0.1
        # Set the parameters for de-noising (with clustering DBSCAN)
        eps = 10
        min_samples = 50
        # Set the parameters for the clustering DBSCAN
        clustering_eps = 3.5
        clustering_min_samples = 30
        # Set a boolean to say if the choice is from the kitti dataset
        usingKitti = False
        # Get the calibration matrices (we don't have the calibration matrices)
        cam1, cam2 = None, None
        return left_image, right_image, left_image_color, window_size, previous_windows_num, baseline, fov, \
            d_threshold, eps, min_samples, clustering_eps, clustering_min_samples, usingKitti, cam1, cam2
    else:
        # The choice is not valid
        return False, False, False, False, False, False, False, False, False, False, False, False, False, False, False


def points_distance(p1, p2) -> float:
    """
    Calculate the distance between two getPoints
    :param p1: The first point
    :param p2: The second point
    :return: The distance between the two getPoints
    """
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)


# Calculate the distance between a point and an edge
def point_to_edge_distance(point, edge) -> float:
    """
    Calculate the distance between a point and an edge
    :param point: The point
    :param edge: The edge
    :return: The distance between the point and the edge
    """
    v1 = [edge.p1.x - point.x, edge.p1.y - point.y, edge.p1.z - point.z]
    v2 = [edge.p1.x - edge.p2.x, edge.p1.y - edge.p2.y, edge.p1.z - edge.p2.z]
    return np.linalg.norm(np.cross(v1, v2)) / np.linalg.norm(v2)
