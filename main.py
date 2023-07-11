import time
import numpy as np
import open3d as o3d
import utility as util
import KittiReconstruction as KR
import pcdManipulation as pcdManip
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from pcds import *
from ransac import ransac
from kd_tree import kd_tree
from BallPivot import BallPivotAlgorithm
from open3d.visualization.rendering import Camera
from open3d.visualization.gui import MouseEvent, KeyEvent


defaultUnlit = rendering.MaterialRecord()
defaultUnlit.shader = "defaultLit"  # Change to "defaultUnlit" to see without lighting

data_path = "./data/"  # Set the data path
point_clouds_path = data_path + pointcloud_folder  # Set the path for the point clouds

g = 9.81  # Set the gravity constant


class AppWindow:

    def __init__(self, width, height, window_name="Stereo Reconstruction"):
        # window size
        self.w_width = width
        self.w_height = height
        self.first_click = True

        # boilerplate - initialize window & scene
        self.window = gui.Application.instance.create_window(window_name, width, height)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)

        # basic layout
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)

        # set mouse and key callbacks
        self._scene.set_on_key(self._on_key_pressed)
        # self._scene.set_on_mouse(self._on_mouse_pressed)

        # set up camera
        bounds = self._scene.scene.bounding_box
        center = bounds.get_center()
        self._scene.look_at(center, center - [0, 0, 12], [0, 1, 0])

        self.geometries = {}  # Initialize the geometries dictionary

        self.objects = {}  # Initialize a dictionary for the "objects" in the scene

        self.segments = {}  # Initialize a dictionary for the segments detected by the RANSAC algorithm

        self.bounding_boxes = {}  # Initialize a dictionary for the bounding boxes of each object

        self.segment_models = {}  # Initialize a dictionary for the planes models for each segment

        self.outliers = {}  # Initialize a dictionary for each segment's outliers point cloud

        self.rest = None  # Initialize a variable for the rest of the point cloud

        self.planes_to_remove = None  # Initialize the amount of segments we want to remove

        self.aabb_on = False  # Initialize a variable that will tell us if the AABB is on or off

        self.outliers_on = False  # Initialize a variable that will tell us if the outliers are on or off

        self.inliers_on = False  # Initialize a variable that will tell us if the inliers are on or off

        self.original_pointcloud = None  # Initialize a variable for the original point cloud

        self.triangulation_done = False  # Initialize a variable to check if the triangulation has been done

        self.original_points = np.array([])  # Initialize a variable to keep the points of the original point cloud

        self.original_colors = np.array([])  # Initialize a variable to keep the colors of the original point cloud

        self.sphere_position = np.array([0.0, 0.0, 0.0])  # Initialize a variable to keep the positions of the sphere

        self.direction = np.array([0.2, 0.2, 0.0])  # Initialize a variable to keep the direction of the sphere

        self.planar_areas_removed = False  # Initialize a variable to check if the planar areas have been removed

        self.pointcloud_added = False  # Initialize a variable to check if the artificial point cloud has been added

        self.pointcloud_de_noised = False  # Initialize a variable to check if the pointcloud has been de-noised

        self.pointcloud_clustered = False  # Initialize a variable to check if the pointcloud has been clustered

        self.sphere_thrown = False  # Initialize a variable to check if the sphere has been thrown

        self.artificial_pointcloud_choice = None  # Initialize a variable for the choice of the point cloud

        self.stereo_images_choice = None  # Initialize a variable for the stereo pair choice

        self.disparity_map = None  # Initialize a variable for the disparity map

        self.eps = 0.01  # Initialize a variable for the epsilon value of the DBSCAN clustering algorithm

        self.min_points = 10  # Initialize a variable for the minimum points value of the DBSCAN clustering algorithm

        self.d_threshold = 0.01  # Initialize a variable for the distance threshold value of the RANSAC algorithm

        self.clustering_eps = 0.01  # Initialize a variable for the epsilon value of the DBSCAN clustering algorithm

        self.clustering_min_samples = 10  # Initialize a variable for the minimum points value of the DBSCAN

        self.calculated_average_distance = False  # Initialize a variable to check if the average distance is calculated

    def _on_layout(self, event):
        r = self.window.content_rect
        self._scene.frame = r

    def add_geometry(self, geometry, name):

        self._scene.scene.add_geometry(name, geometry, defaultUnlit)
        self.geometries[name] = geometry

    def remove_geometry(self, name):

        self._scene.scene.remove_geometry(name)

    def _on_key_pressed(self, event):
        # print("Pressed key: " + str(event.key))

        # I key: Show only the inliers of the point cloud
        if event.key == 105:
            if self.planar_areas_removed and not self.inliers_on:
                # Set the inliers_on variable to True
                self.inliers_on = True

                for key in self.outliers:
                    self.remove_geometry("outliers" + str(key))

                # If the outliers were on that means the inliers must be added again to the scene
                if self.outliers_on:
                    # Set the outliers_on variable to False
                    self.outliers_on = False
                    # Add the inliers to the scene
                    for key in self.segments:
                        self.add_geometry(self.segments[key], "segment" + str(key))

                print("Outliers removed")
                return gui.Widget.EventCallbackResult.HANDLED

            print("No inliers detected yet so there are none to show")
            return gui.Widget.EventCallbackResult.HANDLED

        # O key: Show only the outliers of the point cloud
        elif event.key == 111:
            if self.planar_areas_removed and not self.outliers_on:
                # Set the outliers_on variable to True
                self.outliers_on = True

                # Remove the inliers from the scene
                for key in self.segments:
                    self.remove_geometry("segment" + str(key))

                # If the inliers were on that means the outliers must be added again to the scene
                if self.inliers_on:
                    # Set the inliers_on variable to False
                    self.inliers_on = False
                    # Add the outliers to the scene (Only the last one is added to the scene)
                    self.add_geometry(self.outliers[self.planes_to_remove], "outliers" + str(self.planes_to_remove))

                print("Inliers removed")
                return gui.Widget.EventCallbackResult.HANDLED

            print("No outliers detected yet so there are none to show")
            return gui.Widget.EventCallbackResult.HANDLED

        # P key: Show both the outliers and the inliers of the point cloud
        elif event.key == 112:
            if self.planar_areas_removed and (self.inliers_on or self.outliers_on):
                # Clear the scene in order to remove the outliers and inliers
                self.clearScene()

                # Add again the inliers and outliers to the scene
                for key in self.segments:
                    self.add_geometry(self.segments[key], "segment" + str(key))

                # add the outliers to the scene (Only the last one is added to the scene)
                self.add_geometry(self.outliers[self.planes_to_remove], "outliers" + str(self.planes_to_remove))

                # Set the inliers_on and outliers_on variables to False
                self.inliers_on = False
                self.outliers_on = False

                print("Inliers and outliers added")
                return gui.Widget.EventCallbackResult.HANDLED

            return gui.Widget.EventCallbackResult.HANDLED

        # 1-9 keys: Detect as much planar areas as the number pressed
        elif 49 <= event.key <= 57:
            print("1-9 pressed")
            # Get the amount of planar areas to remove
            self.planes_to_remove = int(event.key - 48)
            print("Amount of planar areas to remove: ", self.planes_to_remove)

            # Get the original point cloud from the dictionary with the geometries
            pointcloud = self.original_pointcloud

            if not self.planar_areas_removed:
                # Remove the planar areas
                self.remove_planes(pointcloud, self.planes_to_remove)

            return gui.Widget.EventCallbackResult.HANDLED

        # S key: Stereo reconstruction Pointcloud
        elif event.key == 115:
            if not self.pointcloud_added:
                # Get the stereo images and the data needed for the reconstruction depending on the choice of the user
                left_image, right_image, left_color, \
                    window_size, previous_windows_num, \
                    baseline, fov,\
                    d_threshold, \
                    eps, min_samples,\
                    clustering_eps, \
                    clustering_min_samples,\
                    usingKitti, \
                    cam1, cam2 = util.getDataForUsersStereoImagesChoice(self.stereo_images_choice)

                # Visualize the stereo images
                util.visualize_stereo_frames(left_image, right_image)

                # Calculate the disparity map
                disparity_map = util.stereo2disparity(left_image, right_image, window_size, previous_windows_num)

                # Visualize the disparity map
                util.visualize_disparity_map(disparity_map)

                # Get the point cloud from the disparity map
                if not usingKitti:
                    # Get the rows and columns of the disparity map
                    rows, cols = disparity_map.shape

                    # Get the vertices and colors of the point cloud
                    vertices, colors = util.disparity2pointcloud(rows,
                                                                 cols,
                                                                 disparity_map,
                                                                 baseline,
                                                                 left_color,
                                                                 fov)
                else:
                    vertices, colors = KR.disparity2points(disparity_map, left_color, cam1, cam2)

                # Create the point cloud
                pcd = pcdManip.createPointCloudFromVertices(vertices, colors)

                if not pcd:
                    print("Something went wrong while creating the point cloud")
                    print("Please restart the program and try again")
                    return gui.Widget.EventCallbackResult.HANDLED

                # Re-center the point cloud
                pcdManip.recenterPointCloud(pcd)

                # Set the original point cloud and the original points, d_threshold, eps and min_samples (for DBSCAN
                # for de-noising and for clustering)
                self.original_pointcloud = pcd
                self.disparity_map = disparity_map
                self.original_points = np.asarray(pcd.points)
                self.original_colors = np.asarray(pcd.colors)
                self.d_threshold = d_threshold
                self.eps = eps
                self.min_samples = min_samples
                self.clustering_eps = clustering_eps
                self.clustering_min_samples = clustering_min_samples

                # Add the point cloud to the scene
                self.add_geometry(pcd, "original_pcd")

                # Set the pointcloud_added variable to True
                self.pointcloud_added = True

                print("Point cloud added")
                return gui.Widget.EventCallbackResult.HANDLED

            print("Point cloud already added")
            return gui.Widget.EventCallbackResult.HANDLED

        # A key: Artificial pointcloud
        elif event.key == 97:
            print("User wants to use an artificial point cloud")
            # If the artificial point cloud has not been added yet
            if not self.pointcloud_added:
                # Get the data depending on the choice of the user
                pcd, d_threshold, eps, min_samples, recenter = util.getDataForUsersPointCloudChoice(
                                                                                    self.artificial_pointcloud_choice)

                if not pcd:
                    print("No data found for the users choice")
                    print("Please restart the program and choose a valid option")
                    return gui.Widget.EventCallbackResult.HANDLED

                # Get the pointcloud that the use wants to use as the artificial pointcloud
                pcd = pcdManip.createPointCloudFromFile(pcd, point_clouds_path)

                # Re-center the point cloud
                if recenter:
                    pcd = pcdManip.recenterPointCloud(pcd)

                # Set the original point cloud, the original points, the distance threshold, the epsilon
                # and the minimum samples
                self.original_pointcloud = pcd
                self.original_points = np.asarray(pcd.points)
                self.d_threshold = d_threshold
                self.clustering_eps = eps
                self.clustering_min_samples = min_samples

                # Add the point cloud to the scene
                self.add_geometry(pcd, "original_pcd")

                # Set the variable to True
                self.pointcloud_added = True

                print("Point cloud added")
                return gui.Widget.EventCallbackResult.HANDLED

            print("Point cloud already added")
            return gui.Widget.EventCallbackResult.HANDLED

        # D key: Denoise the point cloud using the DBSCAN algorithm
        elif event.key == 100:
            if not self.pointcloud_de_noised:
                # Get the points from the original point cloud
                points = self.original_points

                # Get the colors from the original point cloud
                colors = self.original_colors

                # Clear the scene that has the original point cloud
                self.clearScene()

                # Denoise the point cloud
                new_points, new_colors = util.denoiseWithClustering(points, colors, self.eps, self.min_samples)

                # Create the new point cloud
                pcd = pcdManip.createPointCloudFromVertices(new_points, new_colors)

                # Set the original point cloud to the new point cloud (so that we can use that one for the next)
                self.original_pointcloud = pcd

                # Add the new point cloud to the scene
                self.add_geometry(pcd, "de_noised_pcd")

                # Set the variable to True
                self.pointcloud_de_noised = True

                print("Point cloud de-noised")
                return gui.Widget.EventCallbackResult.HANDLED

            print("Point cloud already de-noised")
            return gui.Widget.EventCallbackResult.HANDLED

        # M key: Denoise the disparity map and visualize it again
        elif event.key == 109:
            if self.disparity_map is not None:
                # Denoise the disparity map by applying a gaussian filter
                new_disparity_map = util.gaussianSmoothing(self.disparity_map)

                # Visualize the disparity map
                util.visualize_disparity_map(new_disparity_map, title="Disparity map de-noised")

                print("Disparity map de-noised (gaussian smoothing)")

            print("No disparity map found")
            return gui.Widget.EventCallbackResult.HANDLED

        # R key: Resets everything
        elif event.key == 114:
            # Clear the scene
            self.clearScene()

            # Reset all the initialize variables
            self.resetEverything()

            return gui.Widget.EventCallbackResult.HANDLED

        # C key: Cluster the point cloud using the DBSCAN algorithm
        elif event.key == 99:
            if not self.pointcloud_clustered:
                # If the outliers of the point cloud have not been detected yet use the original point cloud
                if not self.planar_areas_removed:
                    pcd = self.original_pointcloud
                else:
                    # Else the pointcloud is the last outlier
                    pcd = self.outliers[self.planes_to_remove]

                # Cluster the point cloud
                self.clusterPointCloud(pcd)

                # Add the objects, their bounding boxes and the rest of the pointcloud to the scene
                self.drawAfterClustering()

                # Set the variable to True
                self.pointcloud_clustered = True

                # Set the variable for the bounding boxes to true
                self.aabb_on = True

                print("Point cloud clustered")
                return gui.Widget.EventCallbackResult.HANDLED

            print("Point cloud already clustered")
            return gui.Widget.EventCallbackResult.HANDLED

        # N key: Hide the noise (rest of the point cloud)
        elif event.key == 110:
            if self.pointcloud_clustered:
                if "rest" in self.geometries:
                    del self.geometries["rest"]
                    self.remove_geometry("rest")

                    print("Noise hidden")
                    return gui.Widget.EventCallbackResult.HANDLED

                print("Noise is already hidden")
                return gui.Widget.EventCallbackResult.HANDLED

            print("Point cloud not clustered yet")
            return gui.Widget.EventCallbackResult.HANDLED

        # Z key: Show the noise (rest of the point cloud)
        elif event.key == 122:
            if self.pointcloud_clustered:
                # If the key rest is not in the geometries
                if "rest" not in self.geometries:
                    self.add_geometry(self.rest, "rest")

                    print("Noise added to the scene")
                    return gui.Widget.EventCallbackResult.HANDLED

                print("Noise is already shown")
                return gui.Widget.EventCallbackResult.HANDLED

            print("Point cloud not clustered yet")
            return gui.Widget.EventCallbackResult.HANDLED

        # B key: Shows the AABB
        elif event.key == 98:
            if self.pointcloud_clustered:
                if not self.aabb_on:
                    # Show the AABB
                    self.showAABB()

                    # Set the variable to True
                    self.aabb_on = True

                    print("The bounding boxes are shown")
                    return gui.Widget.EventCallbackResult.HANDLED

                print("The AABB is already shown")
                return gui.Widget.EventCallbackResult.HANDLED

            print("The point cloud is not clustered yet")
            return gui.Widget.EventCallbackResult.HANDLED

        # H key: Hides the AABB
        elif event.key == 104:
            if self.pointcloud_clustered:
                if self.aabb_on:
                    # Hide the AABB
                    self.hideAABB()

                    # Set the variable to False
                    self.aabb_on = False

                    print("The bounding boxes are hidden")
                    return gui.Widget.EventCallbackResult.HANDLED

                print("The AABB is already hidden")
                return gui.Widget.EventCallbackResult.HANDLED

            print("The point cloud is not clustered yet")
            return gui.Widget.EventCallbackResult.HANDLED

        # T key: Throw a ball
        elif event.key == 116:
            if not self.sphere_thrown:
                # Move the sphere
                self.move_sphere(self.direction)

                print("Sphere thrown")
                return gui.Widget.EventCallbackResult.HANDLED

            self.sphere_thrown = False
            print("Sphere already thrown")
            return gui.Widget.EventCallbackResult.HANDLED

        # F key: Throw a ball to the opposite direction
        elif event.key == 102:
            if not self.sphere_thrown:
                # Move the sphere
                self.move_sphere(np.negative(self.direction))

                print("Sphere thrown")
                return gui.Widget.EventCallbackResult.HANDLED

            self.sphere_thrown = False
            print("Sphere already thrown")
            return gui.Widget.EventCallbackResult.HANDLED

        # V key: Prints the geometries in the scene
        elif event.key == 118:
            print(self.geometries)
            return gui.Widget.EventCallbackResult.HANDLED

        # G key: Perform object triangulation
        elif event.key == 103:
            if not self.triangulation_done:
                # Clear the scene
                self.clearScene()

                # Load the point cloud for triangulation
                data = np.loadtxt(point_clouds_path + triangulation_pcd, skiprows=1)

                # Create a point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
                pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)
                pcd.normals = o3d.utility.Vector3dVector(data[:, 6:9])

                # Take 10000 random points from the point cloud
                points = np.asarray(pcd.points)[np.random.choice(np.asarray(pcd.points).shape[0], 10000, replace=False)]

                # Change the points of the point cloud to the random points
                pcd.points = o3d.utility.Vector3dVector(points)

                # Get the average distance between the points
                # Create a KDTree with the points
                kdtree = kd_tree(points=points)

                # Find the 7 nearest neighbors for each point in the point cloud
                indices, distances = util.find_k_nearest_neighbors_with_kd_tree(points, kdtree, 7)

                # Find the average distance of the 7 nearest neighbors for each point
                average_distances = np.mean(distances, axis=0)

                # Find the mean of the average distances
                mean_distance = np.mean(average_distances)

                # Create a ball pivot algorithm object
                ball_pivot = BallPivotAlgorithm(points, mean_distance)

                # Begin the algorithm at a random point
                triangles = ball_pivot.create_mesh(np.random.randint(0, len(points)))

                # Get the number of triangles
                num_triangles = len(triangles)
                print('Done creating mesh with {} triangles'.format(num_triangles))

                # Create a mesh object
                mesh = o3d.geometry.TriangleMesh()

                # Assign the vertices and triangles to the mesh
                mesh.vertices = o3d.utility.Vector3dVector(points)
                mesh.triangles = o3d.utility.Vector3iVector(triangles)

                # Re-center both the point cloud and the mesh
                pcd.translate(-pcd.get_center())
                mesh.translate(-mesh.get_center())

                # Paint the mesh (triangles) red
                mesh.paint_uniform_color([1, 0, 0])

                # Add the mesh and the point cloud to the scene
                self.add_geometry(mesh, "mesh_from_triangulation")
                self.add_geometry(pcd, "pointcloud_for_triangulation")

                # Set the variable to True
                self.triangulation_done = True
                print("Triangulation done")

                return gui.Widget.EventCallbackResult.HANDLED

            print("Triangulation already done")
            return gui.Widget.EventCallbackResult.HANDLED

        # Χ key: Find the average distance between the points of the original point cloud
        elif event.key == 120:
            if not self.calculated_average_distance:
                # Get the points of the point cloud
                points = np.asarray(self.original_pointcloud.points)

                # Create a KDTree with the points
                kdtree = kd_tree(points=points)

                # Find the 7 nearest neighbors for each point in the point cloud
                indices, distances = util.find_k_nearest_neighbors_with_kd_tree(points, kdtree, 7)

                # Find the average distance of the 7 nearest neighbors for each point
                average_distances = np.mean(distances, axis=0)

                # Find the mean of the average distances
                mean_distance = np.mean(average_distances)

                # Set the variable to True
                self.calculated_average_distance = True

                # Print the mean distance
                print("The average distance between the points is: {}".format(mean_distance))

                return gui.Widget.EventCallbackResult.HANDLED

            print("The average distance has already been calculated dont need to calculate it again")
            return gui.Widget.EventCallbackResult.HANDLED

        # Up arrow
        elif event.key == 265:
            pass

        # Down arrow
        elif event.key == 266:
            pass

        # T key:
        elif event.key == 116:
            pass

        # R key:
        elif event.key == 114:
            pass

        return gui.Widget.EventCallbackResult.HANDLED

    # Find the planes in the point cloud using the RANSAC algorithm
    def remove_planes(self, point_cloud, planes_to_remove):
        self.outliers[0] = point_cloud

        for i in range(planes_to_remove):
            # Get the segments and the models of the segments
            inliers, self.segment_models[i] = ransac(np.array(self.outliers[i].points), self.d_threshold, 5000)

            # Create a point cloud with the inliers
            self.segments[i] = self.outliers[i].select_by_index(inliers)

            # Pain the points of the segment in red
            self.segments[i].paint_uniform_color([1, 0, 0])

            # Get the outliers of the segment
            self.outliers[i+1] = pcdManip.createOutlierPointCloud(np.asarray(self.outliers[i].points), inliers)

            # Paint the outliers in blue
            self.outliers[i+1].paint_uniform_color([0, 0, 1])

            # Update the scene by adding the segment
            self.add_geometry(self.segments[i], "segment" + str(i))

        # Add only the last outliers to the scene
        self.add_geometry(self.outliers[planes_to_remove], "outliers" + str(planes_to_remove))

        # Remove the original point cloud from the scene
        self.remove_geometry("original_pcd")

        # Set the variable to true to indicate that the planar areas have been removed
        self.planar_areas_removed = True

        return gui.Widget.EventCallbackResult.HANDLED

    # Cluster the point cloud using the DBSCAN algorithm
    def clusterPointCloud(self, pcd):
        # Check if the pointcloud is more than 200000 points
        if len(pcd.points) > 200000:
            # If it is, then down-sample it
            pcd = pcdManip.downSamplePointcloud(pcd)

        # Cluster the point cloud using the DBSCAN algorithm
        labels = np.array(pcd.cluster_dbscan(eps=self.clustering_eps,
                                             min_points=self.clustering_min_samples,
                                             print_progress=True))

        # Get the max label
        max_label = np.max(labels)

        # Print the number of clusters
        print("Number of clusters: " + str(max_label + 1))

        # Store each cluster (object) in the objects dictionary
        for i in range(max_label):
            # Get the part of the point cloud that belongs to the cluster
            self.objects[i] = pcd.select_by_index(list(np.where(labels == i)[0]))

            # Create a random color for the object
            color = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]

            # Paint the object with the random color
            self.objects[i].paint_uniform_color(color)

            # Get the bounding box of the object and store it in the bounding_boxes dictionary
            self.bounding_boxes[i] = pcdManip.getAABB(self.objects[i])

        # Get the part of the point cloud that wasn't in any cluster
        self.rest = pcd.select_by_index(list(np.where(labels < 0)[0]))

        # Paint the rest of the point cloud black
        self.rest.paint_uniform_color([0, 0, 0])

    # A function that moves the sphere
    def move_sphere(self, speed=None):
        # Check if the user gave a speed for the sphere otherwise set it to [0.1, 0.1, 0.1]
        if speed is None:
            speed = [0.1, 0.1, 0.1]

        # Remove the sphere from the scene if it is already there
        if "sphere" in self.geometries:
            del self.geometries["sphere"]
            self.remove_geometry("sphere")

        # Create a sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)

        # Set the color of the sphere
        sphere.paint_uniform_color([0.1, 0.1, 0.7])

        # Check if the sphere is colliding with any bounding box
        collision, aabb, i, collision_point = self.detectCollision()

        # If there is a collision, calculate the new speed of the sphere
        if collision:
            # The new speed is the cross product of the original speed and the normal of the bounding box's side
            min_bound = aabb.get_min_bound()
            max_bound = aabb.get_max_bound()

            normal = [0, 0, 0]

            # Get the normal of the side of the bounding box (it helps that the bounding box is axis aligned)
            if collision_point[0] == min_bound[0]:
                normal = [1, 0, 0]
            elif collision_point[0] == max_bound[0]:
                normal = [-1, 0, 0]
            elif collision_point[1] == min_bound[1]:
                normal = [0, 1, 0]
            elif collision_point[1] == max_bound[1]:
                normal = [0, -1, 0]
            elif collision_point[2] == min_bound[2]:
                normal = [0, 0, 1]
            elif collision_point[2] == max_bound[2]:
                normal = [0, 0, -1]

            # Normalize the direction of the speed
            speed = speed / np.linalg.norm(speed)

            # Calculate the reflection vector (the new speed)
            sphere_speed = speed - 2 * np.dot(speed, np.array(normal)) * np.array(normal)

            # Change the direction of the speed
            self.direction = sphere_speed

            print(f"Collision Detected! with bounding box {i} at point {collision_point}")
        else:
            # Else the speed is the original speed that the user gave
            sphere_speed = speed

        # Move the sphere
        self.sphere_position = self.sphere_position + sphere_speed

        # Set the position of the sphere
        sphere.translate(self.sphere_position)

        # Add the sphere to the scene
        self.add_geometry(sphere, "sphere")

        # Set the variable to True
        self.sphere_thrown = True

    # Detect if the sphere is colliding with any bounding box
    def detectCollision(self):
        # Check with every bounding box
        for i in range(len(self.bounding_boxes)):
            aabb = self.bounding_boxes[i]

            # Get the sphere's center
            sphere_center = self.sphere_position

            # Get the sphere's radius (it is always 0.2)
            sphere_radius = 0.5

            # Get the closest point on the bounding box to the sphere
            x = max(aabb.get_min_bound()[0], min(sphere_center[0], aabb.get_max_bound()[0]))
            y = max(aabb.get_min_bound()[1], min(sphere_center[1], aabb.get_max_bound()[1]))
            z = max(aabb.get_min_bound()[2], min(sphere_center[2], aabb.get_max_bound()[2]))

            distance = np.sqrt(((x - sphere_center[0]) * (x - sphere_center[0])) +
                               ((y - sphere_center[1]) * (y - sphere_center[1])) +
                               ((z - sphere_center[2]) * (z - sphere_center[2])))

            if distance < sphere_radius:
                collision_point = np.array([x, y, z])
                return True, aabb, i, collision_point

        return False, None, None, None

    # Draw the scene after clustering the point cloud
    def drawAfterClustering(self):
        # First clear the scene
        self.clearScene()

        # Add the objects and their bounding boxes to the scene
        for i in range(len(self.objects)):
            self.add_geometry(self.objects[i], "object" + str(i))
            self.add_geometry(self.bounding_boxes[i], "bounding_box" + str(i))

        # Add the rest of the point cloud to the scene
        self.add_geometry(self.rest, "rest")

    # Reset all the variables of the class and the scene
    def resetEverything(self):
        # Reset all the initialize variables of the class
        if self.pointcloud_added:
            self.pointcloud_added = False

        if self.planar_areas_removed:
            self.planar_areas_removed = False

        if self.aabb_on:
            self.aabb_on = False

        if self.outliers_on:
            self.outliers_on = False

        if self.inliers_on:
            self.inliers_on = False

        if self.pointcloud_de_noised:
            self.pointcloud_de_noised = False

        if self.pointcloud_clustered:
            self.pointcloud_clustered = False

        if self.geometries:
            self.geometries = {}

        if self.outliers:
            self.outliers = {}

        if self.segments:
            self.segments = {}

        if self.segment_models:
            self.segment_models = {}

        if self.objects:
            self.objects = {}

        if self.bounding_boxes:
            self.bounding_boxes = {}

        if self.original_pointcloud:
            self.original_pointcloud = None

        if self.planes_to_remove:
            self.planes_to_remove = None

        if self.rest:
            self.rest = None

        if self.original_points.any():
            self.original_points = np.array([])

        if self.sphere_thrown:
            self.sphere_thrown = False

        if self.calculated_average_distance:
            self.calculated_average_distance = False

        self.d_threshold = 0.01
        self.eps = 0.01
        self.min_points = 10
        self.sphere_position = np.array([0, 0, 0])
        self.clustering_eps = 0.1
        self.clustering_min_samples = 10

        return gui.Widget.EventCallbackResult.HANDLED

    # Hide all the bounding boxes:
    def hideAABB(self):
        for key in self.bounding_boxes:
            # Remove the bounding box from the scene and the dictionary
            self.remove_geometry("bounding_box" + str(key))
            del self.geometries["bounding_box" + str(key)]

    def showAABB(self):
        for key in self.bounding_boxes:
            # Add the bounding box to the scene and the dictionary
            self.add_geometry(self.bounding_boxes[key], "bounding_box" + str(key))

    # Clear the scene
    def clearScene(self):
        if self.geometries:
            for key in self.geometries:
                self.remove_geometry(key)

            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.HANDLED


if __name__ == "__main__":
    gui.Application.instance.initialize()

    # Ask the user which point cloud to load
    print("Which point cloud do you want to use as the artificial point cloud?")
    print("Choose 1 for the play ground (NOT STABLE DUE TO THE LARGE NUMBER OF POINTS)")
    print("Choose 2 for the researchers desk")
    print("Choose 3 for the adas lidar")
    print("Choose 4 for the kitchen")
    print("Choose 5 for the kitchen (without the walls)")
    artificial_pointcloud_choice = input("Enter your choice: ")
    print("You chose: ", artificial_pointcloud_choice)

    print("Which pair of stereo images do you want to use for the stere reconstruction?")
    print("Choose 1 for the pair 78_10")
    print("Choose 2 for the pair 89_10")
    print("Choose 3 for the pair 102_10")
    print("Choose 4 for the pair of the storage room")
    stereo_images_choice = input("Enter your choice: ")
    print("You chose: ", stereo_images_choice)

    # Set the application window
    app = AppWindow(1280, 720)

    # Set the choice of the user for the artificial point cloud
    app.artificial_pointcloud_choice = artificial_pointcloud_choice

    # Set the choice of the user for the stereo images
    app.stereo_images_choice = stereo_images_choice

    # Run the application
    gui.Application.instance.run()
