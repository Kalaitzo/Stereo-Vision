import cv2
import numpy as np


# Function that loads the calibration parameters
def loadCalib(calib_file):
    """
    A function that loads the calibration parameters from the calibration file
    :param calib_file: The path to the calibration file
    :return: The calibration matrices for the left and right cameras
    """
    matrix_type_1 = 'P2'
    matrix_type_2 = 'P3'

    with open(calib_file, 'r') as f:
        fin = f.readlines()
        for line in fin:
            if line[:2] == matrix_type_1:
                calib_matrix_1 = np.array(line[4:].strip().split(" ")).astype('float32').reshape(3, -1)
            elif line[:2] == matrix_type_2:
                calib_matrix_2 = np.array(line[4:].strip().split(" ")).astype('float32').reshape(3, -1)

    cam1 = calib_matrix_1[:, :3]
    cam2 = calib_matrix_2[:, :3]

    return cam1, cam2


# Function that uses the calibration parameters to calculate the points in 3D space
def disparity2points(disparity, img_left, cam1, cam2):
    """
    A function that uses the calibration parameters to calculate the points in 3D space
    :param cam1: The projection matrix for the left camera
    :param cam2: The projection matrix for the right camera
    :param disparity:
    :param img_left: The left image
    :return: The points in 3D space
    """
    baseline = np.array([0.54, 0., 0.])

    rev_proj_matrix = np.zeros((4, 4))

    cv2.stereoRectify(cameraMatrix1=cam1, cameraMatrix2=cam2,
                      distCoeffs1=0, distCoeffs2=0,
                      imageSize=img_left.shape[:2],
                      R=np.identity(3), T=baseline,
                      R1=None, R2=None,
                      P1=None, P2=None, Q=rev_proj_matrix)

    points = cv2.reprojectImageTo3D(disparity, rev_proj_matrix)

    # Reflect on x-axis
    reflect_matrix = np.identity(3)
    reflect_matrix[0] *= -1
    points = np.matmul(points, reflect_matrix)

    # Extract colors from image
    colors = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

    # Mask colors
    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colors = colors[mask] / 255.0

    # # Filter by dimension
    # idx = np.fabs(out_points[:, 0]) < 4.5
    # out_points = out_points[idx]
    # out_colors = out_colors.reshape(-1, 3)
    # out_colors = out_colors[idx] / 255.0

    return out_points, out_colors
