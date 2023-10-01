import cv2
import numpy as np

def remove_distortion(img,
           focal_length_x=100.0,  # Expressed in mm/pixel
           focal_length_y=100.0,
           p1=-0.1,
           p2=0.0):
    height, width = img.shape[:2]
    principal_point_x = width / 2
    principal_point_y = height / 2
    # Define the camera matrix (intrinsic matrix)
    camera_matrix = np.array([[focal_length_x, 0, principal_point_x],
                              [0, focal_length_y, principal_point_y],
                              [0, 0, 1]], dtype=np.float32)

    # Define the distortion coefficients (k1, k2, p1, p2, k3)
    k1 = 0
    k2 = 0
    k3 = 50
    distortion_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    corrected_img = cv2.undistort(img, camera_matrix, distortion_coeffs)
    return corrected_img

def trapezoidal_warping(image,
         X_stretching_in_pixels=13,
         Y_stretching_in_pixels=13):

    # Define the source rectangle (four corners of the rectangle)
    src = np.array([[0, 0],                          # Top-left corner
                    [image.shape[1], 0],             # Top-right corner
                    [0, image.shape[0]],             # Bottom-left corner
                    [image.shape[1], image.shape[0]]],  # Bottom-right corner
                    dtype=np.float32)

    # Define the destination trapezoid (stretch the top)
    # Here, we're making the top side shorter
    x1_dest, y1_dest = X_stretching_in_pixels, 0                    # Top-left corner
    x2_dest, y2_dest = image.shape[1]-X_stretching_in_pixels, 0       # Top-right corner
    x3_dest, y3_dest = 0, image.shape[0]+Y_stretching_in_pixels      # Bottom-left corner
    x4_dest, y4_dest = image.shape[1], image.shape[0]+Y_stretching_in_pixels  # Bottom-right corner

    dst = np.array([[x1_dest, y1_dest],
                    [x2_dest, y2_dest],
                    [x3_dest, y3_dest],
                    [x4_dest, y4_dest]],
                    dtype=np.float32)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Apply the perspective transformation to the image
    warped_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]+Y_stretching_in_pixels))
    
    return warped_image
