import cv2

def remove(img,
           focal_length_x=100.0,  # Expressed in mm/pixel
           focal_length_y=100.0,
           p1=1.0,
           p2=0.0):
    height, width = img.shape[:2]
    principal_point_x = width / 2
    principal_point_y = height / 2
    # Define the camera matrix (intrinsic matrix)
    camera_matrix = np.array([[focal_length_x, 0, principal_point_x],
                              [0, focal_length_y, principal_point_y],
                              [0, 0, 1]], dtype=np.float32)

    # Define the distortion coefficients (k1, k2, p1, p2, k3)
    k1 = k2 = k3 = 0
    distortion_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    corrected_img = cv2.undistort(img, camera_matrix, distortion_coeffs)
    return corrected_img
