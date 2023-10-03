'''Trapezoidal warping of a sequence of images.'''

import cv2
import numpy as np
import os

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except:
    IN_COLAB = False
logger.info(f"Running in Google Colab: {IN_COLAB}")

def image_trapezoidal_warping(image,
         X_stretching_in_pixels=13,
         Y_stretching_in_pixels=13):

    filling_val = int(np.min(image))

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
    warped_image = cv2.warpPerspective(src=image,
                                       M=matrix,
                                       dsize=(image.shape[1], image.shape[0]+Y_stretching_in_pixels),
                                       borderValue=filling_val)
    
    return warped_image

def __sequence_trapezoidal_warping(
        X_stretching_in_pixels=13,
        Y_stretching_in_pixels=13,
        input_sequence_prefix="/tmp/input",
        output_sequence_prefix="/tmp/output",
        img_extension=".png"):
    list_of_imagenames = np.array(
        [img for img in os.listdir(input_sequence_prefix)
         if img_extension in img])
    total_imgs = len(list_of_imagenames)
    for image_name in list_of_imagenames:
        img_name = f"{input_sequence_prefix}/{image_name}"
        image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        logger.debug(f"Warping {img_name} {image.dtype} {np.min(image)} {np.max(image)} {np.average(image)}")
        warped_img = image_trapezoidal_warping(
            image,
            X_stretching_in_pixels=13,
            Y_stretching_in_pixels=13)
        img_name = f"{output_sequence_prefix}/{image_name}"
        logger.debug(f"Saving {img_name} {warped_img.dtype} {np.min(warped_img)} {np.max(warped_img)} {np.average(warped_img)}")
        cv2.imwrite(img_name, warped_img)

if __name__ == "__main__":

    def int_or_str(text):
        '''Helper function for argument parsing.'''
        try:
            return int(text)
        except ValueError:
            return text

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.description = __doc__

    parser.add_argument("-i", "--input", type=int_or_str,
                        help="Prefix of the input image sequence",
                        default="/tmp/input")
    
    parser.add_argument("-o", "--output", type=int_or_str,
                        help="Prefix of the output image sequence",
                        default="/tmp/output")

    parser.add_argument("-e", "--extension", type=int_or_str,
                        help="Image extension",
                        default=".png")

    parser.add_argument("-y", "--Y_stretching", type=int_or_str,
                        help="Stretching of the image in the Y axis in pixels",
                        default=13)

    parser.add_argument("-x", "--X_stretching", type=int_or_str,
                        help="Stretching of the image in the X axis in pixels",
                        default=13)

    args = parser.parse_args()
    
    sequence_trapezoidal_warping(
        X_stretching_in_pixels=args.X_stretching,
        Y_stretching_in_pixels=args.Y_stretching,
        input_sequence_prefix=args.input,
        output_sequence_prefix=args.output,
        img_extension=args.extension)

    logger.info(f"Your files should be in {args.output}")


'''
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
'''
