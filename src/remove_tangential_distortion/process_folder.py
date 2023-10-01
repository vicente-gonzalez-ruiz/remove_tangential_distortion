import numpy as np
import cv2
import os
import tqdm



def detect_seq(
        input_sequence_prefix="/tmp/input/",
        output_sequence_prefix="/tmp/output/",
        img_extension=".jpg",
        first_img_index=0,
        last_img_index=120):

    list_of_imagenames = np.array([img for img in os.listdir(input_sequence_prefix) if img_extension in img])
    total_imgs = len(list_of_imagenames)
    founds = 0
    for image_name in list_of_imagenames:
        img_name = f"{input_sequence_prefix}/{image_name}"
        with mp_pose.Pose(static_image_mode=True) as pose_tracker:
            img = cv2.imread(img_name)
            result = pose_tracker.process(image=img)
            pose_landmarks = result.pose_landmarks
            pose_img = img.copy()
            if pose_landmarks is not None:
              mp_drawing.draw_landmarks(image=pose_img,
                  landmark_list=pose_landmarks,
                  connections=mp_pose.POSE_CONNECTIONS)
              pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)
              founds += 1
              found = True
              logger.info(f"Pose found in {image_name} (total {founds} of {total_imgs} possible poses found)")
            else:
              found = False
              logger.debug(f"Pose not found in {image_name}")

            if IN_COLAB:
                cv2_imshow(pose_img)

            cv2.imwrite(f"{output_sequence_prefix}{image_name}", pose_img)w
            
    ratio = founds/total_imgs

    logger.info(f"founds={founds} of {total_imgs}, ratio={ratio}")
    return ratio

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
                        default="/tmp/input/")
    
    parser.add_argument("-o", "--output", type=int_or_str,
                        help="Prefix of the output image sequence",
                        default="/tmp/output/")

    parser.add_argument("-e", "--extension", type=int_or_str,
                        help="Image extension",
                        default=".jpg")

    parser.add_argument("-f", "--first", type=int_or_str,
                        help="Index of the first image",
                        default=0)

    parser.add_argument("-l", "--last", type=int_or_str,
                        help="Index of the last image",
                        default=120)

    args = parser.parse_args()
    
    detect_seq(
        input_sequence_prefix=args.input,
        output_sequence_prefix=args.output,
        img_extension=args.extension,
        first_img_index=args.first,
        last_img_index=args.last)
