import argparse
import os
import cv2
from matplotlib import pyplot as plt

min_number_of_matches = 300

ap = argparse.ArgumentParser()
ap.add_argument("-query", required = True, help = "Path to where the image to match is stored")
ap.add_argument("-covers", required = True, help = "Path to folder of candidate images")
args = vars(ap.parse_args())

main_image_path = args["query"]
covers_folder = args["covers"]

orb = cv2.ORB_create(nfeatures=2000)
main_image = cv2.imread(main_image_path, cv2.IMREAD_GRAYSCALE)
main_image_kps, main_image_des = orb.detectAndCompute(main_image, None)

images_paths = os.listdir(covers_folder)
images_paths.sort()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

max_num_of_matches = 0
max_matches_img = None
max_matches_img_kps = None
max_matches_img_matches = None

for image_path in images_paths:
    image = cv2.imread(covers_folder + "/" + image_path, cv2.IMREAD_GRAYSCALE)
    image_kps, image_des = orb.detectAndCompute(image, None)
    matches = bf.match(main_image_des, image_des)
    num_of_matches = len(matches)

    if num_of_matches > max_num_of_matches:
        max_num_of_matches = num_of_matches
        max_matches_img = image
        max_matches_img_kps = image_kps
        max_matches_img_matches = matches

if max_num_of_matches >= min_number_of_matches:
    img3 = None
    img3 = cv2.drawMatches(main_image, main_image_kps, max_matches_img, max_matches_img_kps,
                           max_matches_img_matches, img3, flags=2)
    plt.imshow(img3), plt.show()
else:
    print("Any cover has enough matches with the query image")
