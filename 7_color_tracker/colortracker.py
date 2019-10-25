import os
import cv2
import numpy as np

path = "./sequences"
image_prefix = "frame"
image_suffix = ".jpg"
output_folder = "./output/"
lower_color = np.array([29, 43, 126], dtype='uint8')
upper_color = np.array([88, 255, 255], dtype='uint8')
erosion_iterations = 2
dilation_iterations = 2
area_threshold = 3000
max_tracking_lines = 5
images_paths = []

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

num_images = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

for i in range(num_images):
    images_paths.append(path + "/" + image_prefix + str(i+1) + image_suffix)

tracking_lines = []

count = 0
for image_path in images_paths:
    image = cv2.imread(image_path)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower_color, upper_color)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=erosion_iterations)
    dilation = cv2.dilate(erosion, kernel, iterations=dilation_iterations)

    contours, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours_areas = [cv2.contourArea(contour) for contour in contours]
    largest_area = max(contours_areas)
    largest_contour = contours[contours_areas.index(largest_area)]

    M = cv2.moments(array=largest_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    if largest_area > area_threshold:
        cv2.circle(image, (cx, cy), 30, (0, 255, 0), 3)
        tracking_lines.append([cx, cy])
        tracking_lines = tracking_lines[1:max_tracking_lines] if len(tracking_lines) > max_tracking_lines else \
            tracking_lines

        if len(tracking_lines) > 1:
            cv2.polylines(image, np.int32([np.asarray(tracking_lines)]), False, (255, 0, 0), 3)
    else:
        tracking_lines = []

    cv2.imwrite(output_folder + image_prefix + str(count+1) + image_suffix, image)
    count += 1
    print(str(round((count / len(images_paths) * 100), 2))  + "%")
