import os
import cv2
import numpy as np
import argparse

erosion_iterations = 2
dilation_iterations = 2
area_threshold = 3000
max_tracking_lines = 5

ap = argparse.ArgumentParser()
ap.add_argument("--video", required = True, help = "Path to where the video file resides")
ap.add_argument("--min_values", required = True, help = "Lower HSV values", nargs="+", type=int)
ap.add_argument("--max_values", required = True, help = "Upper HSV values", nargs="+", type=int)
ap.add_argument("--output", required = False, help = "Path where the video will be deployed (optional)")
args = vars(ap.parse_args())

path = args["video"]
lower_color = np.array(args["min_values"], dtype='uint8')
upper_color = np.array(args["max_values"], dtype='uint8')
output_folder = args["output"]

cap = cv2.VideoCapture(path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/100)

if output_folder:
    fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    out = cv2.VideoWriter(output_folder, fourcc, 24.0, (width, height))

tracking_lines = []

count = 0
while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)
        image_hsv = cv2.GaussianBlur(image_hsv, np.shape(kernel), cv2.IMREAD_UNCHANGED)
        mask = cv2.inRange(image_hsv, lower_color, upper_color)

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

        cv2.imshow("Frame", image)
        if cv2.waitKey(1) == ord('q'):
            break
        if output_folder:
            out.write(image)
        count += 1
        print(str(round((count/frame_count)*100, 2)) + "%")
    else:
        break

cap.release()
if output_folder:
    out.release()
cv2.destroyAllWindows()
