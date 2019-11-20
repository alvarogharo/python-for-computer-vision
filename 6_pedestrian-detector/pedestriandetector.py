import os
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-images", required = True, help = "Path to where the images resides")
ap.add_argument("-out", required = True, help = "Path to where the video will be stored")
args = vars(ap.parse_args())

path = args["images"]
output_folder = args["out"]

imagesPaths = os.listdir(path)
imagesPaths.sort()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
image_size = np.shape(cv2.imread(path + "/" + imagesPaths[0]))[0:2]

out = cv2.VideoWriter(output_folder, fourcc, 24.0, (image_size[1], image_size[0]))
count = 0
for imagePath in imagesPaths:
    image = cv2.imread(path + "/" + imagePath)
    rects, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(32, 32), scale=1.05)

    for r in rects:
        cv2.rectangle(image, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 255, 0), 3)

    out.write(image)
    count += 1
    print(str(round((count/len(imagesPaths))*100, 2)) + "%")

out.release()
cv2.destroyAllWindows()
