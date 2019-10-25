import os
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True, help = "Path to where the images resides")
ap.add_argument("-o", "--out", required = True, help = "Path to where the video will be stored")
args = vars(ap.parse_args())

path = args["images"]
output_folder = args["out"]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

imagesPaths = os.listdir(path)
imagesPaths.sort()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


count = 0
for imagePath in imagesPaths:
    image = cv2.imread(path + "/" + imagePath)

    rects, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(32, 32), scale=1.05)

    for r in rects:
        cv2.rectangle(image, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 255, 0), 3)

    cv2.imwrite(output_folder + "/" + imagePath, image)
    count += 1
    print(str(round((count/len(imagesPaths))*100), 2) + "%")

