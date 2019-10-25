import os
import argparse
import cv2
import numpy as np

faceCascadePath = "./classifiers/haarcascade_frontalface_default.xml"
eyeCascadePath = "./classifiers/haarcascade_eye.xml"
output_folder = "./output/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required = True, help = "Path to where the video file resides")
args = vars(ap.parse_args())
fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")


path = args["video"]
name = path.split("/")
name = name[len(name)-1].split(".")[0]

out = cv2.VideoWriter(output_folder + name + ".avi", fourcc, 24.0, (640, 266))

cap = cv2.VideoCapture(path)
while cap.isOpened():
    ret, frame = cap.read()

    faceCascade = cv2.CascadeClassifier(faceCascadePath)
    rects_face = faceCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    eyeCascade = cv2.CascadeClassifier(eyeCascadePath)
    rects_eye = eyeCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    for r in rects_face:
        cv2.rectangle(frame, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 255, 0), 3)

    for r in rects_eye:
        cv2.rectangle(frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255, 0, 0), 3)

    out.write(frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
