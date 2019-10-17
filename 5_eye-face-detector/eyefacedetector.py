import argparse
import cv2

faceCascadePath = "./classifiers/haarcascade_frontalface_default.xml"
eyeCascadePath = "./classifiers/haarcascade_eye.xml"

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required = True, help = "Path to where the video file resides")
args = vars(ap.parse_args())

path = args["video"]

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
        cv2.rectangle(frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 3)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", gray)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
