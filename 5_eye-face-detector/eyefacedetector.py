import argparse
import cv2


faceCascadePath = "./classifiers/haarcascade_frontalface_default.xml"
eyeCascadePath = "./classifiers/haarcascade_eye.xml"

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required = True, help = "Path to where the video file resides")
ap.add_argument("-o", "--out", required = True, help = "Path to output the results")
args = vars(ap.parse_args())
fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")


path = args["video"]
output_folder = args["out"]

out = cv2.VideoWriter(output_folder, fourcc, 24.0, (640, 266))

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
