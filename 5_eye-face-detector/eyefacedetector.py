import argparse
import cv2


faceCascadePath = "./classifiers/haarcascade_frontalface_default.xml"
eyeCascadePath = "./classifiers/haarcascade_eye.xml"

ap = argparse.ArgumentParser()
ap.add_argument("-video", required = True, help = "Path to where the video file resides")
ap.add_argument("-out", required = True, help = "Path to output the results")
args = vars(ap.parse_args())
fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")


path = args["video"]
output_folder = args["out"]

cap = cv2.VideoCapture(path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(output_folder, fourcc, 24.0, (width, height))

count = 0
while cap.isOpened() & (count < frame_count):

    ret, frame = cap.read()

    if ret:
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
        count += 1
        print(str(round((count/frame_count)*100, 2)) + "%")
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
