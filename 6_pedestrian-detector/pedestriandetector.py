import os
import argparse
import cv2

faceCascadePath = "./classifiers/haarcascade_frontalface_default.xml"
eyeCascadePath = "./classifiers/haarcascade_eye.xml"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True, help = "Path to where the images resides")
ap.add_argument("-o", "--out", required = True, help = "Path to where the video will be stored")
args = vars(ap.parse_args())
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


path = args["video"]
output_folder = args["video"]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

num_images = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

for i in range(num_images):
    images.append(Image.open(path + "/" + image_prefix + str(i) + image_suffix))

out = cv2.VideoWriter(output_folder + name + ".mp4", fourcc, 33.0, (640, 480))

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
