import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="Path to where the video to track is stored")
ap.add_argument("--tracker", required=False, help="Path to where the video to track is stored")
args = vars(ap.parse_args())

video = args["image"]
tracker_type = args["tracker"]

cap = cv2.VideoCapture(video)
ret, frame = cap.read()
initBB = cv2.selectROI("Object to track", frame)
cv2.destroyWindow("Object to track")

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create(),
    "kcf": cv2.TrackerKCF_create(),
    "boosting": cv2.TrackerBoosting_create(),
    "mil": cv2.TrackerMIL_create(),
    "tld": cv2.TrackerTLD_create(),
    "medianflow": cv2.TrackerMedianFlow_create(),
    "mosse": cv2.TrackerMOSSE_create()
}

if tracker_type:
    tracker = OPENCV_OBJECT_TRACKERS[tracker_type]
else:
    tracker = cv2.TrackerKCF_create()
tracker.init(frame, initBB)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    (success, box) = tracker.update(frame)

    if success:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    cv2.waitKey(25)
