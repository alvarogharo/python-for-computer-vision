from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
min_confidence = 0.5
east = "frozen_east_text_detection.pb"
padding = 0.0


def create_boxes(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		for x in range(0, numCols):

			if scoresData[x] < min_confidence:
				continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	return (rects, confidences)


def find_text(image):
	orig = image.copy()
	(origH, origW) = image.shape[:2]

	(newW, newH) = (320, 320)
	rW = origW / float(newW)
	rH = origH / float(newH)

	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	net = cv2.dnn.readNet(east)

	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
								 (123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(rects, confidences) = create_boxes(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	results = []

	for (startX, startY, endX, endY) in boxes:
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		dX = int((endX - startX) * padding)
		dY = int((endY - startY) * padding)

		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))

		# extract the actual padded ROI
		roi = orig[startY:endY, startX:endX]

		config = ("-l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(roi, config=config)

		results.append(((startX, startY, endX, endY), text))

	results = sorted(results, key=lambda r: r[0][1])
	return results


def print_results_and_show(results):
	for ((startX, startY, endX, endY), text) in results:
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		output = orig.copy()
		cv2.rectangle(output, (startX, startY), (endX, endY),
					  (255, 255, 255), 2)
		cv2.putText(output, text, (startX, startY - 20),
					cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

		# show the output image
		cv2.imshow("Text Detection", output)
		cv2.waitKey(1)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
ap.add_argument("-v", "--video", type=str, help="path to input video")
ap.add_argument("-e", "--east", type=str, help="east clasiffier weights")
ap.add_argument("-t", "--tesseract", type=str, help="teserract executable path")
args = vars(ap.parse_args())

if args["east"]:
	east = args["east"]

if args["tesseract"]:
	pytesseract.pytesseract.tesseract_cmd = args["tesseract"]

if args["image"]:
	image = cv2.imread(args["image"])
	orig = image.copy()

	results = find_text(image)

	print_results_and_show(results)
elif args["video"]:
	cap = cv2.VideoCapture(args["video"])

	while cap.isOpened():
		ret, image = cap.read()

		if not ret:
			break
		orig = image.copy()

		results = find_text(image)

		print_results_and_show(results)
else:
	print("Any source path declared on run arguments")

