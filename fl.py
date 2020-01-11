# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
vid = cv2.VideoCapture(0)




while True:
	ret, image = vid.read()
	# image = cv2.imread(args["image"])
	image = imutils.resize(image, width=600)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image

		count = 0

		pface = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
		pmouth = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}

		for (x, y) in shape:
			cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
			count += 1		
   			if count is 1:
				pface['p1x'] = x
				pface['p1y'] = y

			if count is 17:
				pface['p2x'] = x
				pface['p2y'] = y
    
			if count is 49:
				pmouth['p1x'] = x
				pmouth['p1y'] = y

			if count is 55:
				pmouth['p2x'] = x
				pmouth['p2y'] = y

		dface = math.sqrt( ((pface['p1x']-pface['p2x'])**2)+((pface['p1y']-pface['p2y'])**2) )
		dmouth = math.sqrt( ((pmouth['p1x']-pmouth['p2x'])**2)+((pmouth['p1y']-pmouth['p2y'])**2) )
		print(dface / dmouth)

		cv2.line(image, (5,10), (60,45), (0,255,0), 2)

	# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", image)
	# cv2.waitKey(0)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

vid.release();
cv2.destroyAllWindows();
