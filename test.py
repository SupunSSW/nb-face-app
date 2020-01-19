from imutils import face_utils
import imutils
import dlib
import cv2
import math
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def coords(shape, dtype="int"):
    	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

vid = cv2.VideoCapture(0)

while True:
	ret, image = vid.read()
	# image = cv2.imread('example_03.jpg')
	image = imutils.resize(image, width=600)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)
   
	if rects:
		print "==============================="
	else:
		pass
	
	exit = False

	for (i, rect) in enumerate(rects):	
  
		if i is 0:
			shape = predictor(gray, rect)
			# shape = coords(shape)

			(x, y, w, h) = face_utils.rect_to_bb(rect)
			roi = image[y:y+h,x:x+w]
	
			# # snapshot of face
			cv2.imwrite("face1.jpg",roi)
			# cv2.imwrite("face2.jpg",roi)
			# cv2.imwrite("face3.jpg",roi)
   
			if roi.size:
				exit = True
			
			# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	if exit:
		break


# while True:
image = cv2.imread('face1.jpg')
image = imutils.resize(image, width=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
	shape = predictor(image, rect)
	shape = coords(shape)

	# cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
	# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	count = 0
	pchin = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
	peye = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
	pmid = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
	pmouth = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}

	for (x, y) in shape:
		cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
		count += 1
		if count is 4:
			pchin['p1x'] = x
			pchin['p1y'] = y
		elif count is 14:
			pchin['p2x'] = x
			pchin['p2y'] = y
		if count is 31:
				pmid['p1x'] = x
				pmid['p1y'] = y
		elif count is 37:
			peye['p1x'] = x
			peye['p1y'] = y
		if count is 46:
				peye['p2x'] = x
				peye['p2y'] = y
		elif count is 49:
			pmouth['p1x'] = x
			pmouth['p1y'] = y
		if count is 55:
				pmouth['p2x'] = x
				pmouth['p2y'] = y

	d1chin = math.sqrt( ((pchin['p1x']-pmid['p1x'])**2)+((pchin['p1y']-pmid['p1y'])**2) )
	cv2.line(image, (pchin['p1x'],pchin['p1y']), (pmid['p1x'],pmid['p1y']), (0,255,0), 2)

	d2chin = math.sqrt( ((pchin['p2x']-pmid['p1x'])**2)+((pchin['p2y']-pmid['p1y'])**2) )
	cv2.line(image, (pchin['p2x'],pchin['p2y']), (pmid['p1x'],pmid['p1y']), (0,255,0), 2)

	deye = math.sqrt( ((peye['p1x']-peye['p2x'])**2)+((peye['p1y']-peye['p2y'])**2) )
	cv2.line(image, (peye['p2x'],peye['p2y']), (peye['p1x'],peye['p1y']), (255,0, 0), 2)

	dmouth = math.sqrt( ((pmouth['p1x']-pmouth['p2x'])**2)+((pmouth['p1y']-pmouth['p2y'])**2) )
	cv2.line(image, (pmouth['p2x'],pmouth['p2y']), (pmouth['p1x'],pmouth['p1y']), (255,0,0), 2)

	dxmouth1 = math.sqrt( ((pchin['p1x']-pmouth['p1x'])**2)+((pchin['p1y']-pmouth['p1y'])**2) )
	cv2.line(image, (pchin['p1x'],pchin['p1y']), (pmouth['p1x'],pmouth['p1y']), (0,255,0), 2)

	dxmouth2 = math.sqrt( ((pchin['p2x']-pmouth['p2x'])**2)+((pchin['p2y']-pmouth['p2y'])**2) )
	cv2.line(image, (pchin['p2x'],pchin['p2y']), (pmouth['p2x'],pmouth['p2y']), (0,255,0), 2)

	print("deye / dmouth " + str(deye / dmouth))
	print("d1chin / dxmouth1 " + str(d1chin / dxmouth1))
	print("d2chin / dxmouth2 " + str(d2chin / dxmouth2))
	# print(deye / dmouth)

	if ((deye / dmouth) > 1.85 ) & ((deye / dmouth) < 2.0) :
		cv2.putText(image, "Face #{Supun}", (100, 100),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	if ((d1chin / dxmouth1) > 1.75 ) & ((deye / dmouth) < 1.85) :
			cv2.putText(image, "Face #{Supun}", (100, 100),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	cv2.line(image, (5,10), (60,45), (0,255,0), 2)
	if roi.any():
		cv2.imshow("roid", roi)

	cv2.imshow("Output", image)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

vid.release();
cv2.destroyAllWindows();
