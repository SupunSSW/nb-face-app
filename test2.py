from imutils import face_utils
import imutils
import dlib
import cv2 as cv
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


vid = cv.VideoCapture(0)

while True:
    ret, frame = vid.read()
    
    frame = imutils.resize(frame, width=600)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    if rects:
        print "==============================="
    else:
        pass
    
    for (i, rect) in enumerate(rects):
        shape = predictor(frame, rect)
        shape = coords(shape)
        
        count = 0
        pchin = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
        peye = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
        pmid = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
        pmouth = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
        
        for (x, y) in shape:
            cv.circle(frame, (x,y), 3, (0,0,255), -1)
            count += 1
            if count is 4:
                pchin['p1x'] = x
                pchin['p1y'] = y
            elif count is 14:
                pchin['p2x'] = x
                pchin['p2y'] = y
            elif count is 31:
                pmid['p1x'] = x
                pmid['p1y'] = y
            elif count is 37:
                peye['p1x'] = x
                peye['p1y'] = y
            elif count is 46:
                peye['p2x'] = x
                peye['p2y'] = y
            elif count is 49:
                pmouth['p1x'] = x
                pmouth['p1y'] = y
            elif count is 55:
                pmouth['p2x'] = x
                pmouth['p2y'] = y
                
        d1chin = math.sqrt( ((pchin['p1x']-pmid['p1x'])**2)+((pchin['p1y']-pmid['p1y'])**2) )
        cv.line(frame, (pchin['p1x'],pchin['p1y']), (pmid['p1x'],pmid['p1y']), (0,255,0), 2)

        d2chin = math.sqrt( ((pchin['p2x']-pmid['p1x'])**2)+((pchin['p2y']-pmid['p1y'])**2) )
        cv.line(frame, (pchin['p2x'],pchin['p2y']), (pmid['p1x'],pmid['p1y']), (0,255,0), 2)

        deye = math.sqrt( ((peye['p1x']-peye['p2x'])**2)+((peye['p1y']-peye['p2y'])**2) )
        cv.line(frame, (peye['p2x'],peye['p2y']), (peye['p1x'],peye['p1y']), (255,0, 0), 2)

        dmouth = math.sqrt( ((pmouth['p1x']-pmouth['p2x'])**2)+((pmouth['p1y']-pmouth['p2y'])**2) )
        cv.line(frame, (pmouth['p2x'],pmouth['p2y']), (pmouth['p1x'],pmouth['p1y']), (255,0,0), 2)

        dxmouth1 = math.sqrt( ((pchin['p1x']-pmouth['p1x'])**2)+((pchin['p1y']-pmouth['p1y'])**2) )
        cv.line(frame, (pchin['p1x'],pchin['p1y']), (pmouth['p1x'],pmouth['p1y']), (0,255,0), 2)

        dxmouth2 = math.sqrt( ((pchin['p2x']-pmouth['p2x'])**2)+((pchin['p2y']-pmouth['p2y'])**2) )
        cv.line(frame, (pchin['p2x'],pchin['p2y']), (pmouth['p2x'],pmouth['p2y']), (0,255,0), 2)

        print("deye / dmouth " + str(deye / dmouth))
        print("d1chin / dxmouth1 " + str(d1chin / dxmouth1))
        print("d2chin / dxmouth2 " + str(d2chin / dxmouth2))
                    
    
    
    cv.imshow("ss", frame)
    
    if cv.waitKey(1) & 0xff == ord('q'):
        break

vid.release();
cv.destroyAllWindows();