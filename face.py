import dlib
import cv2 as cv
import math
import numpy as np
import os
from imutils import face_utils
import speech_recognition as sr
from PIL import Image
import imagehash

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def coords(shape, dtype="int"):
    	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords



def rect_to_bb(rect):
    	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)



def rect_to_bb(rect):
    	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)



def get_roi(snapshot):
    img = cv.imread("snap"+str(snapshot)+".jpg", 0)
    img = resize(img, width=600)
    face_rect = detector(img, 1)
    
    
    count = 0
    w,h = 0,0
    
    pmid = {'p1x':0, 'p1y':0}
    pchin = {'p1x':0, 'p1y':0}
    
    
    for (i, rect) in enumerate(face_rect):
        shape = predictor(img, rect)
        shape = coords(shape)
        
        for(x,y) in shape:
            count += 1
            if count is 9:
                pchin['p1x'] = x
                pchin['p1y'] = y
            elif count is 31:
                pmid['p1x'] = x
                pmid['p1y'] = y
                
    d = math.sqrt( ((pchin['p1x']-pmid['p1x'])**2)+((pchin['p1y']-pmid['p1y'])**2) )
    w = d * 1.5
    
    img = img[int(math.sqrt((pmid['p1y'] - w)**2)): int(math.sqrt((pmid['p1y'] + w)**2)), int(math.sqrt(( pmid['p1x'] - w) **2 )): int(math.sqrt((pmid['p1x'] + w)**2))]
    
    
    print(int(pmid['p1y'] - w))
    print(int(pmid['p1y'] + w))
    print(int(pmid['p1x'] - w))
    print(int(pmid['p1x'] + w))
    cv.imwrite("roi"+str(snapshot)+".jpg", img)
    cv.imshow("roi", img)



def recognition_face(snapframe):
    global deyemouth
    img = cv.imread("roi"+str(snapframe)+".jpg", 0)
    
    # if snapframe is 10:
    #     img = resize(img, width=600)
    #     face_rect = detector(img, 1)
    
    img = resize(img, width=600)
    face_rect = detector(img, 1)
    
        
    for (i, rect) in enumerate(face_rect):
        shape = predictor(img, rect)
        shape = coords(shape)
        
	    # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # (x, y, w, h) = face_utils.rect_to_bb(rect)
	    # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        count = 0
        pchin = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
        pleye = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
        preye = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
        pmid = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
        pmouth = {'p1x':0, 'p1y':0, 'p2x':0, 'p2y':0}
        
        pntop = {'p1x':0, 'p1y':0}
        pltop = {'p1x':0, 'p1y':0}
        
        for (x, y) in shape:
            cv.circle(img, (x,y), 3, (0,0,255), -1)
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
            elif count is 37: # left eye left point
                pleye['p1x'] = x
                pleye['p1y'] = y
            elif count is 46: # right eye right point
                preye['p1x'] = x
                preye['p1y'] = y
            elif count is 40:
                pleye['p2x'] = x
                pleye['p2y'] = y
            elif count is 43:
                preye['p2x'] = x
                preye['p2y'] = y
            elif count is 28: # nose top point
                pntop['p1x'] = x
                pntop['p1y'] = y
            elif count is 52: # upper lip highest point
                pltop['p1x'] = x
                pltop['p1y'] = y
                

        cv.imwrite("face"+str(i)+".jpg", img)
        
        # d1chin = math.sqrt( ((pchin['p1x']-pmid['p1x'])**2)+((pchin['p1y']-pmid['p1y'])**2) )
        # cv.line(img, (pchin['p1x'],pchin['p1y']), (pmid['p1x'],pmid['p1y']), (0,255,0), 2)

        # d2chin = math.sqrt( ((pchin['p2x']-pmid['p1x'])**2)+((pchin['p2y']-pmid['p1y'])**2) )
        # cv.line(img, (pchin['p2x'],pchin['p2y']), (pmid['p1x'],pmid['p1y']), (0,255,0), 2)

        # deye = math.sqrt( ((peye['p1x']-peye['p2x'])**2)+((peye['p1y']-peye['p2y'])**2) )
        # cv.line(img, (peye['p2x'],peye['p2y']), (peye['p1x'],peye['p1y']), (255,0, 0), 2)

        # dmouth = math.sqrt( ((pmouth['p1x']-pmouth['p2x'])**2)+((pmouth['p1y']-pmouth['p2y'])**2) )
        # cv.line(img, (pmouth['p2x'],pmouth['p2y']), (pmouth['p1x'],pmouth['p1y']), (255,0,0), 2)

        # dxmouth1 = math.sqrt( ((pchin['p1x']-pmouth['p1x'])**2)+((pchin['p1y']-pmouth['p1y'])**2) )
        # cv.line(img, (pchin['p1x'],pchin['p1y']), (pmouth['p1x'],pmouth['p1y']), (0,255,0), 2)

        # dxmouth2 = math.sqrt( ((pchin['p2x']-pmouth['p2x'])**2)+((pchin['p2y']-pmouth['p2y'])**2) )
        # cv.line(img, (pchin['p2x'],pchin['p2y']), (pmouth['p2x'],pmouth['p2y']), (0,255,0), 2)
        
        crossX = math.sqrt( ((pleye['p1x']-preye['p1x'])**2)+((pleye['p1y']-preye['p1y'])**2) )
        mideye = math.sqrt( ((pleye['p2x']-preye['p2x'])**2)+((pleye['p2y']-preye['p2y'])**2) )
        crossY = math.sqrt( ((pntop['p1x']-pltop['p1x'])**2)+((pntop['p1y']-pltop['p1y'])**2) )

        print("deye / dmouth ")
        
        # print("deye / dmoupntopth " + str(deye / dmouth))
        # deyemouth += (deye / dmouth)
        # return (deyemouth/5.0)
        # print("d1chin / dxmouth1 " + str(d1chin / dxmouth1))
        # print("d2chin / dxmouth2 " + str(d2chin / dxmouth2))
        
        deyemouth += (crossX / crossY)
        return (deyemouth/5.0)

framecount = 0
nullframes = 0
deyemouth = 0.0
x,y,w,h = 0,0,0,0

vid = cv.VideoCapture(0)

while True:
    ret, frame = vid.read()
    
    frame = cv.GaussianBlur(frame,(5,5),0)
    frame = resize(frame, width=600)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    
    # print(rects)
    # for rr in rects:
    #     print rr[0]
    
    # (x,y,w,h) = rect_to_bb(rects)
    
    
    # for (i,rect) in enumerate(rects):
    #     # (x,y,w,h) = rect_to_bb(rect)
    #     # frame = frame[y:y+h, x:x+w]
    #     # pass
        

    if rects:
        print("===============================")
        
        nullframes = 0
        
        # try:
        #     for
        # except expression as identifier:
        #     pass
        
        
        if framecount is not 0 and framecount in range(31):
            framecount += 1

            if framecount is 10:
                cv.imwrite("snap"+str(framecount)+".jpg", frame)
                get_roi(framecount)
                recognition_face(framecount)
            elif framecount is 15:
                cv.imwrite("snap"+str(framecount)+".jpg", frame)
                get_roi(framecount)
                recognition_face(framecount)
            elif framecount is 20:
                cv.imwrite("snap"+str(framecount)+".jpg", frame)
                get_roi(framecount)
                recognition_face(framecount)
            elif framecount is 25:
                cv.imwrite("snap"+str(framecount)+".jpg", frame)
                get_roi(framecount)
                recognition_face(framecount)
            elif framecount is 30:
                cv.imwrite("snap"+str(framecount)+".jpg", frame)
                get_roi(framecount)
                print(recognition_face(framecount))
                
            
        elif framecount < 31:
            framecount += 1
        # elif framecount >= 31:
        
        
        
        #
        #
        
        
            
            
    else:
        if framecount >= 31:
            if nullframes > 40:
                # if user left
                framecount, nullframes, deyemouth = 0, 0, 0
                try:
                    os.remove("snap10.jpg")
                    os.remove("snap15.jpg")
                    os.remove("snap20.jpg")
                    os.remove("snap25.jpg")
                    os.remove("snap30.jpg")
                except:
                    pass
            else:
                nullframes += 1
                
            print (framecount)
            print (nullframes)
            
    
    # cv.putText(frame, "Face #{}", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # cv.namedWindow("f", cv.WINDOW_AUTOSIZE)
    
    cv.imshow("f", frame)
    
    if cv.waitKey(1) & 0xff == ord('q'):
        break

try:
    os.remove("snap10.jpg")
    os.remove("snap15.jpg")
    os.remove("snap20.jpg")
    os.remove("snap25.jpg")
    os.remove("snap30.jpg")
except:
    pass



vid.release();
cv.destroyAllWindows();