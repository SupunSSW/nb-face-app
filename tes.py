import dlib
import cv2 as cv
import math
import numpy as np
import os
from imutils import face_utils
from imutils.face_utils import FaceAligner
# import speech_recognition as sr
# from PIL import Image
# import imagehash

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

        try:
            r = height / float(h)
        except:
            r = 1
        
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions

        try:
            r = width / float(w)
        except:
            r = 1

        
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


def get_roi(snapshot):
    img = cv.imread("snap"+str(snapshot)+".jpg", 0)
    img = resize(img, width=600)
    face_rect = detector(img, 1)

    # alignFace(img)
    
    
    count = 0
    w = 0
    
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
    
    
    # print(int(pmid['p1y'] - w))
    # print(int(pmid['p1y'] + w))
    # print(int(pmid['p1x'] - w))
    # print(int(pmid['p1x'] + w))


    # scale_percent = 150 # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)

    # resize imageg


    try:
        img = resize(img, 300, int(img.shape[0]/img.shape[1])*300)
    except:
        img = resize(img, 300, 300)


    


    cv.imwrite("roi"+str(snapshot)+".jpg", img)

    cv.imshow("roi", img)



def alignFace(frame):

    img = cv.imread("snap"+str(frame)+".jpg", 0)
    
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # load the input image, resize it, and convert it to grayscale
    # gray = resize(frame, width=800)

    # show the original input image and detect faces in the grayscale
    # image
    rects = detector(gray, 2)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        # (x, y, w, h) = rect_to_bb(rect)
        # faceOrig = resize(frame[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(img, gray, rect)

        # import uuid
        # f = str(uuid.uuid4())
        # cv.imwrite("foo/" + f + ".png", faceAligned)

        # display the output images
        cv.imshow("Aligned", faceAligned)
        # cv.waitKey(0)
        cv.imwrite("roi"+str(frame)+".jpg", faceAligned)


def distances(p1, p2):
    dist =  math.sqrt( ((p1['x']-p2['x'])**2)+((p1['y']-p2['y'])**2) )

    return round(dist, 3)


def getRatio(l1, l2 = 5.0, d = 1):
    # print(">>>>>>>>>>>>>>>>>")
    # print(l1/l2)
    return round(l1/l2, d)


def findArea(a, b, c):
    s = (a + b + c) / 2

    return (s*(s-a)*(s-b)*(s-c)) ** 0.5




def getAngle(p1,p2,p3,p4):
    x = ((p3['y'] - p1['y'])*(p1['x'] - p2['x'])*(p3['x'] - p4['x']) + p1['x']*(p1['y'] - p2['y'])*(p3['x'] - p4['x']) - p3['x'] * (p3['y'] - p4['y']) * (p1['x'] - p2['x'])) / ((p1['y'] - p2['y']) * (p3['x'] - p4['x']) - (p3['y'] - p4['y']) * (p1['x'] - p2['x']))
    y = p1['y'] * (p1['x'] - p2['x']) + (p1['y'] - p2['y']) * (x - p1['x'])

    pcenter = {'x' : x, 'y' : y}
    pmid = {'x': x, 'y': p1['y']}

    c = distances(pcenter, pmid)
    b = distances(p1, pmid)

    return (b /c)




def recognition_face(snapframe):
    global deyemouth
    global mid_angle
    global dplus

    global temp1
    global temp2
    global temp3
    global temp4
    global temp5
    global temp6
    global temp7
    global temp8
    global temp9
    global temp10
    global temp11
    global temp12
    global temp13
    global temp14
    global temp15
    global temp16
    global temp17
    global temp18
    global temp19
    global temp20
    global temp21
    global temp22
    global temp23
    global temp24
    global temp25
    global temp26
    global temp27
    global temp28

    alignFace(snapframe)

    # img = cv.imread("roi"+str(snapframe)+".jpg", 0)
    img = cv.imread("roi"+str(snapframe)+".jpg", 0)

    
    # if snapframe is 10:
    #     img = resize(img, width=600)
    #     face_rect = detector(img, 1)
    
    # img = resize(img, width=600)
    face_rect = detector(img, 1)
    
        
    for (i, rect) in enumerate(face_rect):
        shape = predictor(img, rect)
        shape = coords(shape)
        
	    # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # (x, y, w, h) = face_utils.rect_to_bb(rect)
	    # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        count = 0



        p = {'x': 0, 'y': 0}

        
        for (x, y) in shape:
            cv.circle(img, (x,y), 1, (0,0,255), -1)
            count += 1

            p[count] = {'x': x, 'y': y}

            # p[count]['x'] = x
            # p[count]['y'] = y

            # print(p[count])
            # print(count)



        
        cv.line(img, (p[37]['x'],p[37]['y']), (p[34]['x'],p[34]['y']), (0,255,0), 2)
        cv.line(img, (p[46]['x'],p[46]['y']), (p[34]['x'],p[34]['y']), (0,255,0), 2)

        cv.line(img, (p[40]['x'],p[40]['y']), (p[34]['x'],p[34]['y']), (200,100,0), 2)
        cv.line(img, (p[43]['x'],p[43]['y']), (p[34]['x'],p[34]['y']), (200,100,0), 2)

        cv.line(img, (p[37]['x'],p[37]['y']), (p[40]['x'],p[40]['y']), (50,100,200), 2)
        cv.line(img, (p[43]['x'],p[43]['y']), (p[46]['x'],p[46]['y']), (50,100,200), 2)

        cv.line(img, (p[40]['x'],p[40]['y']), (p[43]['x'],p[43]['y']), (250,100,200), 2)

        cv.line(img, (p[34]['x'],p[34]['y']), (p[49]['x'],p[49]['y']), (250,100,200), 2)
        cv.line(img, (p[34]['x'],p[34]['y']), (p[55]['x'],p[55]['y']), (250,100,200), 2)

        cv.line(img, (p[49]['x'],p[49]['y']), (p[55]['x'],p[55]['y']), (250,50,100), 2)

        cv.line(img, (p[37]['x'],p[37]['y']), (p[49]['x'],p[49]['y']), (50,50,250), 2)
        cv.line(img, (p[46]['x'],p[46]['y']), (p[55]['x'],p[55]['y']), (50,50,250), 2)


        cv.imwrite("face"+str(i)+".jpg", img)



        # chin

        # l1 = math.sqrt( ((p[1]['x']-p[17]['x'])**2)+((p[1]['y']-p[17]['y'])**2) )
        l1 = distances(p[1], p[17])
        l2 = distances(p[2], p[16])
        l3 = distances(p[3], p[15])
        l4 = distances(p[4], p[14])
        l5 = distances(p[5], p[13])
        l6 = distances(p[6], p[12])


        # eye

        l7 = distances(p[37], p[46])
        l77 = distances(p[40], p[43])

        # eye to mouth corner
        l8 = distances(p[37], p[49])
        l9 = distances(p[46], p[55])

        # vertical distance
        l10 = distances(p[28], p[9])

        # mouth distance
        l11 = distances(p[49], p[55])


        # /
        l12 = distances(p[49], p[6])

        # \
        l13 = distances(p[55], p[12])

        # A
        l14 = distances(p[28], p[32])
        l15 = distances(p[28], p[36])


        # ><
        l16 = distances(p[55], p[37])
        l17 = distances(p[49], p[46])

        # _
        l18 = distances(p[18], p[27])
        l19 = distances(p[18], p[23])
        l20 = distances(p[27], p[22])


        a1 = getAngle(p[37],p[36],p[46],p[32])

        # l21 = (((l18/l19) + (l18/l20))/2)/l10
        # l21 = getRatio(((l18/l19) + (l18/l20))/2,l10)
        l21 = (((l18/l19) + (l18/l20))/2) / l10

        # l22 = getRatio(l2,(((l18/l19) + (l18/l20))/2))
        l22 = l2 / (((l18/l19) + (l18/l20))/2)

        # l23 = getRatio(l2,l7)
        l23 = l2 / l7

        l24 = l7 / l77

        l25 = l23 / l24

        l26 = l2 / ((l12 + l13) / 2)

        l27 = l10 / l7

        l28 = l10 / l11

        l29 = l10 / ((l8 + l9) / 2.0)

        l30 = l4 / l11

        # l22 = l2 / (((l18/l19) + (l18/l20))/2)

        # l23 = l2 / l7

        # l24 = l7 / l77

        # l25 = l23 / l24

        # l26 = l2 / ((l12 + l13) / 2)

        # print(l21)
        # print(l1)
        # print("l22 ")
        # print(l22)
        # print(l23)
        # print(2 * )


        
        # d1chin = math.sqrt( ((pchin['p1x']-pmid['p1x'])**2)+((pchin['p1y']-pmid['p1y'])**2) )
        # cv.line(img, (pchin['p1x'],pchin['p1y']), (pmid['p1x'],pmid['p1y']), (0,255,0), 2)

        print("deye / dmouth ")
        
        # print("deye / dmoupntopth " + str(deye / dmouth))
        # deyemouth += (deye / dmouth)
        # return (deyemouth/5.0)
        # print("d1chin / dxmouth1 " + str(d1chin / dxmouth1))
        # print("d2chin / dxmouth2 " + str(d2chin / dxmouth2))
        
        # deyemouth += (crossX / crossY)
        deyemouth += l1
        mid_angle += a1

        # - | -
        dplus += l21

        # temp1 += a1
        # temp2 += l21
        # temp3 += l22
        # temp4 += l25
        # temp5 += l26
        # temp6 += l27

        

        # area set 1
        a1 = findArea(distances(p[37], p[34]), distances(p[34], p[49]), distances(p[49], p[37]))
        a2 = findArea(distances(p[37], p[34]), distances(p[34], p[40]), distances(p[37], p[40]))
        a3 = findArea(distances(p[40], p[43]), distances(p[34], p[40]), distances(p[34], p[43]))
        a4 = findArea(distances(p[43], p[34]), distances(p[43], p[46]), distances(p[34], p[46]))
        # a5 = findArea(distances(p[34], p[46]), distances(p[46], p[55]), distances(p[55], p[34]))
        a6 = findArea(distances(p[49], p[34]), distances(p[34], p[55]), distances(p[55], p[49]))

        # area set 2
        a11 = findArea(distances(p[3], p[28]), distances(p[28], p[34]), distances(p[34], p[3]))
        a21 = findArea(distances(p[28], p[15]), distances(p[15], p[34]), distances(p[34], p[28]))
        a31 = findArea(distances(p[3], p[34]), distances(p[34], p[6]), distances(p[6], p[3]))
        a41 = findArea(distances(p[34], p[15]), distances(p[15], p[12]), distances(p[12], p[34]))
        a51 = findArea(distances(p[6], p[28]), distances(p[28], p[34]), distances(p[34], p[6]))
        a61 = findArea(distances(p[34], p[28]), distances(p[28], p[12]), distances(p[12], p[34]))
        a71 = findArea(distances(p[6], p[34]), distances(p[34], p[9]), distances(p[9], p[6]))
        a81 = findArea(distances(p[9], p[34]), distances(p[34], p[12]), distances(p[12], p[9]))






        dist1 =  math.sqrt((p[37]['x']-p[34]['x'])**2)
        dist2 =  math.sqrt((p[34]['y']-p[37]['y'])**2)

        dist3 =  math.sqrt((p[34]['x']-p[46]['x'])**2)
        dist4 =  math.sqrt((p[34]['y']-p[46]['y'])**2)

        ag1 = math.atan2(dist1, dist2)
        ag2 = math.atan2(dist3, dist4)


        ag = ag1 + ag2


        # angle ratio set 1 ##################
        # r1 = a1 /a2
        # r2 = a1 /a3
        # r3 = a1 /a4
        # r4 = a1 /a5
        r5 = a1 /a6
        # r6 = a2 /a3
        # r7 = a2 /a4
        # r8 = a2 /a5
        r9 = a2 /a6
        # r10 = a3 /a4
        # r11 = a3 /a5
        r12 = a3 /a6
        # r13 = a4 /a5
        r14 = a4 /a6
        # r15 = a5 /a6


        # angle ratio set 2 ##################
        r001 = a11 / a21

        #
        r002 = a11 / a31


        r003 = a11 / a41
        r004 = a11 / a51
        r005 = a11 / a61
        r006 = a11 / a71
        r007 = a11 / a81
        r008 = a21 / a31
        r009 = a21 / a41
        r010 = a21 / a51

        #
        r011 = a21 / a61


        r012 = a21 / a71
        r013 = a21 / a81
        r014 = a31 / a41

        #
        r015 = a31 / a51


        r016 = a31 / a61

        #
        r017 = a31 / a71


        r018 = a31 / a81
        r019 = a41 / a51
        r020 = a41 / a61
        r021 = a41 / a71

        #
        r022 = a41 / a81


        r023 = a51 / a61
        r024 = a51 / a71
        r025 = a51 / a81
        r026 = a61 / a71

        #
        r027 = a61 / a81


        r028 = a71 / a81

        temp1 += r001
        temp2 += r002
        temp3 += r003

        temp4 += r004

        temp5 += r005

        temp6 += r006
        temp7 += r007
        temp8 += r008

        temp9 += r009

        temp10 += r010
        temp11 += r011

        temp12 += r012

        temp13 += r013

        temp14 += r014

        temp15 += r015

        temp16 += r016
        temp17 += r017
        temp18 += r018
        temp19 += r019
        temp20 += r020
        temp21 += r021
        temp22 += r022
        temp23 += r023
        temp24 += r024
        temp25 += r025
        temp26 += r026
        temp27 += r027
        temp28 += r028

        # return getRatio(temp1), getRatio(temp2), getRatio(temp3), getRatio(temp4), getRatio(temp5), getRatio(temp6), getRatio(temp7),getRatio(temp8),getRatio(temp9),getRatio(temp10),getRatio(temp11),getRatio(temp12),getRatio(temp13),getRatio(temp14),getRatio(temp15),getRatio(temp16,5,1)
        # return getRatio(temp1), getRatio(temp2), getRatio(temp3), getRatio(temp4), getRatio(temp5), getRatio(temp6), getRatio(temp7), getRatio(temp8), getRatio(temp9), getRatio(temp10), getRatio(temp11), getRatio(temp12), getRatio(temp13), getRatio(temp14), getRatio(temp15), getRatio(temp16), getRatio(temp17), getRatio(temp18), getRatio(temp19), getRatio(temp20), getRatio(temp21), getRatio(temp22), getRatio(temp23), getRatio(temp24), getRatio(temp25), getRatio(temp26), getRatio(temp27), getRatio(temp28)
        return getRatio(temp2), getRatio(temp11), getRatio(temp15), getRatio(temp17), getRatio(temp22), getRatio(temp27)
        # return getRatio(temp2,5)

framecount = 0
nullframes = 0
deyemouth = 0.0
mid_angle = 0
dplus, temp = 0, 0
x,y,w,h = 0,0,0,0
temp1 = 0
temp2 = 0
temp3 = 0
temp4 = 0
temp5 = 0
temp6 = 0
temp7 = 0
temp8 = 0
temp9 = 0
temp10 = 0
temp11 = 0
temp12 = 0
temp13 = 0
temp14 = 0
temp15 = 0
temp16 = 0
temp17 = 0
temp18 = 0
temp19 = 0
temp20 = 0
temp21 = 0
temp22 = 0
temp23 = 0
temp24 = 0
temp25 = 0
temp26 = 0
temp27 = 0
temp28 = 0
truFactor = 0



vid = cv.VideoCapture(0)

while True:
    ret, frame = vid.read()
    
    frame = cv.GaussianBlur(frame,(5,5),0)
    frame = resize(frame, width=600)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    

    if rects:
        print("===============================")
        
        nullframes = 0
        temp51, temp91, temp121, temp141, temp01, temp02, temp03,temp04 = 0,0,0,0,0,0,0,0
        framedone = False
        
        
        
        if framecount is not 0 and framecount in range(31):
            framecount += 1

            if framecount is 10:
                cv.imwrite("snap"+str(framecount)+".jpg", frame)
                # get_roi(framecount)
                recognition_face(framecount)
            elif framecount is 15:
                cv.imwrite("snap"+str(framecount)+".jpg", frame)
                # get_roi(framecount)
                recognition_face(framecount)
            elif framecount is 20:
                cv.imwrite("snap"+str(framecount)+".jpg", frame)
                # get_roi(framecount)
                recognition_face(framecount)
            elif framecount is 25:
                cv.imwrite("snap"+str(framecount)+".jpg", frame)
                # get_roi(framecount)
                recognition_face(framecount)
            elif framecount is 30:
                cv.imwrite("snap"+str(framecount)+".jpg", frame)
                # get_roi(framecount)
                print("(0.8, 1.1, 0.8, 1.4, 1.1, 1.1, 0.9, 1.4, 1.0, 1.8, 1.4, 1.4, 1.1, 0.7, 1.3, 1.0, 1.0, 0.8, 1.7, 1.4, 1.3, 1.1, 0.8, 0.8, 0.6, 1.0, 0.8, 0.8)")
                print("(0.8, 1.1, 0.8, 1.4, 1.1, 1.1, 0.9, 1.4, 1.0, 1.8, 1.4, 1.4, 1.1, 0.7, 1.3, 1.0, 1.0, 0.8, 1.7, 1.4, 1.3, 1.1, 0.8, 0.8, 0.6, 1.0, 0.8, 0.8)")
                print("(0.8, 1.1, 0.8, 1.4, 1.1, 1.1, 0.9, 1.4, 1.0, 1.8, 1.4, 1.4, 1.1, 0.7, 1.3, 1.0, 1.0, 0.8, 1.7, 1.4, 1.3, 1.1, 0.8, 0.8, 0.6, 1.0, 0.8, 0.8)")
                print(recognition_face(framecount))
                # framedone = True
                # temp51, temp91, temp121, temp141, temp01, temp02, temp03, temp04 = recognition_face(framecount)
                
            
        elif framecount < 31:
            framecount += 1
        # elif framecount >= 31:
        
        
        
        #


        # output1 = '' + str(temp51) + str(temp91) + str(temp121) + str(temp141) + str(temp01) + str(temp02) + str(temp03) + str(temp04)


        # if(output1 == "2.01.01.01.02.02.02.01.7"):
        #     print("Supun")
        # elif(output1 == "3.02.03.02.02.02.02.01.5"):
        #     print("Ranna")
        # elif(output1 == "2.01.02.01.02.02.02.01.5"):
        #     print("Moriya")
        # else:
        #     if framedone:
        #         print(temp51, temp91, temp121, temp141, temp01, temp02, temp03, temp04)



        #
        
        
            
            
    else:
        if framecount >= 31:
            if nullframes > 40:
                # if user left
                framecount, nullframes, deyemouth = 0, 0, 0
                mid_angle, dplus, temp = 0, 0, 0
                temp1 = 0
                temp2 = 0
                temp3 = 0
                temp4 = 0
                temp5 = 0
                temp6 = 0
                temp7 = 0
                temp8 = 0
                temp9 = 0
                temp10 = 0
                temp11 = 0
                temp12 = 0
                temp13 = 0
                temp14 = 0
                temp15 = 0
                temp16 = 0
                temp17 = 0
                temp18 = 0
                temp19 = 0
                temp20 = 0
                temp21 = 0
                temp22 = 0
                temp23 = 0
                temp24 = 0
                temp25 = 0
                temp26 = 0
                temp27 = 0
                temp28 = 0
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



vid.release()
cv.destroyAllWindows()