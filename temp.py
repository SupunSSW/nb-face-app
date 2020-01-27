import cv2 as cv

face_cascade = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

vid = cv.VideoCapture(0)

while True:
    ret, img = vid.read();
    
    faces = face_cascade.detectMultiScale(img, 1.5, 3)
    
    for x, y, w, h in faces:
        cv.rectangle(img, (x,y), (x + w, y + h), (0,255,0), 2)
    
    cv.imshow('sss', img)
    if cv.waitKey(1) & 0xff == ord('q'):
        break
    