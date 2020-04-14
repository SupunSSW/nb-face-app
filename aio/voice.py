import numpy as np
import cv2
import speech_recognition as sr
from PIL import Image
import imagehash

r = sr.Recognizer()

with sr.Microphone(0) as source:
	r.adjust_for_ambient_noise(source)
	print("SAY Next or Previous")
	audio = r.listen(source)

				
try:
	a = r.recognize_google(audio)
	print("a")

except:
	pass


# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Write some Text

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

cv2.putText(img,a,
bottomLeftCornerOfText,
font,
fontScale,
fontColor,
lineType)


#Display the image
cv2.imshow("img",img)

#Save image
cv2.imwrite("out.jpg", img)




image = cv2.imread("saved.jpg",0)
height,width=image.shape[:2]
WP1= cv2.countNonZero(image)
print(WP1)

image = cv2.imread("out.jpg",0)
height,width=image.shape[:2]
WP2= cv2.countNonZero(image)
print(WP2)

X=WP1-WP2
print(X)

if X == 0:
	print("can go next")
elif X == -921:
	print("can go previous")
else :
	print("can't continuee")

cv2.waitKey(0)
cv2.destroyAllWindows()