# import db

# # da = db.checkUsr("0")

# # if da:
# #     print(da)

# # else:
# #     pass


# da = db.getNotices(34,'MGT')

# if da:
#     print(da)

# else:
#     pass



import imageio
import matplotlib.pyplot as plt


from mlxtend.image import extract_face_landmarks

img = imageio.imread('roi30.jpg')
landmarks = extract_face_landmarks(img)
print(landmarks.shape)
print('\n\nFirst 10 landmarks:\n', landmarks[:10])

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 3, 1)
ax.imshow(img)
ax = fig.add_subplot(1, 3, 2)
ax.scatter(landmarks[:, 0], -landmarks[:, 1], alpha=0.8)
ax = fig.add_subplot(1, 3, 3)
img2 = img.copy()

for p in landmarks:
    img2[p[1]-3:p[1]+3, p[0]-3:p[0]+3, :] = (255, 255, 255)
    # note that the values -3 and +3 will make the landmarks
    # overlayed on the image 6 pixels wide; depending on the
    # resolution of the face image, you may want to change
    # this value

ax.imshow(img2)
plt.show()