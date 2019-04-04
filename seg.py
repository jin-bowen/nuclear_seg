import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import feature

img = cv2.imread('sample1.png')

# enhance contrast
#hist,bins = np.histogram(img.flatten(),256,[0,256])
#cdf = hist.cumsum()
#cdf_normalized = cdf * 255/ cdf[-1]
#img_enh = cdf_normalized[img] 

img_enh = np.array(img, dtype=np.uint8)
# blur the image
img_blur = cv2.medianBlur(img_enh,7)

# convert the image to grayscale
img_gray = cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# remove noise 
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(th2,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations = 3)
## Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
## Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
## Add one to all labels so that sure background is not 0, but 1
markers = markers+1
## Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
## set the border to 0
markers[ 0,:] = 0
markers[-1,:] = 0
markers[:,-1] = 0
markers[:, 0] = 0

## transfer the marker to edge coordinate
watershed_contour = []
for n, line in enumerate(markers):
  for m, val in enumerate(line):
    if val == -1: watershed_contour.append([n,m])
init = np.array(watershed_contour)

#plt.plot(init[:,1],init[:,0],'-r')
#plt.imshow(img)
#plt.show()

fiure,ax = plt.subplots(3,sharex=True)
ax[0].imshow(img_gray)
ax[1].imshow(opening)
ax[2].imshow(dist_transform)
plt.show()

# nuclear center detection with LoG 
img_inv = np.invert(img_gray)
center = feature.blob_log(img_inv, max_sigma=30, num_sigma=10, threshold=.1)

coord = np.array(center[:,[0,1]])
coord = coord.astype(int)
x = []
y = []

for i in range(len(coord)):
  if img_inv[coord[i,0],coord[i,1]] > 200: 
    x.append(coord[i,0])
    y.append(coord[i,1])


# test
#img[markers == -1] = [255,0,0]
fiure,ax = plt.subplots(2,sharex=True)
plt.plot(y[i],x[i],'b+')
for i in range(len(x)):
#    print img_inv[x[i],y[i]]
    plt.plot(y[i],x[i],'r+')

plt.plot(snake[:, 0], snake[:, 1], '-b')
ax[0].imshow(img_inv)
ax[1].imshow(img)
plt.show()

