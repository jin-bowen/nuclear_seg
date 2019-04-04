import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color, filters
from sklearn.cluster import DBSCAN
from skimage import measure
from skimage.segmentation import (inverse_gaussian_gradient, 
                                  morphological_chan_vese,
                                  circle_level_set)

img = cv2.imread('sample1.png')
# convert the image to grayscale
img_gray = color.rgb2grey(img).astype(float)/255.0

plm_filter = np.load("point.npy")
sig_max = 10

#plm_filter = np.delete(plm_filter, 0, axis = 1)
print plm_filter

i = 1
inter = plm_filter[ plm_filter[:,0] == i  ]
plt.plot(inter[:,2], inter[:,1], 'r+',markersize=3)
plt.imshow(img)
plt.axis('off')
plt.savefig("inter1.png",dpi=300)




def getnucmean(plm_filter):
    # numpy as np
    # sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps = sig_max, min_samples = 1).fit(plm_filter)
    label = clustering.labels_
    
    uni,c = np.unique(label, return_counts=True)
    dup   = uni[c >  1]
    nodup = uni[c == 1]
    
    nucleus = []
    for n in dup:
      pn      = [ point for i, point in enumerate(plm_filter) if label[i] == n]
      mean    = np.mean(pn, axis = 0)
      nucleus.append(mean)
    
    for i, point in enumerate(plm_filter):
      if label[i] in nodup: nucleus.append(point)

    return np.array(nucleus)

nucleus = getnucmean(plm_filter) 

init = np.zeros_like(img_gray)
for point in nucleus:
  mat = circle_level_set(img_gray.shape, point, 10)
  init = init + mat

ls = morphological_chan_vese(img_gray, iterations=100, 
                             init_level_set=init,
                             smoothing=1)
plt.contour(init, [0.01],colors='blue',linewidths = 0.5, linestyles = 'dashed')
plt.plot(nucleus[:,1], nucleus[:,0], 'r+',markersize=3)
plt.imshow(img)
plt.axis('off')
plt.savefig("ct00.png",dpi=300)


