import numpy as np
from matplotlib import pyplot as plt

def AniLoG(sigmax, sigmay, theta, indice):

  nx = range(-1*indice, indice + 1)
  ny = range(-1*indice, indice + 1)

  cost    = np.cos(theta)
  sint    = np.sin(theta)
  sin2t   = 2.0 * np.sin(theta) * np.cos(theta) 
  sq_cost = np.cos(theta) * np.cos(theta) 
  sq_sint = np.sin(theta) * np.sin(theta) 
  varx = sigmax * sigmax
  vary = sigmay * sigmay

  a = sq_cost/(2.0 * varx) + sq_sint/(2.0 * vary)
  b = sin2t/(4.0 * vary)   - sin2t/(4.0 * varx)
  c = sq_sint/(2.0 * varx) + sq_cost/(2.0 * vary)

  kernel = np.array( [[ 0.0 for j in range(len(nx)) ] for i in range(len(ny))] )
  for i,x in enumerate(nx):
    for j,y in enumerate(ny):
      expterm = np.exp(-(a*x*x + 2.0*b*x*y + c*y*y))
      gaussian = sigmax * sigmay * expterm
      SecDerx = ( 2.0*a*x + 2.0*b*y )*( 2.0*a*x + 2.0*b*y ) - 2.0*a
      SecDery = ( 2.0*b*x + 2.0*c*y )*( 2.0*b*x + 2.0*c*y ) - 2.0*c
      kernel[i,j] = SecDerx * gaussian + SecDery * gaussian 

  return kernel/kernel.sum()

###########################################
##         generate kernal               ##
###########################################
# kernal parameter
indice = 15
maxsig = 10.0
minsig = 6.0
step   = 1.0
tnum   = 6 

mmn = (maxsig - minsig)/step + 1
class kernel:
    def __init__(self):
      self.data = []

theta   = [ (i*np.pi)/tnum for i in range(tnum) ]
sigmaxy = [[x,y] for x in np.arange(minsig,maxsig + step,step) 
                 for y in np.arange(minsig,x      + step,step)]
kernels = np.array([ [kernel for j in range(len(sigmaxy))] for i in range(len(theta)) ])

for i,t in enumerate(theta):
  for j,xy in enumerate(sigmaxy):
    sigmax = xy[0]
    sigmay = xy[1]
    kernels[i,j] = AniLoG(sigmax,sigmay,t,indice)

#fig, ax = plt.subplots(len(theta),len(sigmaxy) )
#for i,row in enumerate(kernels):
#  for j,kernel in enumerate(row):
#    ax[i,j].imshow(-kernel)
#    ax[i,j].axis('off')
#plt.savefig("glog.png")

kernels = kernels.flatten()

###########################################
##         convolute image               ##
###########################################
import cv2
from scipy import signal 
from skimage import color, filters
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN

img = cv2.imread('sample1.png')

# enhance contrast
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * 255/ cdf[-1]
img_enh = cdf_normalized[img]

# convert the image to grayscale
img_gray = color.rgb2grey(img_enh).astype(float)/255.0

# LoG filter
log = [ -signal.convolve2d(img_gray,ikernel,mode='same') for ikernel in kernels ]

image_cube = np.stack(log)
plm = peak_local_max(image_cube, threshold_rel=0.2, exclude_border=False)

for i in range(len(log)):
  inter = plm[ plm[:,0] == i  ]
  plt.plot(inter[:,2], inter[:,1], 'r+',markersize=3)
  plt.imshow(img)
  plt.axis('off')
  plt.savefig("inter" +str(i)+".png",dpi=300)

# generate rough area to filter the noise center
img2      = np.array(img)
img_blur2 = cv2.medianBlur(img2,11)
img_gray2 = cv2.cvtColor(img_blur2,cv2.COLOR_BGR2GRAY)
ret2, th2 = cv2.threshold(img_gray2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

for i, point in enumerate(plm):
  xi = point[2]
  yi = point[1]
  if th2[yi, xi] != 255: plm[i,0] = -1 
plm_filter = plm[ plm[:,0] == -1 ] 

np.save("point.npy",plm_filter)

clustering = DBSCAN(eps = 8, min_samples = 1).fit(plm_filter[:,[1,2]])
ccenter = clustering.labels_
core    = clustering.core_sample_indices_

plt.imshow(th2)
plt.axis('off')
plt.savefig("th.png",dpi=300)
plt.close()

plt.plot(plm[:,2],plm[:,1],'r+',markersize=3)
plt.imshow(img)
plt.axis('off')
plt.savefig('plm.png',dpi=300)
plt.close()

plt.plot(plm_filter[:,2],plm_filter[:,1],'r+',markersize=3)
plt.imshow(img)
plt.axis('off')
plt.savefig('plm_cluster.png',dpi=300)
plt.close()





