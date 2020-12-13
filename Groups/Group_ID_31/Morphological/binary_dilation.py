import skimage.morphology  as  mp
import numpy as np
import cv2 as cv
from  disk import disk_r
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#selem=mp.disk(3,np.uint8)
#img=cv.imread('binary.png')
#ret,img = cv.threshold(img,127,1,cv.THRESH_BINARY)
#rgbtobin(img)
image=cv.imread('binary.png',cv.IMREAD_GRAYSCALE)

img = cv.threshold(image,128, 255, cv.THRESH_BINARY)[1]
# cv.imshow('image',img)
# cv.waitKey(0)
#cv.imwrite('binary_con.png',img)
#final=cv.imread('binary_con.png')
#bin_img=cv.imread('binary_con.png')
print(img)
selem=disk_r(2)
ans=mp.binary_dilation(img,selem).astype(int)
print(ans)
#print(bin_img)
print(ans.shape)
plt.imsave('dilated.png', np.array(ans).reshape(194,259), cmap=cm.gray)
# ans_i=cv.imread('filename.png',filename.png)
# cv.imshow('img',ans_i)
# #print(img)

# cv.waitKey(0)
image = np.eye(194)[:,:259]
binary = image > 0
plt.imshow(binary)
plt.show()


