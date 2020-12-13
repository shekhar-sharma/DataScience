import skimage.morphology  as  mp
import numpy as np
import cv2 as cv
from convert_binary import rgbtobin
dil=[]
selem=mp.disk(5,np.uint8)
#img=cv.imread('binary.png')
#ret,img = cv.threshold(img,127,1,cv.THRESH_BINARY)
#rgbtobin(img)
image=cv.imread('binary.png',cv.IMREAD_GRAYSCALE)
	


img = cv.threshold(image,128, 255, cv.THRESH_BINARY)[1]

cv.imwrite('binary_con.png',img)
#final=cv.imread('binary_con.png')
bin_img=cv.imread('binary_con.png')

ans=mp.binary_dilation(bin_img,selem)
print(ans)
print(bin_img)
# print(dil)
#cv.imshow('img',img)
#cv.waitKey(0)
#print(img)