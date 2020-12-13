import cv2 as cv

def rgbtobin(img):

	#image=cv.imread('binary.png',cv.IMREAD_GRAYSCALE)
	image=cv.imread(img,cv.IMREAD_GRAYSCALE)
	


	img = cv.threshold(image,128, 255, cv.THRESH_BINARY)[1]

	cv.imwrite('binary_con.png',img)
	final=cv.imread('binary_con.png')
	print(final)
	#return final