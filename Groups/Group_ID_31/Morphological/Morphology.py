import numpy as np
from scipy import ndimage as ndi
import cv2 as cv 


class Morphology:

	def binary(self,image):
		# image=cv.imread(image,cv.IMREAD_GRAYSCALE)
		image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		# image=cv.threshold(image,128, 255, cv.THRESH_BINARY)[1]
		image= cv.threshold(image,128, 255, cv.THRESH_BINARY)[1]
		return image

	def square(self,dim, dtype=np.uint8):
		return np.ones((dim, dim), dtype=dtype)

	def rectangle(self,nrows, ncols, dtype=np.uint8):
		return np.ones((nrows, ncols), dtype=dtype)

	def dilation(self,img,shape,*args):
		if(str(shape)=='square'):
			struct=self.square(args[0])
		elif(str(shape)=='rectangle'):
			struct=self.rectangle(args[0],args[1])
		else:
			print('Enter a valid shape and dimension')
		# image=cv.threshold(image,128, 255, cv.THRESH_BINARY)[1]

		img= self.binary(img)
		#printing dimension of arrray to check     -print(img.ndim)
		img=self.dilation_b(img,struct)
		return img
		
	def erosion(self,img,shape,*args):
		if(str(shape)=='square'):
			struct=self.square(args[0])
		elif(str(shape)=='rectangle'):
			struct=self.rectangle(args[0],args[1])
		else:
			print('Enter a valid shape and dimension')
		img= self.binary(img)
		#printing dimension of arrray to check     -print(img.ndim)
		img=self.erosion_b(img,struct)
		return img

	def erosion_b(self,img,struct):
		im=img.shape
		s=struct.shape
		#print(s)
		#print(struct)
		img=img/255
		#print(img)
		R=im[0]+s[0]-1
		C=im[1]+s[1]-1
		N=np.zeros((R,C))
		#print(N)
		for i in range(im[0]):
			for j in range(im[1]):
				N[i+1,j+1]=img[i,j]
				
		#print(N)

		for i in range(im[0]):
			for j in range(im[1]):
				k=N[i:i+s[0],j:j+s[1]]
				#print(k)
				result= (k==struct)
				final=np.all(result==True)
				if final:
					img[i,j]=1
				else:
					img[i,j]=0
		return img*255


	def dilation_b(self,img,struct):
		im=img.shape
		s=struct.shape
		#print(s)
		#print(struct)
		img=img/255
		#print(img)
		R=im[0]+s[0]-1
		C=im[1]+s[1]-1
		N=np.zeros((R,C))
		#print(N)
		for i in range(im[0]):
			for j in range(im[1]):
				N[i+1,j+1]=img[i,j]
				# N[i+(np.int((s[0]-1)/2)),j+np.int((s[0]-1)/2)]=img[i,j]
		#print(N)

		for i in range(im[0]):
			for j in range(im[1]):
				k=N[i:i+s[0],j:j+s[1]]
				#print(k)
				result= (k==struct)
				final=np.any(result==True)
				if final:
					img[i,j]=1
				else:
					img[i,j]=0
		return img*255


	def opening(self,img,shape,*args):
		if(str(shape)=='square'):
			struct=self.square(args[0])
		elif(str(shape)=='rectangle'):
			struct=self.rectangle(args[0],args[1])
		else:
			print('Enter a valid shape and dimension')

		img= self.binary(img)
		img1=self.erosion_b(img,struct)
		img2=self.dilation_b(img1,struct)
		return img2

	def skeleton(self,img):
		img=self.binary(img)
		struct=self.square(4)
		im=img.shape
		s=struct.shape
		#print(s)
		#print(struct)
		img=img/255

		cp=np.copy(img)
		
		R=im[0]+s[0]-1
		C=im[1]+s[1]-1
		N=np.zeros((R,C))
		copy=N
		#print(N)
		for i in range(im[0]):
			for j in range(im[1]):
				N[i+1,j+1]=img[i,j]
		# 		# N[i+(np.int((s[0]-1)/2)),j+np.int((s[0]-1)/2)]=img[i,j]
		#print(N)

		for i in range(im[0]):
			for j in range(im[1]):
				k=N[i:i+s[0],j:j+s[1]]
				#print(k)
				result= (k==struct)
				final=np.all(result==True)
				if final:
					img[i,j]=1
				else:
					img[i,j]=0
		#img=img*255
		# 

		img=np.subtract(cp,img)
		# print(img)
		return img

	def closing(self,img,shape,*args):
		if(str(shape)=='square'):
			struct=self.square(args[0])
		elif(str(shape)=='rectangle'):
			struct=self.rectangle(args[0],args[1])
		else:
			print('Enter a valid shape and dimension')

		img= self.binary(img)
		img1=self.dilation_b(img,struct)
		img2=self.erosion_b(img1,struct)
		return img2

	