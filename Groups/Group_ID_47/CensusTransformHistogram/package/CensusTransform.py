import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt

class CensusTransformHistogram:

	#constructor to take windowsize (default value 3s)
	def __init__(self, win_size=3):
		self.win_size = win_size

	#preprocess funtion for preprocessing
	def preprocess(self, img):
		
		#half window size
		hws = self.win_size // 2

		#adding border of size hws
		bimg = ImageOps.expand(img, border = hws, fill = 'white')

		#GrayScaling the image
		gry_img = ImageOps.grayscale(bimg)

		return(gry_img)

	#imghistogram function for making histogram of image
	def imghistogram(self, img, imgtitle):
		plt.figure(figsize=(10,10))
		plt.title(imgtitle)
		plt.imshow (img, cmap = plt.cm.gray )
		plt.show()

	#Main censustransform Function
	def censustransform(self, img):

		#half window size
		hws = self.win_size // 2
        
                
		#storing image size in w and h
		w, h = img.size

		#Convert image to Numpy array
		src_bytes = np.asarray(img)

		#Initialize transform array
		census = np.zeros((h-hws*2, w-hws*2), dtype='uint8')

		#centre pixels, which are offset by (hws, hws)
		cp = src_bytes[hws:h-hws, hws:w-hws]
		#offsets of non-central pixels 
		offsets = [(j, i) for i in range(self.win_size) for j in range(self.win_size) if not j == hws == i]

		#Do the pixel comparisons
		for j,i in offsets:
			census = (census << 1) | (src_bytes[i: i+h - hws*2, j: j+w - hws*2] >= cp)

        #Convert transformed pixel array to image
		out_img = Image.fromarray(census)
        
        #Print information of image and window size
		print('\n Census transform array of image \n \n', census)
		print('\n Processed(bordered) Image size: %d x %d = %d' % (w, h, w * h))        
		print('\n Image mode:', img.mode)
		print('\n Window Size = ', str(self.win_size))
        



		return out_img


    















