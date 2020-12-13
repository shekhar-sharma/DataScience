import numpy as np
import cv2

# color_histogram class that contains both global and regional
# color based histogram
class ColorHistogram:
    # contructor to take bins value
    def __init__(self, bins):
        self.bins = bins
    
    # _Histogram fuction will take two argument image and mask 
    # for global mask=None 
    # this function will caluculate histogram form given image for red, green and blue color
    # flattern this value in 1-D array
    # normalize all the value so that all the value lies between [0...1]
    # finally returns it
    def _Histogram(self, image, mask):
		    hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,[0, 256, 0, 256, 0, 256])  
		    hist = cv2.normalize(hist, hist).flatten()
		    return hist
    
    # this function will calculate histogram of red, green and blue without masking
    def Global(self, image):
        features = []
        hist=self.Histogram(image,None)
        features.extend(hist)
        return features
    
    # this function will calculate histogram of red, green and blue without masking
    # deviding image into 5 parts top-left,bottom-left,top-right,bottom-right abd center elliple
    # calculate histogram of red, green and blue for every image 
    # flattern into 1-D array and return normalize it
    def Regional(self,image):
        features = []
        h,w=image.shape[:2]
        
        # cX,cY are center points of image
        cX,cY=(int(w/2),int(h/2))
        
        # finding all the corner points
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),(0, cX, cY, h)]
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)

        # initilizing ellipse and giving its border points to create its mask
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        for (startX, endX, startY, endY) in segments:
			      cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
			      cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			      cornerMask = cv2.subtract(cornerMask, ellipMask)

                                # calculating histogram for every sub_image
			      hist = self._Histogram(image, cornerMask)
			      features.extend(hist)
        hist = self._Histogram(image, ellipMask)
        features.extend(hist)
        return features
