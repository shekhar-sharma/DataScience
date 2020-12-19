import cv2
from matplotlib import pyplot as plt
'''Importing the package'''
from fourier_transform import fourier_transform

img = cv2.imread("a_0.png",0) #reading a grayscale image
magnitude_spectrum = fourier_transform(img) #using the function

plt.subplot(121),plt.imshow(img, cmap = 'gray') #plotting the output
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
cv2.imwrite("output.jpg", magnitude_spectrum) #saving output
plt.show()
