import cv2
from matplotlib import pyplot as plt

from fourier_transform import fourier_transform

img = cv2.imread("a_0.png",0)
magnitude_spectrum = fourier_transform(img)

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
