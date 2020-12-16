import cv2 as cv
from package.color_histogram import color_histogram as ch
# while running please ensure this example is in same directory as package

image = cv.imread('images/lines.jpg')
c = ch.ColorHistogram((8, 8, 8))
features = c.Regional(image)
print(features)
