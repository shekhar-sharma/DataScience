import cv2 as cv
from package.edge_direction_histogram import edge_direction_histogram as edh
# while running please ensure this example is in same directory as package

image = cv.imread('images/lines.jpg', 0) # read in gray scale
features = edh.edge_direction(image)
print(features)
