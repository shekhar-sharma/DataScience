import cv2 as cv
from package.local_self_similarity import local_self_similarity as lss
# while running please ensure this example is in same directory as package

image = cv.imread('images/chai-pani.jpg')
image = cv.resize(image, (150, 100))
features = lss.local_self_similarity(image)
print(features)
