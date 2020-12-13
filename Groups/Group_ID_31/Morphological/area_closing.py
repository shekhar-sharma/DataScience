import skimage.morphology as mp
import cv2 as cv

img=cv.imread("bw.jpg")
closed=mp.area_closing(img,135,connectivity=1)
cv.imshow('closed',closed)
cv.waitKey(0)
