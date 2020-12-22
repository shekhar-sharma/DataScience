from package.CensusTransform import CensusTransformHistogram as CTH
from PIL import Image
import numpy as np

image = Image.open('images/ex1.jpg')

#window size
WS5 = CTH(5)
processedimg = WS5.preprocess(image)
ct_img = WS5.censustransform(processedimg)

#histogram of original image
WS5.imghistogram(image, "Original Image:")

#histogram of preprocessed image
WS5.imghistogram(processedimg, "Grayscale(preprocessed) image:")

#Census transform histogram of image
WS5.imghistogram(ct_img, 'Census Transform Histogram of image:')