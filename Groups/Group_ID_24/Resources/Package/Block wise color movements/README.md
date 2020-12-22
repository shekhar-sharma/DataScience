# Block wise color movements

## Introduction
The recent emergence of multimedia and the availability of large images have made content-based information retrieval an important research topic. The most frequently cited visual contents for image retrieval are colour, texture, and shape. Among them, the colour feature is most commonly used Stricker and Orengo used colour moments as feature vectors for image retrieval. Since any colour distribution can be characterised by its moments, and most information is concentrated in the low-order moments, only the first moment (mean), the second moment (variance) and the third moment (skewness) are taken as features

## Library Used
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import rgb_to_hsv
from scipy.stats import skew
```
## Parameters:

**data:** An image is partitioned into blocks of size (e.g. 4*4) to form matrices of pixel values.

## Methods:

**mean(a):** passing the block(matrices of pixel) 
<br> whose values are summed up and divided by length of matrices to find average or mean value stored in variable res.
  
**std(a):** Scans the matrices/Block and each value of matrices is further used to find the variance.

**skew(a):** Passed the Block/matries and read all the pixel value present in block to get the skewness of each block.

**img_partition(img, gpc):**  img ->image in HSV format &ensp;&ensp; gpc: grid partition constant for partitioning the Image into Blocks
<br>  &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; Returns the image into blocks of defined size

**extractCM(img, gpc):**  img ->image in HSV format &ensp;&ensp; gpc: grid partition constant for partitioning the Image into Blocks
<br>  &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;method to Extract and appends the value of features(mean, variance, skewness) of blocks into an array

## Formula Used
```
mean = sum of the value of terms/number of terms
Skew = 3 * (Mean – Median) / Standard Deviation.
```
**Mean**  mean is the average of the numbers. It is easy to calculate: add up all the numbers, then divide by how many numbers there are. In other words it is the sum divided by the count.

**variance**  The sum of the squared distances of each term in the distribution from the mean (μ), divided by the number of terms in the distribution (N).

**Skewness** It can be understood as a measure of the degree of asymmetry in the distribution.
