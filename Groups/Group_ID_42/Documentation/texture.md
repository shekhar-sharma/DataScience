# Wavelet Texture

## Prerequisite

numpy

skimage.feature(greycomatrix,greycoprops)

from skimage import io,color,img_as_ubyte



## Installation

pip install numpy

pip install skimage 

## Usage
```python
from skimage import io
img=io.imread('C:/ProgramData/imgg.png')
```
```python
from skimage import color 
gray = color.rgb2gray(img)
```
```python
from skimage import img_as_ubyte
image = img_as_ubyte(gray)

```
```python
from skimage.feature import greycoprops
contrast=greycoprops(x,'contrast')
```

## Example Code

```python
import numpy as np
from  skimage.feature import greycomatrix, greycoprops
from skimage import io,color, img_as_ubyte
 
def contrast_feature(x):
    contrast=greycoprops(x,'contrast')
    return "Contrast =",contrast
 
# Load image=
img=io.imread('C:/ProgramData/imgg.png')
gray = color.rgb2gray(img)
image = img_as_ubyte(gray)
max_value = img.max()
matrix_coocurrence = greycomatrix(image, [1], [0, np.pi/4, np.pi/2],                                             
         levels=max_value+1, normed=False, symmetric=False)
 
print(contrast_feature(matrix_coocurrence))
```
