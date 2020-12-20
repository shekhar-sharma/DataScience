# Twice Wavelet Transform

## Prerequisite

matplotlib
pywt 


## Installation

pip install matplotlib
pip install PyWavelets

## Usage
```python
import matplotlib.image as mpimg
img=mpimg.imread('C:/ProgramData/imgg.jpeg')
```
```python
import pywt
coeffs2 = pywt.dwt2(img, 'bior1.3')
```

## Example Code

```python
import matplotlib.pyplot as plt
import pywt
import pywt.data
import matplotlib.image as mpimg
 
img=mpimg.imread('C:/ProgramData/imgg.jpeg')
 
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(img, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
 
fig.tight_layout()
plt.show()
```
