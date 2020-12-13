import skimage.morphology  as mp   
import numpy as np
from disk import disk_r

a= np.array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])

selem=disk_r(1)
print(selem)

print(mp.binary_dilation(a,selem).astype(int))