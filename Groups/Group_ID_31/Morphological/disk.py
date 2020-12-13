import skimage.morphology as mp
import numpy as np
def disk_r(radius):
	sele=mp.disk(radius,np.uint8)
	return sele