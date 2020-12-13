# import skimage.morphology as mp
# import numpy as np
# sele=mp.diamond(10,np.uint8)
# print(sele)

import skimage.morphology as mp
import numpy as np
def disk_r(radius):
	sele=mp.diamond(radius,np.uint8)
	return sele