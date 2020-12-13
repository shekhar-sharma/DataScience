# import skimage.morphology as mp
# import numpy as np
# sele=mp.square(5,np.uint8)
# print(sele)

import skimage.morphology as mp
import numpy as np
def square_s(side):
	sele=mp.disk(side,np.uint8)
	return sele