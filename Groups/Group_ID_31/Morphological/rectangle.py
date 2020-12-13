# import skimage.morphology as mp
# import numpy as np
# sele=mp.rectangle(5,6,np.uint8)
# print(sele)

import skimage.morphology as mp
import numpy as np
def rectangle_wh(width,height):
	sele=mp.recatngle(width,height,np.uint8)
	return sele