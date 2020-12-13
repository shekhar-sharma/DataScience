import cv2
import numpy as np

from kl_transform import kl_transform

#reading a grayscale image
img = cv2.imread('Lenna_(test_image).png', 0)

#calling the function
klt, eigenVal, eigenVec = kl_transform(img)

#displaying the kl transformed image
cv2.imshow("KL Transform",np.uint8(klt))

# cv2_imshow(np.dot(klt.T,eigenVec[:100,:512]).T)
#getting back the original image
cv2.imshow("Retrieved Image ",np.uint8(np.dot(klt.T,eigenVec).T))
cv2.waitKey(0)
cv2.destroyAllWindows()
