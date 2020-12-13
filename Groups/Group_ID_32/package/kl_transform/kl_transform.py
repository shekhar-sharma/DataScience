import cv2
import numpy as np
# from google.colab.patches import cv2_imshow

'''
This method is used to get the KL-Transform of the image
The function requires an image as argument
It returns the transform, the transformation matrix, and the eigen values 
in descending order.
'''
def kl_transform(img):
  '''Finding the eigen vector and value of covariance matrix of the image'''
  eigenVal, eigenVec = np.linalg.eig(np.cov(img))

  '''Sorting the eigen vector in descending order of thier values'''
  idx = eigenVal.argsort()[::-1]   
  eigenVal = eigenVal[idx]
  eigenVec = eigenVec[:,idx]

  '''Finding the transform'''
  # klt = np.dot(eigenVec[:100,:512], img)
  klt = np.dot(eigenVec, img)
 
  return klt, eigenVal, eigenVec



'''
Example
img = cv2.imread('Lenna_(test_image).png', 0)
klt, eigenVal, eigenVec = kl_transform(img)
cv2_imshow((klt))

# cv2_imshow(np.dot(klt.T,eigenVec[:100,:512]).T)
cv2_imshow(np.dot(klt.T,eigenVec).T)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''