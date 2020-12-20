import numpy as np

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
