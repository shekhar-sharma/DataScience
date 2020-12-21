from PIL import Image
import numpy as np

views = 8
objects = 10

def get_input(V, O):
  X = np.zeros((11,32,32))

  for view in range(V):
    #initialize current view with a 32x32 matrix
    curr_view = np.zeros((32,32))
    
    for object in range(1,O + 1):
      
      img = np.array(Image.open('obj'+str(object)+'__'+str(view)+'.png'))
      # print('image', img.shape)
      curr_view = np.append(curr_view, img, axis = 0)

    # print(curr_view.shape)
    # now curr_view is in 11 128x128 matrix form
    curr_view = np.reshape(curr_view, (11,32,32))
    curr_view = curr_view[1:]
    # print(curr_view.shape)

    X = np.append(X, curr_view, axis = 0)
  X = X[11:]
  X = np.reshape(X, (V,O,32,32))
  return X

def reduceDimension(X):
  X_2d =  np.zeros((objects,1024))
  for view in range(views):
    temp = np.reshape(X[view], (objects,1024))
    X_2d = np.append(X_2d, temp, axis = 0)
  # removing the first zero matrix
  X_2d = X_2d[objects:]
  X_2d = np.reshape(X_2d, (views,objects,1024))
  return X_2d

def compute_L(X_2d):
  L = np.zeros((views, 1024,1024))
  for view in range(views):
    L = np.append(L, np.zeros((views, 1024,1024)), axis = 0)
  L = L[views:]
  L = np.reshape(L, (views,views,1024,1024))
  for view in range(views):
    # dimension of Xa -> (10, 1024)
    Xa = X_2d[view]
    # dimension of Xa' -> (1024, 10)
    Xa_t = np.transpose(Xa)

    # d_element -> multiply Xa and Xa'
    d_element = np.matmul(Xa_t, Xa)
    L[view][view] = d_element
  #reshape L to a 2d matrix as follows
  L = np.reshape(L,(1024*8, 1024*8))
  return L

gamma = 0.9

def compute_O(L, X_2d):
  I = np.identity(8192)

  term1 = np.add(L, gamma*I)

  #finally take inverse
  term1 = np.linalg.inv(term1)

  # print('Term 1 -> ', term1.shape)
  sum = np.zeros((10,8192))

  #calculating the summation
  for view in range(8):
    
    # dimension of Xa -> (10, 1024)
    Xa = X_2d[view]

    # dimension of Xa' -> (1024, 10)
    Xa_t = np.transpose(Xa)
    
    X_sum = np.reshape(X_2d, (10,8192))
    
    sum = np.add(sum,np.matmul(np.matmul(Xa,Xa_t),X_sum))

  temp_X2d = np.reshape(X_2d, (10,8192))

  #calculating the term 2
  term2 = np.matmul(np.transpose(temp_X2d), sum)
  # print('Term2 -> ',term2.shape)
  O = np.matmul(term1, term2)
  return O


def compute_P(O):
  eig_value, eig_vectors = np.linalg.eig(O)
  #choose first 8 eigen vectors for V = 8
  P = eig_vectors[:views,:40]
  return P


def compute_Z(P, X_2d):
  Z = np.zeros((8,256))
  for view in range(8):
    Xa = X_2d[view]
    Xa = np.reshape(Xa, (256,40))
    Pa = P[view,:]
    temp = np.matmul(Xa,Pa)

    Z = np.add(Z,temp)
  return Z

def crmvfe(X, views, objects):
  X_2d = reduceDimension(X)
  L = compute_L(X_2d)
  O = compute_O(L,X_2d)
  P = compute_P(O)
  Z = compute_Z(P, X_2d)
  return Z

# X = get_input(views, objects)
# Z = crmvfe(X, views, objects)
# Z.shape