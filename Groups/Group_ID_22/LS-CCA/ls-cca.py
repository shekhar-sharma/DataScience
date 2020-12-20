import numpy as np
from numpy import linalg as la
from numpy.lib import scimath as smath
from sklearn.preprocessing import StandardScaler
import pandas as pd

# m1 = np.array([[5, 10, 15], [3, 6, 9], [4, 8, 12]])
# m2 = np.array([[9, 18, 27], [11, 22, 33], [13, 26, 39]])

m1 = pd.DataFrame([[5, 10, 15], [3, 6, 9], [4, 8, 12]])
m2 = pd.DataFrame([[9, 18, 27], [11, 22, 33], [13, 26, 39]])

# m1 = np.random.randint(1,20,(10, 10))
# m2 = np.random.randint(1,20,(10, 10))

m1_rows, m1_cols = m1.shape
m2_rows, m2_cols = m2.shape

print(m1)
print(m2)

n_components = min(m1_cols, m2_cols)
rows = m1_rows # N

m1 = StandardScaler().fit_transform(m1)
m2 = StandardScaler().fit_transform(m2)

print(m1)
print(m2)

r11 = np.dot(np.transpose(m1), m1)
r12 = np.dot(np.transpose(m1), m2)
r21 = np.dot(np.transpose(m2), m1)
r22 = np.dot(np.transpose(m2), m2)

print("rxx xy")
print(r11)
print(r12)

# rmat = np.array([[m1.transpose()*m1, m1.transpose()*m2], [m2.transpose()*m1, m2.transpose()*m2]])

# dmat = np.array([[m1.transpose()*m1, np.zeros((rows, n_components), dtype=int)], [np.zeros((rows, n_components), dtype=int), m2.transpose()*m2]])

# tmat = smath.sqrt(r11*(-1)) * r12
# tmat = tmat * smath.sqrt(r22*(-1))

tmat = np.dot(r11**(-0.5), r12)
tmat = np.dot(tmat, r22**(-0.5))

u, s, vh = la.svd(tmat)

# canonical matrices

wx = np.dot(r11**(-0.5), u)
wy = np.dot(r22**(-0.5), vh.transpose())

print("wx, wy")
print(wx)
print(wy)

# canonical variables

zx = np.dot(wx.transpose(), m1)
zy = np.dot(wy.transpose(), m2)

print("zx, zy")
print(zx)
print(zy)
