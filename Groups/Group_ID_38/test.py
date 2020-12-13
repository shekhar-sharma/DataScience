import numpy as np
import l21cca
from sklearn.preprocessing import StandardScaler


X=[np.random.randn(10,10) for i in range(10)]
reduced1=l21cca.l21_cca(X,5)
reduced2=l21cca.l21_cca(X,10,20,5,5)
