import numpy as np
import l21cca

# X contains list of 10 views

X=[np.random.randn(50,50) for i in range(10)]
reduced1=l21cca.l21_cca(X,5)
print(reduced1)


reduced2=l21cca.l21_cca(X,10,20)
print(reduced2)
