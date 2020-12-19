from sklearn.datasets import load_iris
from RMvFS import RMvFS

r = RMvFS(threshold=0.5)
X,y = load_iris(return_X_y=True)
print("shape of X:")
print(X.shape)
r.fit(X,Y=y)
print(r.transform(X))
print(r.feature_importance)
