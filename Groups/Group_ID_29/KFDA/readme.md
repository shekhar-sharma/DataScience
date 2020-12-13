# Kernel Fisher Discriminant Analysis

kernel Fisher discriminant analysis (KFDA) also known as kernel discrimi-nant analysis, is a kernelized version of linear discriminant analysis (LDA).It is named after Ronald Fisher. Using the kernel trick, LDA is implicitlyperformed in a new feature space, which allows non-linear mappings to belearned.The principle that underlies KFDA is that input data are mapped into ahigh-dimensional feature space by using a nonlinear function, after whichFDA is used for recognition or classification in feature space.


## Package organization
Kernel Fisher Discriminant Analysis (KFDA) Discriminant Analysis in highdimension using the kernel trick.

```python
Class KFDA(n_components=2, kernel='linear', alpha=1e-3, tol=1e-4, **kprms)
```

**Parameters:**
- *n_components* : int, the amount of Fisher directions to use.default=2.This is limited by the amount of classes minus one.Number of components (lower than number of classes -1) for dimension reduction.

- *kernel* : str, [ ”linear”|”poly”|”rbf”|”sigmoid”|”cosine”|”precomputed”] default=”linear”.
The kernel to use. Use **kwds to pass arguments to these functions.See https://scikit-learn.org/stable/modules/metrics.html#polynomial-kernel for more details.

- *alpha* : float, default=1e-3
Regularization term for singular within-class matrix.

- *tol* : float, default=1e-4
Singularity toleration level.

- *kprms* : mapping of string to any, default=None
parameters to pass to the kernel function.

**Attributes:**
- X : Training vector after applying input validation
- y : label vector after applying input validation
- W_ : array of shape (n_components) contains weights of eigen vectors
- unique_classes : array of shape (n_classes),The unique class labels

 ## Methods
```python
fit(self, X,y)
```
fit the model from the data in X and the labels in y.
	
**Parameters**
- *X* : array-like, shape (N x d) Training vector, where N is the number of samples, and d is the number of features.
-  *y* : array-like, shape (N) Labels vector, where N is the number of samples.
#### Returns
the instance itself (*self*)

------------------------------------------------------------------ 


```python
transform(self, X=None)
```
Applies the kernel transformation.
 
 **Parameters**
- *X* : Data to transform. If not supplied, the training data will be used.
	samples.
	
#### Returns
transformed : (N x d') matrix.
Input data transformed by the learned mapping.



## Usage
 **Example:1**
```python
from KFDA import KFDA
y=np.array ([ 1, 1, 1, 1, 2, 2, 2 ])
X=np.array ([[2, 3], [3, 3], [4, 5], [5, 5], [1, 0], [2 ,1], [3, 1]])
kfda = KFDA( n_components = 1 , kernel = "linear")
kfda.fit (X, y)
print(kfda.W_)
#array([[-0.46499795, -0.34924755, -0.28050306, -0.00799093, -0.65249595,-0.35993865,  0.16658463]]) 
```
 **Example:2**
```python
from KFDA import KFDA
y=np.array ([ 1, 1, 1, 1, 2, 2, 2 ])
X=np.array ([[2, 3], [3, 3], [4, 5], [5, 5], [1, 0], [2 ,1], [3, 1]])
kfda = KFDA( n_components = 2 , kernel = "linear")
kfda.fit (X, y)
trans=kfda.transform()
print(trans)
"""array([[-20.26033083],
       [-24.2726556 ],
       [-36.44210123],
       [-40.454426  ],
       [ -4.01232477],
       [-12.10320997],
       [-16.11553474]])
"""
```
