# Similarity Distance Based CCA

Through SDCCA latent space for multiview learning is determined by incorporating similarities and complementary information from multiple views with cross correlation analysis.


## Package organization

```python
Class SDCCA(scale = True, k = 2)
```

**Parameters:**
- *scale* : boolean, default = True, Whether to scale the data or not.

- *k* : int, default = 2, Number of neighbours to consider in K-Means Algorithm.

**Attributes:**
- *x* : first data set of dimension n x p with n samples and p features.
- *y* : second data set of dimension n x q with n samples and q features.
- *n* : number of samples in datasets
- *p* : number of features in x dataset
- *q* : number of features in y dataser
- *wx* , *wy* : final projection vectors of two views of p x p and q x q respectively

 ## Methods
```python
fit(self, x, y)
```
Fit the model from the data in x and the labels in y and finds the projection vectors

**Parameters**
- *x* : array-like, shape (n x p)
      Training vector, where n is the number of samples, and p is the number of features.
- *y* : array-like, shape (n x q)
      Training vector, where n is the number of samples, and q is the number of features.

#### Returns
- *wx* , *wy* : Projection vectors
------------------------------------------------------------------ 


```python
fit_transform(self, x, y)
```
Applies the projection vectors on the dataset 

 **Parameters**
- *x* : array-like, shape (n x p)
    Training vector, where n is the number of samples, and p is the number of features.
- *y* : array-like, shape (n x q)
    Training vector, where n is the number of samples, and q is the number of features.
	
#### Returns
*x_new* , *y_new* : Projected views
Input data transformed by the projected vectors



## Usage
 **Example:1**
```python
from SDCCA import SDCCA
x = np.array([[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]])
y = np.array([[0.1, 0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]])
sdcca = SDCCA(scale = False)
wx,wy = sdcca.fit (x, y)
print(wx)
"""
array([[-28.4951993   0.2682586  -0.975527 ]
 [-42.1618415  -0.5543857   0.2679151]
 [-35.6725722   0.3283515   0.5998168]]) 
"""
print(wy)
"""
array([[257.9830053  -0.0865349]
 [262.5667474   0.0852448]])
"""
```
 **Example:2**
```python
from SDCCA import SDCCA
x = np.array([[1., 2.1, 1.], [1., 2.1, 0.5], [2.,2.,2.], [3.,5.,4.]])
y = np.array([[0.1, 4.2], [0.9, 2.1], [7.2, 5.9], [12.9, 1.3]])
sdcca = SDCCA(k=4)
x_new, y_new = sdcca.fit_transform(x,y)
print(x_new)
"""
[[ -5.8380473  -7.4261078  -6.2538804]
 [ -5.823948   -7.4249809  -6.2690812]
 [ -12.4623148 -13.7376682 -12.9121544]
 [ -18.0069253 -21.6219808 -18.9701994]]
"""
print(y_new)
"""
[[  3.896946    12.9154699]
 [ -1.4534688   8.8647358]
 [ -22.7799413  38.134078 ]
 [ -50.2993919  40.4397741]]
"""
```
