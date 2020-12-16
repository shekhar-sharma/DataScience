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
- *wx* , *wy* : final projection vectors of two views.

 ## Methods
```python
fit(self, x, y)
```
Fit the model from the data in x and the labels in y and finds the projection vectors

**Parameters**
- *x* : dataframe-like, shape (n x p)
      Training vector, where n is the number of samples, and p is the number of features.
- *y* : dataframe-like, shape (n x q)
      Training vector, where n is the number of samples, and q is the number of features.

#### Returns
- *wx* , *wy* : Projection vectors
------------------------------------------------------------------ 


```python
fit_transform(self, x, y)
```
Applies the projection vectors on the dataset 

 **Parameters**
- *x* : dataframe-like, shape (n x p)
    Training vector, where n is the number of samples, and p is the number of features.
- *y* : dataframe-like, shape (n x q)
    Training vector, where n is the number of samples, and q is the number of features.
	
#### Returns
*x_new* , *y_new* : Projected views
Input data transformed by the projected vectors



## Usage
 **Example:1**
```python
from SDCCA import SDCCA
x = pd.DataFrame([[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]])
y = pd.DataFrame([[0.1, 0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]])
sdcca = SDCCA(scale = False)
wx,wy = sdcca.fit (x, y)
print(wx)
"""
array([[-1.0930932 -0.0522547  0.0001379  0.0001285]
 [-1.1647563  0.0369279 -0.0007687 -0.000467 ]
 [-2.2689889 -0.1522781 -0.0021162 -0.0014276]
 [-0.849689   0.0001141  0.0006268  0.0003738]]) 
"""
print(wy)
"""
array([[-4.9045218 -0.0316331  0.0003566  0.0000783]
 [-1.9608417  0.0077403 -0.0001677 -0.0000368]
 [-0.8116817  0.0079527 -0.0000677 -0.0000149]
 [-0.5704856  0.0044708 -0.000052  -0.0000114]])
"""
```
 **Example:2**
```python
from SDCCA import SDCCA
x = pd.DataFrame([[1., 2.1, 1.], [1., 2.1, 0.5], [2.,2.,2.], [3.,5.,4.]])
y = pd.DataFrame([[0.1, 4.2], [0.9, 2.1], [7.2, 5.9], [12.9, 1.3]])
sdcca = SDCCA(k=4)
x_new, y_new = sdcca.fit_transform(x,y)
print(x_new)
"""
    0      1      2
0   -31.28 -49.88 -29.80
1   -12.14  -6.69 -13.22
2   -0.14  -0.01  -0.37
3   -0.00  -0.00  -0.00
"""
print(y_new)
"""
    0      1
0   -42.81 -30.81
1   -3.89  -4.68
2   2.33   1.63
3   6.69  -0.29
"""
```
