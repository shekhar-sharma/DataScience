# Regularized Generalized Canonical Correlation Analysis (RCCA)

## Package Name: RGCCA
```
import numpy as np
from sklearn.preprocessing import StandardScaler
```
```
import rgcca
```
## Parameters:

**data:** list of n views of training datasets  
&ensp;&ensp;&ensp;&ensp;&ensp;X<sub>1</sub> is 1<sup>st</sup> view of shape (4,5)  
&ensp;&ensp;&ensp;&ensp;&ensp;X<sub>2</sub> is 2<sup>nd</sup> view of shape (4,6)  
&ensp;&ensp;&ensp;&ensp;&ensp;... so on

## Methods:

**fit(data):** fits model to data  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; returns RGCCA object with weight matrices and correlation between variates
  
**transform(data):** scales the data using Standard Scaler of sklearn.preprocessing  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; returns the dot product of standardized data which is returned by fit method 

## Example:
```
x1 = np.random.randn(4,5)
x2 = np.random.randn(4,6)
x3 = np.random.randn(4,5)
x4 = np.random.randn(4,6)
x5 = np.random.randn(4,7)
```


```
rgc = rgcca.RGCCA()
```

```
rgc.fit([x1,x2,x3,x4,x5])
```
<rgcca.RGCCA at 0x2d71fc99888>
```
res = rgc.transform([x1,x2,x3,x4,x5])
print(res)
```
[array([[-0.16949665,  0.05793227],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.0211854 , -0.16001056],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.10798999, -0.03515629],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.08269206,  0.13723458]]), array([[-0.17090446,  0.0609658 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.02197387, -0.16084664],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.114171  , -0.03672295],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.07870733,  0.1366038 ]]), array([[-0.17219026,  0.05880788],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.02196213, -0.15672825],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.11426655, -0.03811533],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.07988584,  0.1360357 ]]), array([[-0.17219929,  0.06023548],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.02200528, -0.16072357],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.11385121, -0.0369466 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.08035336,  0.13743469]]), array([[-0.17245942,  0.0605209 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.02066908, -0.15806151],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.11352794, -0.03623691],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.07960056,  0.13377752]])]  
