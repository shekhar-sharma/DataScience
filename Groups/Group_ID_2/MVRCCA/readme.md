# MultiView Regularized Canonical Correlation Analysis (MVRCCA)

## Package Name: MVRCCA
```
import numpy as np
from sklearn.preprocessing import StandardScaler
```
```
import mvrcca
```
## Parameters:

**data:** list of n views of training datasets  
&ensp;&ensp;&ensp;&ensp;&ensp;X<sub>1</sub> is 1<sup>st</sup> view of shape (4,5)  
&ensp;&ensp;&ensp;&ensp;&ensp;X<sub>2</sub> is 2<sup>nd</sup> view of shape (4,6)  
&ensp;&ensp;&ensp;&ensp;&ensp;... so on

## Methods:

**fit(data):** fits model to data  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; returns MVRCCA object with weight matrices and correlation between variates
  
**transform(data):** scales the data using Standard Scaler of sklearn.preprocessing  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; returns the dot product of standardized data which is returned by fit method 

## Example:
```
x1 = np.random.randn(4,5)
x2 = np.random.randn(4,6)
x3 = np.random.randn(4,5)
x4 = np.random.randn(4,6)
```


```
mvr = mvrcca.MVRCCA()
mvr.fit([x1,x2,x3,x4])
```
<mvrcca.MVRCCA at 0x20c088294c8>

```
res = mvr.transform([x1,x2,x3,x4])
```
```
print(res)
```
[array([[-0.15641757, -0.14274856],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.11667493, -0.05984903],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.08621575,  0.19310336],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.12595839,  0.00949422]]), array([[-0.15739442, -0.14281866],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.11682799, -0.05870743],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.08618499,  0.19246567],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.12675142,  0.00906042]]), array([[-0.15644633, -0.14272396],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.11655146, -0.06145792],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.08653512,  0.1933625 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.12643   ,  0.01081938]]), array([[-0.15734139, -0.14378836],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.11637749, -0.0575494 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.08646279,  0.19383508],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.12742668,  0.00750268]])]  
```
abc = mvrcca.MVRCCA()
res1 = abc.fit_transform([x1,x2,x3,x4])
```
[array([[-0.15641757, -0.14274856],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.11667493, -0.05984903],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.08621575,  0.19310336],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.12595839,  0.00949422]]), array([[-0.15739442, -0.14281866],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.11682799, -0.05870743],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.08618499,  0.19246567],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.12675142,  0.00906042]]), array([[-0.15644633, -0.14272396],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.11655146, -0.06145792],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.08653512,  0.1933625 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.12643   ,  0.01081938]]), array([[-0.15734139, -0.14378836],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.11637749, -0.0575494 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.08646279,  0.19383508],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [ 0.12742668,  0.00750268]])]  
