# Regularized Generalized Canonical Correlation Analysis (RGCCA)

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
&ensp;&ensp;&ensp;&ensp;&ensp;X<sub>1</sub> is 1<sup>st</sup> view of shape (r<sub>1</sub>,c<sub>1</sub>)  
&ensp;&ensp;&ensp;&ensp;&ensp;X<sub>2</sub> is 2<sup>nd</sup> view of shape (r<sub>2</sub>,c<sub>2</sub>)  
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

```
abc = rgcca.RGCCA()
res1 = abc.fit_transform([x1,x2,x3,x4,x5])
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

```
rgc.weights
```
[array([[ 0.07076432, -0.05528831],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [-0.07778884,  0.04116878],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [-0.00763935, -0.05428245],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [-0.05175223, -0.0253489 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [ 0.02329436, -0.00464579]]),  
 array([[-0.0353552 ,  0.0307235 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [ 0.01490408, -0.01447572],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [ 0.0460204 , -0.01135475],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [ 0.0383193 ,  0.03107638],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [ 0.02838632, -0.01749865],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [ 0.04974561,  0.05493907]]),  
 array([[-0.00916377, -0.0214017 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [ 0.05020281, -0.0219377 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;       [-0.04579865, -0.0108178 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [ 0.01722792, -0.00331134],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [ 0.01188978,  0.11156725]]),  
 array([[ 0.01338884, -0.04091204],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [-0.02425717, -0.03513895],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [-0.04830855, -0.00490196],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [ 0.03599462, -0.02984957],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [-0.02779467, -0.01062314],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [-0.00703663, -0.03264378]]),  
 array([[-0.01465044,  0.02532783],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [-0.05515995, -0.0717137 ],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [-0.00725292,  0.06431659],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [-0.02070039,  0.03913111],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [-0.02456448, -0.05027572],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [-0.04061569,  0.01100244],  
&ensp;&ensp;&ensp;&ensp;&ensp;        [ 0.00446662,  0.00592046]])]  
