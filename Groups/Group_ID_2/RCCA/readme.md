# Regularized Canonical Correlation Analysis (RCCA)

## Package Name: RCCA
```
import rcca
import numpy as np
from sklearn.preprocessing import StandardScaler
```
## Parameters:

**data:** list of two views of training dataset (ndarray of shape(d<sub>i</sub>, n))

## Methods:

**fit(data):** fits model to data  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; returns RCCA object with weight matrices (w<sub>x</sub> and w<sub>y</sub>) and correlation between variates
  
**transform(data):** scales the data using Standard Scaler of sklearn.preprocessing  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; returns the dot product of standardized data which are returned by fit method

**fit_transform(self,data):** To get the combined result of fit and transform method as reduced data.  
Parameters:  
&emsp;&emsp; data: datasets in the form of list of length 2.

## Example:
```
samples = 6

latvar1 = np.random.randn(samples,)
latvar2 = np.random.randn(samples,)

indep1 = np.random.randn(samples, 7)
indep2 = np.random.randn(samples, 8)

# Create two datasets, with each dimension composed as a sum of 75% one of the latent variables and 25% independent component
a = np.array((latvar1,latvar2,latvar1,latvar2,latvar1,latvar2,latvar1,latvar2,latvar1,latvar2,latvar1,latvar2,latvar1,latvar2))
data1 = 0.25*indep1 + 0.75*np.vstack(a[:7]).T
data2 = 0.25*indep2 + 0.75*np.vstack(a[:8]).T

train1 = data1[:4]
train2 = data2[:4]
test1 = data1[4:]
test2 = data2[4:]
```


```
cca = rcca.RCCA(n_comp = 2,reg_param = 100)
cca.fit([train1,train2])
```

```
cca = rcca.RCCA(n_comp = 2,reg_param = 0.6)
sc = StandardScaler()
t1,t2 = sc.fit_transform(train1),sc.fit_transform(train2)
cca.fit([t1,t2])
```
Training RCCA with regularization parameter = 0.6 and 2 components  
<rcca.RCCA at 0x1572b412a08>
```
cca.transform([t1,t2])
```
[array([[-0.04716616,  0.87880051],  
&emsp;        [ 0.81485978, -0.61138334],  
&emsp;        [-1.03193654, -0.46340391],  
&emsp;        [ 0.26424292,  0.19598673]]),  
 array([[-0.0776772 ,  0.94941007],  
&emsp;        [ 0.78324423, -0.61667998],  
&emsp;        [-1.03945516, -0.49006278],  
&emsp;        [ 0.33388813,  0.15733269]])]  
```
cca.variates
```
array([0.99811145, 0.99879991])  
```
cca.weights
```
[array([[-0.10317755,  0.23265324],  
&emsp;        [ 0.11226636,  0.23116019],  
&emsp;        [-0.09207226,  0.19636855],  
&emsp;        [ 0.10464022,  0.19324599],  
&emsp;        [-0.13059829,  0.11761277],  
&emsp;        [ 0.11319952,  0.21327369],  
&emsp;        [-0.10196098,  0.08679625]]),  
 array([[-0.07180132,  0.2069274 ],  
&emsp;        [ 0.11103668,  0.10461872],  
&emsp;        [-0.10239331,  0.19127895],  
&emsp;        [ 0.10072955,  0.11186391],  
&emsp;        [-0.10669881,  0.18798056],  
&emsp;        [ 0.09412687,  0.15772927],  
&emsp;        [-0.09770648,  0.01904776],  
&emsp;        [ 0.10322083,  0.15961865]])]  
```
cca.c_comp
```
[array([[-0.04716616,  0.87880051],  
&emsp;        [ 0.81485978, -0.61138334],  
&emsp;        [-1.03193654, -0.46340391],  
&emsp;        [ 0.26424292,  0.19598673]]),  
 array([[-0.0776772 ,  0.94941007],  
&emsp;        [ 0.78324423, -0.61667998],  
&emsp;        [-1.03945516, -0.49006278],  
&emsp;        [ 0.33388813,  0.15733269]])]  
