# Tensor CCA

A library to calculate tensorCCA for given views.

## Installation



```bash
pip install requirements.txt
```

## Parameters

Views : List of n-dimensional array, where each n-dimensional array represents a view

reduce_to_dim : int. Dimension to which each view will be reduced

max_iter : an integer, (default 500)
the maximum number of iterations of the ALS Algorithm 

## Methods

fit(Views, reduce_to_dim):  Fit method for given views

views_hat( Views):  Mean centering of data

cov_matrix( Views): Calculate covariance matrix of each view

covariance_tensor(Views): Calculate covariance tensor of given views

root_inverse( Cpp): Calculate root inverse of covariance matrix of pth view

ttm(cov_ten, var_matrix_inverse): Calculate tensor times matrix i.e. mode-i product

tcca(Views, var_matrix, cov_ten, reduce_to_dim): Calculate canonical vectors

transform(Views): Reduce dimensions of each view


## Usage Example

```python
from TensorCCA import * 

tencca = TensorCCA(max_iter) # returns object of TensorCCA class
fit = tencca.fit(Views, reduce_to_dim) # returns canonical vector H
reducedData = tencca.transform(Views) # returns views with reduced dimensions
```

