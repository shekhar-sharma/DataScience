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

## Usage Example

```python
from TensorCCA import * 

tencca = TensorCCA(max_iter) # returns object of TensorCCA class
fit = tencca.fit(Views, reduce_to_dim) # returns canonical vector H
reducedData = tencca.transform(Views) # returns views with reduced dimensions
```

