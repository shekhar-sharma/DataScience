# Group ID: 27

- Shivam Kumar Mahto (0801CS171078)
- Vishal Rochlani (0801CS171092)


Project Title: 
1. Multiview Discriminant Analysis.
2. Multiview Fisher Discriminant Analysis.
# mvda:

MultiView Discriminant analysis seeks a nonlinear discriminant and view-invariant representation which is shared among multiple views. MvDA employs a novel eigenvalue-based multi-view decomposition function to encapsulate as much discriminative variance as possible into all the available common feature dimensions.
## Installation:



```bash
pip install pytorch
pip install numpy
```

## Parameters:

Views : List of n-dimensional array, where each n-dimensional array represents a view.

Class : independent variable(i,e class) for sample of views.



## Usage Example:

```python
import mvda
import numpy as np
import torch

MvDA = mvda.MVDA() # returns object of MVDA class.
vTransforms = MvDA.fit_transform(Views, Class) # returns projection of common space.

```

