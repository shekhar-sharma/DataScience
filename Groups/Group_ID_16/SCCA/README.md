### Author
Farida Fakhri       0801CS171024

# SCCA

# PreRequisites

  - NumPy Module
  - Keras Module





### Installation
pip install numpy
pip install keras




### Usage



```sh
from scca import *
fit_transform([dataset1, dataset2], nvecs, verbose)
```

### Example Code

```sh
N = 10 # number of data instances
NVECS = 2 # number of reduced dimensions

# Generate random data
X = np.random.rand(N, 3)
Y = np.random.rand(N, 4)

# Center X to it's mean
X_mean = X.mean(axis = 0)
X = X - X_mean
# Center Y to it's mean
Y_mean = Y.mean(axis = 0)
Y = Y - Y_mean

(u_comp, v_comp), (x_proj, y_proj) = fit_transform([X, Y], nvecs = NVECS, verbose = 1)
```

### File Descriptions

 - scca.py - It contains main code of the module.
 - SCCA_DOCUMENTATION_CS24.pdf - It contains report of the project.
 - example_scca.ipynb - It contains example code.
 - README.md - It contains description of all the things.
 - SCCA_Code_Documentation.html - It contains documentation of API
 
