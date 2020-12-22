## Author
Nisreen Sabir
### Kindly Consider individual marking
# PCCA
## Pre-requisites
- NumPy Module
### Usage 
from pcca import *
fit_transform(params)
## Example Code
``` N = 10 # number of instances
NVECS = 2 # number of reduced dimensions
# Generate random data
X = np.random.rand(N, 4)
Y = np.random.rand(N, 5)

# Center X to it's mean
X_mean = X.mean(axis = 0)
X = X - X_mean
# Center Y to it's mean
Y_mean = Y.mean(axis = 0)
Y = Y - Y_mean

Z, X1, X2 = pcca([X, Y], nvecs = NVECS, nprojs = N)
```
## File Descriptions
- pcca.py -contains main code of the module
- PCCA_Documentation .pdf-contains report of the project
- Example_pcca.ipynb -contains example code 
- PCCA_HTML_Documentation - contains html documentation of the code
- Readme.md - contains the description of all the files in this folder.
