## Prerequisites - 
* python 3.6+
* pytorch 1.0+
* NumPy 1.19+
* cca-zoo 1.1.4 (https://pypi.org/project/cca-zoo/)

## Methods in dgcca.py

1. **Class DNN** : Creates a new Deep Neural Network
* forward(self, l) : forward propagates input tensor into the DNN and returns the output tensor (overriden)

2.  **Class : DGCCA_architecture** - Defines the architecture for three DNNs
* forward(self, x1, x2, x3) : forward propagates x1 into the first DNN, x2 into the second DNN and x3 into the third DNN and returns the outputs. (overriden)

3.  **Class DGCCA** : Implements the DGCCA Algorithm
* fit_transform(self, train_x1, train_x2, train_x3, test_x1, test_x2, test_x3) : Learn and apply the dimension reduction on the train data batch-wise. Trains the networks in mini-batches.

*  predict(self, x1, x2, x3) - returns gcca loss as ndarray and output as list for given inputs x1, x2, x3 for view first, second, third respectively.

*  test(self, x1, x2, x3) - returns gcca loss mean and output as list for given inputs x1, x2, x3 for view first, second, third respectively.
---

## Example Code
contained in dgcca_example.ipynb

---
