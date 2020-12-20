# Sparse Uncorrelated Linear Discriminant Analysis (Group No: 11)

## Group Members: 

Mahak Jain 0801CS171040 </br>
Divyansh Joshi 0801CS171023 </br>

## ULDA
ULDA(Uncorrelated Linear Discriminant Analysis) is a supervised method for feature extraction (FE), discriminant analysis (DA) and biomarker screening based on the Fisher criterion function. 
ULDA successfully extracted optimal features for discriminant analysis and revealed potential biomarkers. Furthermore, by means of cross-validation, the classification model obtained by ULDA showed better predictive ability than PCA, PLS-DA and TP-DA.


## SULDA
SULDA(Sparse Uncorrelated Linear Discriminant Analysis) is an improvement over ULDA where we find the sparse solution directly of the ULDA solution matrix by finding the l1- norm solution from all the solutions with minimum dimension (using Accelerated Linearized Bregman method).


## Example:
Import these

`from sklearn import datasets`

`import matplotlib.pyplot as plt`

`import numpy as np`

`from ulda import ULDA`

Project the data onto the 2 primary linear discriminants

**`ulda = ULDA(2)`**

Fit Model according to your data

`ulda.fit(X, y)`

`X_projected = ulda.transform(X)`



