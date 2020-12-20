# Sparse Uncoorelated Linear Discriminant Analysis (Group No: 11)

## Group Members: 

Mahak Jain 0801CS171040 </br>
Divyansh Joshi 0801CS171023 </br>

## ULDA




## SULDA


## Example:
`from sklearn import datasets`

`import matplotlib.pyplot as plt`

`import numpy as np`

`from ulda import ULDA`

Project the data onto the 2 primary linear discriminants

**`ulda = ULDA(2)`**

`ulda.fit(X, y)`

`X_projected = ulda.transform(X)`

