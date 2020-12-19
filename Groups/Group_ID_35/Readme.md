# Group ID : 35

* Aadeesh Jain (0801CS171001)
* Harsh Pastaria (0801CS171027)
* Kanishk Gupta (0801CS171031)

# Project Title 

* **MLDA** - Multiview Linear Discriminant Analysis
* **MULDA** - Multiview Uncorrelated Linear Discriminant Analysis

# MLDA :

LDA is supervised algorithm for a single view that minimize the within-class variance and maximize between-class variance. MLDA combines **LDA** and **CCA**. Through optimizing the corresponding objective, discrimination and correlation between each views can be maximized. 

# Usage Example:

import mlda <br />
import pandas as pd <br />
import os <br />

Mlda = mlda.MLDA()                                  //return object of MLDA class <br />
vTransforms = Mlda.fit_transform(X,Y,row,col,n)     //return projection directions of common space <br />
