# Group ID : 35

* Aadeesh Jain (0801CS171001)
* Harsh Pastaria (0801CS171027)
* Kanishk Gupta (0801CS171031)

# Project Title 

* **MLDA** - Multiview Linear Discriminant Analysis
* **MULDA** - Multiview Uncorrelated Linear Discriminant Analysis

# MLDA :

LDA is supervised algorithm for a single view that minimize the within-class variance and maximize between-class variance. MLDA combines **LDA** and **CCA**. Through optimizing the corresponding objective, discrimination and correlation between each views can be maximized. 

# MULDA :

Uncorrelated LDA (ULDA) is an extension of LDA by adding some constraints into the optimization objective of LDA, so that the feature vectors extracted by ULDA could contain minimum redundancy.
It extracts uncorrelated features in each view and computes transformations of each view to project data into a common subspace.


# Usage Example:

**MLDA**
import mlda <br />
import pandas as pd <br />
import os <br />

Mlda = mlda.MLDA()                                  //return object of MLDA class <br />
vTransforms = Mlda.fit_transform(X,Y,row,col,n)     //return projection directions of common space <br />

**MULDA**

import mulda <br />
import pandas as pd <br />
import os <br />

Mulda = mulda.MULDA()                                  //return object of MULDA class <br />
featureMatrix = Mulda.combined_Features(X,Y,n,row,col)     //returns combined final features extracted from X and Y <br />

