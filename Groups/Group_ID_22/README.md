# LSCCA (Least Squares Canonical Correlation Analysis)

Group ID: 22
Sarvagna Shukla
0801CS171067

Canonical correlation analysis (CCA) is a well-known technique in multivariate statistical analysis to find maximally correlated projections between two data sets. CCA only leverages two views of the data, extending it to **more than two views** is achieved with **Least Squares based CCA (LS-CCA)**.

This directory contains a folder LS-CCA which contains the following files.

- ls-cca.py
- LSCCA_API_Documentation.html (*with Example*)
- LSCCA_Report.pdf

## Usage

import ```ls-cca.py``` file and write the following
```py
lscca = LSCCA(<number of views>, <scale>)
W = lscca.fit(<array of datasets>)
```
```W```  will be an array having all the weight matrices respective to the datasets.
