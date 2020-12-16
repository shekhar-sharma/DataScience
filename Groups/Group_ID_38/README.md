# l21cca

## Prerequisite

NumPy module (pip install numpy)

## Installation

Download l21cca.py file and go to the folder where your code file is present and 
copy this file. Now, you are able to import l21cca.


## Usage

```python

import l21cca
l21cca.l21_cca(data, reduced_dimension, regularisation_parameter)
```


## Example code

```python
import numpy as np

import l21cca

X=[np.random.randn(10,10) for i in range(10)]

reduced=l21cca.l21_cca(X,5,100)
```
## File descriptions
* l21cca.py - It contains main code of the module.
* 0801CS171025_Report.pdf - It contains report of the project.
* test.py - It contains example code.
* Documentation.html - It contains documentation of API.
* 0801CS171025_PPT.pptx - It contains presentation of Project-1 and Project-2.
* README.md - It contains description of all the things.

## Author
Gunjan Pandey
0801CS171025
