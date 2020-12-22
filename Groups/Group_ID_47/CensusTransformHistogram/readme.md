---


# Census Transform Histogram

The Census Transform R(P) is a non-linear transformation which maps a local neighborhood surrounding a pixel P to a binary string representing the set of neighboring pixels whose intensity is less than that of P.

It is a non-parametric transform that depends only on relative ordering of intensities, and not on the actual values of intensity, making it invariant with respect to monotonic variations of illumination, and it behaves well in presence of multimodal distributions of intensity, e.g. along object boundaries.

With the help of this API you would be able to create Census Transform Histogram of any image.




---



## Files





* ### Documentation

    >Report and Documentation/[CensusTransformHistogram_Documentation.html](https://github.com/shekhar-sharma/DataScience/blob/main/Groups/Group_ID_47/CensusTransformHistogram/Report%20and%20Documentation/CensusTransformHistogram_Documentation.html)
 
   

* ### Report

  >Report and Documentation/[CensusTransformHistogram_report.pdf](https://github.com/shekhar-sharma/DataScience/blob/main/Groups/Group_ID_47/CensusTransformHistogram/Report%20and%20Documentation/CensusTransformHistogram_report.pdf)


* ### Census Transform Module

  >package/[CensusTransform.py](https://github.com/shekhar-sharma/DataScience/blob/main/Groups/Group_ID_47/CensusTransformHistogram/package/CensusTransform.py)

* ### Jupyter Notebook with examples

  >/[Examples(test).ipynb](https://github.com/shekhar-sharma/DataScience/blob/main/Groups/Group_ID_47/CensusTransformHistogram/Examples(test).ipynb)

* ### Test file

  >/[test.py](https://github.com/shekhar-sharma/DataScience/blob/main/Groups/Group_ID_47/CensusTransformHistogram/test.py)

* ### Example Images folder

  >/[images/](https://github.com/shekhar-sharma/DataScience/tree/main/Groups/Group_ID_47/CensusTransformHistogram/images)



---
## Installation

>Keep the *package* folder in the project library and import using:

```python
from package.CensusTransform import CensusTransformHistogram
```



---

## Usage



>Refer [Examples(test).ipynb](https://github.com/shekhar-sharma/DataScience/blob/main/Groups/Group_ID_47/CensusTransformHistogram/Examples(test).ipynb) for usage examples and additional details.


---
