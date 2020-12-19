
# **DisCCA: Discriminant Canonical Correlation Analysis**

This is an implementation of Discriminant Canonical Correlation Analysis (DCCA or Deep CCA) in Python 

 Implementation is based on the following paper:

[Canonical Correlation Analysis (CCA) Based Multi-View Learning: An Overview
Chenfeng Guo and Dongrui Wu](https://arxiv.org/abs/1907.01693)


**Requirements**
```
numpy==1.19.3
pandas==1.1.5
python-dateutil==2.8.1
pytz==2020.4
scipy==1.5.4
six==1.15.0

```
**Documentation of API**

**DisCCA**

**DisCCA.fit**(_X_: _numpy.array_, _Y_: _numpy.array_, _target_: _numpy.array_, output_dimensions: _int_)

to fit weight according to training data

**Parameters**



*   **X** (_numpy ndarray_) – training data of view 1
*   **Y** (_numpy ndarray_) – training data of view 2
*   **target** (_array_) – list of target 

*   **outdim_size** (_int_) – output dimension of data 
get output of the model

Returns

None 

**DisCCA.transform**(_X_: _numpy.array_, _Y_: _numpy.array_)

to convert the data into new dimension

**Parameters**



*   **X** (_numpy ndarray_) –  data of view 1
*   **Y** (_numpy ndarray_) –  data of view 2



Returns transformed X and Y in new dimension.


**DisCCA.fit**(_X_: _numpy.array_, _Y_: _numpy.array_, _target_: _numpy.array_, output_dimensions: _int_)

to fit weight according to training data

**Parameters**



*   **X** (_numpy ndarray_) – training data of view 1
*   **Y** (_numpy ndarray_) – training data of view 2
*   **target** (_array_) – list of target 

*   **outdim_size** (_int_) – output dimension of data 
get output of the model

Returns List containing transformed matrices .

**DisCCA.get_within_class_similarity**()

to get within class similarity

**Parameters**

None

Returns within class similarity



**Example**


```

# Program containing example use of DisCCA

from DisCCA import DisCCA
import pandas as pd

first_dataframe=pd.read_csv("first_view",delim_whitespace=True,header=None)
second_dataframe=pd.read_csv("second_view",delim_whitespace=True,header=None)

target=[]
for i in range(10):
    for j in range(200):
        target+=[i]

D_CCA=DisCCA()
D_CCA.fit(first_dataframe,second_dataframe,target, 47)

transformed_first_dataframe,transformed_first_dataframe=D_CCA.transform(first_dataframe, second_dataframe)


```



