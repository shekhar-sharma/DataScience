# Cross regression for multi-view feature extraction



## Prerequisite

NumPy module
Pillow module (for image)


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install numpy
pip install pillow
```
Download crmvfe.py file from this repo and place it in the folder where your code file is present, it can be easily imported and used then.

Note 1: One needs to update the number of views and objects in the CRMvFE.py file also.

Note 2: All the images present in the coil-20_32x32 folder needs to be in the same folder as CRMvFE.py and test.py files.

## Usage

```python
import crmvfe
crmvfe.CRMvFE(arguments)
```

## Example

```python
import CRMvFE

views=8
objects=10

X = CRMvFE.get_input(views, objects)
Z = CRMvFE.crmvfe(X, views, objects)
print(Z)
```

## File Descriptions
* CRMvFE.py: Package for 'Cross regression for multi-view feature extraction'
* CRMvFE_Documentation.html: Documentation for CRMvFE package
* CRMvFE_Report.pdf: Report for Cross regression for multi-view feature extraction algorithm
* Group_members.txt: Group details
* test.py: test file for the implemented package
 
## Authors
* Utkarsh Shrivastava, 0801CS171090
* Sachin Motwani, 0801CS171065
