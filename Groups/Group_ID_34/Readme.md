## Group ID: 34

## Deep CCA Auto-encoders (DCCAE) -
---------------------------------------

### Conceptual Overview:

Deep CCA computes representations of the two views by passing them through multiple stacked layers of nonlinear transformation. 
Deep canonical correlation analysis (DCCA), first extracts nonlinear features through deep neural networks (DNNs), 
and then uses linear CCA to calculate the canonical matrices. 

Similar to DCCA, discriminative deep canonical correlation analysis (DisDCCA) is a DNN-based extension of discriminative CCA. Discriminative deep canonical correlation analysis (DisDCCA) simultaneously learns two deep mapping networks of the two sets to maximize the within-class correlation and minimize the inter-class correlation.
 
Analysis and working of disDCCA is similiar to DCCA as in disDCCA also we first calculate the loss of each batch and then train the model.

### The following folder contains a detailed report on DCCA, disDCCA, DCCAE and disDCCAE along with their algorithm and implementation. 

----------------------------------------------------------

### Dataset reference:

All models are evaluated on a noisy version of MNIST dataset.

--------------------------------------------------------------------

### Packages and libraries used:

numpy~=1.19.3

torch~=1.7.1+cu110

scikit-learn~=0.23.2

scipy~=1.5.4

matplotlib~=3.3.3

torchvision~=0.8.2+cu110

pandas~=1.1.5

------------------------------------------------------------------------

### Dependencies:

The code was tested using:

tensorflow==1.12 and Keras==2.2.4

------------------------------------------------------

### Installation:

Install $DCCAE_repo by running:
```
pip install DCCAE_repo
```

-------------------------------------------------

### Inside directory:

The folder contains the following:

### DCCA.ipynb

This file contains the implementation of Deep Canonical Correlation Analysis using pytorch.

It contains the following dependencies from torch, configuration, and objective libraries:
matmul, optim, nn, functional

To Run:
```
   import DCCAE_repo
   cfg = Config()
   
   #train_set_1 and train_set_2 are 2 numpy arrays with the same number of samples but potentially different numbers of features
   
   dcca = DCCAE_repo.deepwrapper.DeepWrapper(cfg)
   
   dcca.fit(train_set_1, train_set_2)
```
-----------------------------------------------------------------------------------------------------------------------------

### Deep Canonically Correlated Autoencoders / DCCAE.ipynb:

Contains code on Deep Canonically Correlated Autoencoders (DCCAE), implementation of DCCAE and corresponding documentation. 

It contains the following dependencies from torch and configuration libraries: nn, optim, nn.functional

-----------------------------------------------------------------------------------

### Configuration.ipynb:

The main difference between running linear models and deep learning based models is the Config class from configuration.py.

Config is introduced in order to allow better flexibility of encoder and decoder architectures within broader model architectures
(Deep Canonical Correlation Analysis, Deep Canonically Correlated Autoencoders).

Config() holds default settings such that all model architectures will work out.

-------------------------------------------------------------------------------

### DeepWrapper.ipynb

It contains the fit and transform functions for Deep Canonical Correlation Analysis based on DCCA.

-------------------------------------------------------------------------------

### __init__.ipynb
This file contains curated dependencies that are pervasive in this project.

----------------------------------------------------------------------------

### objectives.ipynb

This file contains classes and methods for optimization test of DCCAE

-------------------------------------------------------------------------------

### test_deepwrapper.ipynb

This contains methods for testing the DCCA and the DCCAE models using Testcase from unittest, in conformation with config file.

--------------------------------------------------------------------

### Deep CCA Auto-encoders Report

This file contains the analysis and description of the core concepts of multiview learning primarily focusing on Deep canonical correlation analysis and Deep CCA Auto-encoders. It contains detailed working, algorithm and explaination of concepts related to DCCA, disDCCA, DCCAE, disDCCAE. 

--------------------------------------------------------------------------



