**Deep Generalized Cannonical Correlation Analysis Implementaion for 3 Views**
==============================================================================

\

### Contents

1.  Classes in DGCCA.py file
2.  Class DNN
3.  Class DGCCA\_architecture
4.  Class DGCCA
5.  Example Using Random Data

\

* * * * *

\

-   [Link to DGCCA Example
    File](https://github.com/shekhar-sharma/DataScience/tree/main/Groups/Group_ID_3/DeepGeneralizedCCA/dgcca_exampe.ipynb)
-   [Link to Complete GitHub
    Repository](https://github.com/shekhar-sharma/DataScience/tree/main/Groups/Group_ID_3)

**Package Name : DeepGeneralizedCCA**
=====================================

[[Source]](https://github.com/shekhar-sharma/DataScience/blob/main/Groups/Group_ID_3/DeepGeneralizedCCA/dgcca.py)

DeepGeneralizedCCA.dgcca.py

### Prerequisites

-   python 3.6+
-   pytorch 1.0+
-   NumPy 1.19+
-   [cca-zoo 1.1.4](https://pypi.org/project/cca-zoo/) (Used for the
    implementation of GCCA)

* * * * *

Classes in dgcca.py file -
==========================

Class DNN : Creates a new Deep Neural Network
---------------------------------------------

DeepGeneralizedCCA.dgcca.DNN(nn.Module) : (self, layer\_size,
activation)

### Parameters :

-   **layer\_size: list** \
     list of size of each layer in the DNN staring from the input layer
-   **activation : str, default : sigmoid** \
     The type of activation function to be used. Choose from 'relu' ,
    'tanh' , 'sigmoid' .

### Methods :

forward(self, l)

\
 forward propogates input tensor into the DNN and returns the output
tensor (overriden)\
\
**Parameters**

-   **l :**torch.Tensor (input to DNN)

**Returns -** torch.Tensor (output of DNN)

* * * * *

Class : DGCCA\_architecture : Defines the architecture for 3 DNNs
-----------------------------------------------------------------

DeepGeneralizedCCA.dgcca.DGCCA\_architecture(nn.Module) : (self,
layer\_size1, layer\_size2, layer\_size3, activation)

### Parameters :

-   **layer\_size1 : list**\
    list of sizes of each layer of first DNN from input to output layer.
-   \
     **layer\_size2 : list**\
    list of sizes of each layer of second DNN from input to output
    layer.
-   **layer\_size3 : list**\
    list of sizes of each layer of third DNN from input to output layer.

### Methods :

forward(self, x1, x2, x3)

\
 forward propogates x1 into the first DNN, x2 into the second DNN and x3
into the third DNN and returns the outputs. (overriden)\
\
**Parameters**

-   **x1 :**torch.Tensor (input to first DNN)
-   **x2 :**torch.Tensor (input to second DNN)
-   **x3 :**torch.Tensor (input to third DNN)

**Returns -** torch.Tensor,torch.Tensor,torch.Tensor (output of first,
second and third DNN)

* * * * *

Class DGCCA : Implements the DGCCA Algorithm
--------------------------------------------

DeepGeneralizedCCA.dgcca.DGCCA(nn.Module) : (self, architecture,
learning\_rate, epoch\_num, batch\_size, reg\_par, out\_size:int)

### Parameters :

-   **architecture** : DGCCA\_architecture \
     object of DGCCA\_architecture class to define structure of the 3
    DNNs.
-   **learning\_rate**: float\
    learning rate of the network
-   **epoch\_num : int**\
    How long to train the model (no of iterations to train the model)
-   **batch\_size : int**\
     Number of examples per minibatch.
-   **reg\_par : float**\
     the regularization parameter of the networks
-   **out\_size : int**\
     the size of the new space learned by the model (number of the new
    features)

### Methods :

fit\_transform(self, train\_x1, train\_x2, train\_x3, test\_x1,
test\_x2, test\_x3)

\
Learn and apply the dimension reduction on the train data batch-wise.
Trains the networks in mini-batches. Back propogates the ggca loss to
tune network acc to data. Each view needs to have the same number of
features as its corresponding view in the training data. \
\
**Parameters**

-   **train\_x1 :**torch.Tensor (Training set for first view)
-   **train\_x2 :**torch.Tensor (Training set for second view)
-   **train\_x3 :**torch.Tensor (Training set for third view)
-   **test\_x1 :**torch.Tensor (Testing set for first view)
-   **test\_x2 :**torch.Tensor (Testing set for second view)
-   **test\_x3 :**torch.Tensor (Testing set for third view)

predict(self, x1, x2, x3)

\
 returns gcca loss and output for each of the three views \
\
**Parameters**

-   **x1 :**torch.Tensor (Input for first view)
-   **x2 :**torch.Tensor (Input for second view)
-   **x3 :**torch.Tensor (Input for third view)

**Returns -** float, list (loss and list of outputs for each of the
three views) \
\

test(self, x1, x2, x3)

\
 returns gcca loss mean and output for each of the three views.\
\
**Parameters**

-   **x1 :**torch.Tensor (Input for first view)
-   **x2 :**torch.Tensor (Input for second view)
-   **x3 :**torch.Tensor (Input for third view)

**Returns**- float, list (mean of loss and list of outputs for each of
the three views) \

* * * * *

Example Using Random Data
=========================

![](example.png)

\

