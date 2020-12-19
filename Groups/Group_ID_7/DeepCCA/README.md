
# **DCCA: Deep Canonical Correlation Analysis**

This is an implementation of Deep Canonical Correlation Analysis (DCCA or Deep CCA) in Python with pytorch, which supports multi-GPU training.

DCCA is a non-linear version of CCA which uses neural networks as the mapping functions instead of linear transformers. DCCA is originally proposed in the following paper:

Galen Andrew, Raman Arora, Jeff Bilmes, Karen Livescu, "[Deep Canonical Correlation Analysis.](http://www.jmlr.org/proceedings/papers/v28/andrew13.pdf)", ICML, 2013.

**Requirements**
```
PyTorch
Numpy
```
**Documentation of API**

**DeepCCAModels**

**_class _DeepCCA.DeepCCAModels.Model**(_layer_sizes1: list_, _layer_sizes2: list_, _input_size1: int_, _input_size2: int_, _outdim_size: int_, _use_all_singular_values: bool = False_, _device: torch.device = device(type='cpu')_)

Bases: torch.nn.modules.module.Module

**__init__**(_layer_sizes1: list_, _layer_sizes2: list_, _input_size1: int_, _input_size2: int_, _outdim_size: int_, _use_all_singular_values: bool = False_, _device: torch.device = device(type='cpu')_)

model initialization

**Parameters**



*   **layer_sizes1** (_list_) – list of layer shape of view 1
*   **layer_sizes2** (_list_) – list of layer shape of view 1
*   **input_size1** (_int_) – input dimension of view 1
*   **input_size2** (_int_) – input dimension of view 2
*   **outdim_size** (_int_) – output dimension of data use_all_singular_values (bool, optional): specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones. Defaults to False.
*   **device** (_torch.device, optional_) – device type GPU/CPU. Defaults to torch.device(‘cpu’).

**DeepCCA**

**_class _DeepCCA.main.DeepCCA**(_model: DeepCCA.DeepCCAModels.Model_, _outdim_size: int_, _epoch_num: int_, _batch_size: int_, _learning_rate: float_, _reg_par: float_, _linearcca: bool = False_, _device=device(type='cpu')_)

Bases: object

**__init__**(_model: DeepCCA.DeepCCAModels.Model_, _outdim_size: int_, _epoch_num: int_, _batch_size: int_, _learning_rate: float_, _reg_par: float_, _linearcca: bool = False_, _device=device(type='cpu')_)

initialize object

**Parameters**



*   **model** (_Model_) – Pytorch Model
*   **outdim_size** (_int_) – size of output dimensions
*   **epoch_num** (_int_) – Number of iterations on data
*   **batch_size** (_int_) – size of batch while training
*   **learning_rate** (_float_) – Learning rate of the model
*   **reg_par** (_float_) – regularization parameter
*   **linearcca** (_bool, optional_) – apply linear cca on model output. Defaults to False.
*   **device** (_[type], optional_) – [select device type GPU / CPU. Defaults to torch.device(‘cpu’).

**fit**(_x1: torch.Tensor_, _x2: torch.Tensor_, _vx1: torch.Tensor = None_, _vx2: torch.Tensor = None_, _tx1: torch.Tensor = None_, _tx2: torch.Tensor = None_, _checkpoint: str = 'checkpoint.model'_)

train model with the given dataset

**Parameters**



*   **x1** (_torch.Tensor_) – training data of view 1
*   **x2** (_torch.Tensor_) – training data of view 2
*   **vx1** (_torch.Tensor, optional_) – validation data of view 1. Defaults to None.
*   **vx2** (_torch.Tensor, optional_) – validation data of view 2. Defaults to None.
*   **tx1** (_torch.Tensor, optional_) – testing data of view 1. Defaults to None.
*   **tx2** (_torch.Tensor, optional_) – testing data of view 2. Defaults to None.
*   **checkpoint** (_str, optional_) – model weights saving location. Defaults to ‘checkpoint.model’.

**transform**(_x1: torch.Tensor_, _x2: torch.Tensor_, _use_linear_cca: bool = False_) → list

get output of the model

**Parameters**



*   **x1** (_torch.Tensor_) – view 1 data
*   **x2** (_torch.Tensor_) – view 2 data
*   **use_linear_cca** (_bool, optional_) – apply linear cca on model output. Defaults to False.

Returns

List containing transformed matrices .



**Example**


```
%load_ext autoreload
%autoreload 2
import torch
import torch.nn as nn
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)

from DeepCCA import DeepCCA , Model

device = torch.device('cuda')
print("Using", torch.cuda.device_count(), "GPUs")
# the path to save the final learned features
save_to = './new_features.gz'
# the size of the new space learned by the model (number of the new features)
outdim_size = 10
# size of the input for view 1 and view 2
input_shape1 = 784
input_shape2 = 784
# number of layers with nodes in each one
layer_sizes1 = [1024, 1024, 1024, outdim_size]
layer_sizes2 = [1024, 1024, 1024, outdim_size]
# the parameters for training the network
learning_rate = 1e-3
epoch_num = 1
batch_size = 800
# apply linear CCA on model output
linear_cca= False

# the regularization parameter of the network
# seems necessary to avoid the gradient exploding especially when non-saturating activations are used
reg_par = 1e-5
# specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
# if one option does not work for a network or dataset, try the other one
use_all_singular_values = False
# if a linear CCA should get applied on the learned features extracted from the networks
# it does not affect the performance on noisy MNIST significantly
apply_linear_cca = True
# end of parameters section
############

data1 = torch.randn((100,784))
data2 = torch.randn((100,784))

model = Model(layer_sizes1, layer_sizes2, input_shape1,input_shape2, outdim_size, use_all_singular_values).double()

deepcca = DeepCCA(model,  outdim_size , epoch_num , batch_size , learning_rate , reg_par , linear_cca  )

deepcca.fit(data1 , data2)
outputs = deepcca.transform(data1,data2, linear_cca)
```



