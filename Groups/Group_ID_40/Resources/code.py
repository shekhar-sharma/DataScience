from vcca import fit,regenerate
import os
import torch
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from autoencoder import Autoencoder
from utils import *

x_view = Variable(torch.randn(10000, 100))
y_view = Variable(torch.randn(10000, 100))
epochs=1
fit(x_view,y_view,10,100,epochs)
