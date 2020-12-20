from vcca import VCCA
import torch
from torch.autograd import Variable


x_view = Variable(torch.randn(10000, 100))
y_view = Variable(torch.randn(10000, 100))
epochs=2
vcca=VCCA(epochs,batch_size=120,ZDIMS=30,input_dim=100)
vcca.fit(x_view,y_view)
