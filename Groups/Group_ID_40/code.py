from vcca import fit
import torch
from torch.autograd import Variable

x_view = Variable(torch.randn(10000, 100))
y_view = Variable(torch.randn(10000, 100))
epochs=1
fit(x_view,y_view,10,100,epochs)
