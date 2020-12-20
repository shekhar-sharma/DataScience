from vcca import VCCA
import torch
from torch.autograd import Variable

x_view = Variable(torch.randn(10000, 100))
y_view = Variable(torch.randn(10000, 100))
epochs=1
vcca=VCCA(epochs,batch_size=120,ZDIMS=30,input_dim=100)
vcca.fit(x_view,y_view)
x,y=vcca.transform(x_view,y_view)
print(x.shape,y.shape)