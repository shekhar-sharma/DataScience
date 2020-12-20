from vcca_pvt import VCCAPrivate
import torch
from torch.autograd import Variable


x_view = Variable(torch.randn(10000, 100))
y_view = Variable(torch.randn(10000, 100))
epochs=1
vccaPvt=VCCAPrivate(epochs,batch_size=120,ZDIMS=30,PDIMS=30,input_dim=100)
vccaPvt.fit(x_view,y_view)
x_latent,y_latent = vccaPvt.transformPrivate(x_view,y_view)
print(f"x_latent shape = {x_latent.shape},y_latent shape = {y_latent.shape}")

