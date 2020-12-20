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

class VCCA():
    def __init__(self,epochs,batch_size,ZDIMS,input_dim):
        self.SEED = 1
        self.BATCH_SIZE = batch_size
        self.LOG_INTERVAL = 10
        self.EPOCHS = epochs
        self.ZDIMS=ZDIMS
        self.input_dim=input_dim
        torch.manual_seed(self.SEED)

    def loss_function(self,recon_x1, recon_x2, x1, x2, mu, logvar) -> Variable:
        # how well do input x and output recon_x agree?
        BCE1 = F.binary_cross_entropy(recon_x1, x1.view(-1, self.input_dim ))
        BCE2 = F.binary_cross_entropy(recon_x2, x2.view(-1, self.input_dim ))

        # KLD is Kullback–Leibler divergence -- how much does one learned
        # distribution deviate from another, in this specific case the
        # learned distribution from the unit Gaussian
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # note the negative D_{KL} in appendix B of the paper
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= self.BATCH_SIZE * self.input_dim

        # BCE tries to make our reconstruction as accurate as possible
        # KLD tries to push the distributions as close as possible to unit Gaussian
        return BCE1 + KLD + BCE2

    def train(self,epoch):
        # toggle model to train mode
        self.model.train()
        train_loss = 0

        for batch_idx, (data1, data2) in enumerate(self.train_loader):      
            data1 = Variable(data1).float()
            data2 = Variable(data2).float()

            self.optimizer.zero_grad()

            recon_batch1, recon_batch2, mu, log_var = self.model(data1, data2)
                # calculate scalar loss
            loss = self.loss_function(recon_batch1, recon_batch2, data1, data2, mu, log_var)
            # calculate the gradient of the loss w.r.t. the graph leaves
            # i.e. input variables -- by the power of pytorch!
            loss.backward()
            train_loss += loss.data
            self.optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))


    def fit(self,x_view,y_view):
        data1 = x_view
        data2 = y_view

        self.train_loader = torch.utils.data.DataLoader(
                    ConcatDataset(
                        data1,
                        data2
                    ),
                    batch_size=self.BATCH_SIZE, shuffle=True)

        self.model = Autoencoder(self.ZDIMS,self.input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        for epoch in range(1, self.EPOCHS + 1):
            # train(model,epoch,train_loader,optimizer,input_dim)
            self.train(epoch)
            #est(epoch)
            self.model.eval()
            
    def transform(self,x_view,y_view):
        mu, log_var = self.model.encode(x_view.view(-1, self.input_dim))
        z1 = self.model.reparameterize(mu, log_var)
        z2 = self.model.reparameterize(mu, log_var)
        return z1,z2




