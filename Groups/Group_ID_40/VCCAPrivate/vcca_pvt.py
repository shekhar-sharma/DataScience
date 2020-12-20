import os
import torch
import torch.utils.data
from torch import optim, cat
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from autoencoder_pvt import AutoencoderPrivate
from utils import *

class VCCAPrivate():

  def __init__(self,epochs,batch_size,ZDIMS,PDIMS,input_dim):
    self.SEED = 1
    self.BATCH_SIZE = batch_size
    self.LOG_INTERVAL = 10
    self.EPOCHS = epochs
    self.ZDIMS=ZDIMS
    self.PDIMS = PDIMS
    self.input_dim=input_dim
    torch.manual_seed(self.SEED)

  # Same for test data
  #test_loader = torch.utils.data.DataLoader(
  #    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
  #    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

  

  def loss_function_private(self,recon_x1, recon_x2, x1, x2, mu, logvar, mu1, logvar1, mu2, logvar2) -> Variable:
      # how well do input x and output recon_x agree?
      BCE1 = F.binary_cross_entropy(recon_x1, x1.view(-1, self.input_dim))
      BCE2 = F.binary_cross_entropy(recon_x2, x2.view(-1, self.input_dim))

      # KLD is Kullback–Leibler divergence -- how much does one learned
      # distribution deviate from another, in this specific case the
      # learned distribution from the unit Gaussian

      # see Appendix B from VAE paper:
      # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
      # https://arxiv.org/abs/1312.6114
      # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
      # note the negative D_{KL} in appendix B of the paper
      KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
      # Normalise by same number of elements as in reconstruction
      KLD /= self.BATCH_SIZE * self.input_dim

      KLD1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
      # Normalise by same number of elements as in reconstruction
      KLD1 /= self.BATCH_SIZE * self.input_dim

      KLD2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
      # Normalise by same number of elements as in reconstruction
      KLD2 /= self.BATCH_SIZE * self.input_dim

      # BCE tries to make our reconstruction as accurate as possible
      # KLD tries to push the distributions as close as possible to unit Gaussian
      return BCE1 + KLD + KLD1 + KLD2 + BCE2


  def train(self,epoch):
      # toggle model to train mode
      self.model.train()
      train_loss = 0
      # in the case of MNIST, len(train_loader.dataset) is 60000
      # each `data` is of BATCH_SIZE samples and has shape [128, 1, 28, 28]
      for batch_idx, (data1, data2) in enumerate(self.train_loader):
          data1 = Variable(data1).float()
          data2 = Variable(data2).float()
          self.optimizer.zero_grad()

          recon_batch1, recon_batch2, mu, log_var, mu1, log_var1, mu2, log_var2 = self.model(data1, data2)
          loss = self.loss_function_private(recon_batch1, recon_batch2, data1, data2, mu, log_var, mu1, log_var1, mu2, log_var2)
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

    self.model = AutoencoderPrivate(self.ZDIMS,self.PDIMS,self.input_dim)
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    for epoch in range(1, self.EPOCHS + 1):
        self.train(epoch)
        #est(epoch)
        self.model.eval()


  def transformPrivate(self,x_view,y_view):
    mu, log_var = self.model.encode(x_view.view(-1, self.input_dim))

    mu1, log_var1 = self.model.private_encoder1(x_view.view(-1, self.input_dim))
    mu2, log_var2 = self.model.private_encoder2(y_view.view(-1, self.input_dim))
    mu1_tmp = cat((mu,mu1), 1)
    log_var1_tmp = cat((log_var,log_var1), 1)
    mu2_tmp = cat((mu, mu2), 1)
    log_var2_tmp = cat((log_var, log_var2), 1)
    z1 = self.model.reparameterize(mu1_tmp, log_var1_tmp)
    z2 = self.model.reparameterize(mu2_tmp, log_var2_tmp)
    return z1,z2