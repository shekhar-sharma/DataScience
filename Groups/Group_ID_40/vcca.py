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

SEED = 1
BATCH_SIZE = 128
LOG_INTERVAL = 10
EPOCHS = 1
torch.manual_seed(SEED)

def loss_function(recon_x1, recon_x2, x1, x2, mu, logvar,input_dim) -> Variable:
    # how well do input x and output recon_x agree?
    BCE1 = F.binary_cross_entropy(recon_x1, x1.view(-1, input_dim ))
    BCE2 = F.binary_cross_entropy(recon_x2, x2.view(-1, input_dim ))

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= BATCH_SIZE * input_dim

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return BCE1 + KLD + BCE2

def train(model,epoch,train_loader,optimizer,input_dim):
    # toggle model to train mode
    model.train()
    train_loss = 0

    for batch_idx, (data1, data2) in enumerate(train_loader):      
        data1 = Variable(data1).float()
        data2 = Variable(data2).float()

        optimizer.zero_grad()

        recon_batch1, recon_batch2, mu, log_var = model(data1, data2)
            # calculate scalar loss
        loss = loss_function(recon_batch1, recon_batch2, data1, data2, mu, log_var,input_dim)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(data1)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def fit(x_view,y_view,ZDIMS,input_dim,epochs):
    EPOCHS=epochs
    data1 = x_view
    data2 = y_view

    train_loader = torch.utils.data.DataLoader(
                ConcatDataset(
                    data1,
                    data2
                ),
                batch_size=BATCH_SIZE, shuffle=True)

    model = Autoencoder(ZDIMS,input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1, EPOCHS + 1):
        train(model,epoch,train_loader,optimizer,input_dim)
        #est(epoch)
        model.eval()
        # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
        # digits in latent space
        sample = Variable(torch.randn(64, ZDIMS))

        sample1 = model.decode_1(sample).cpu()
        # print(sample1)
        sample2 = model.decode_2(sample).cpu()


