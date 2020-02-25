# LagNet: Lagrangian Neural Network
# Christine Allen-Blanchette
# adapted from:
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from utils import choose_nonlinearity

class MLP(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', bias=None):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=bias)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x, separate_fields=False):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    return self.linear3(h)


class MLPAutoencoder(torch.nn.Module):
  '''A salt-of-the-earth MLP Autoencoder + some edgy res connections'''
  def __init__(self, input_dim, hidden_dim, latent_dim, nonlinearity='tanh'):
    super(MLPAutoencoder, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

    self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
    self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear8 = torch.nn.Linear(hidden_dim, input_dim)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4, \
              self.linear5, self.linear6, self.linear7, self.linear8]:
      torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def encode(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = h + self.nonlinearity( self.linear2(h) )
    h = h + self.nonlinearity( self.linear3(h) )
    return self.linear4(h)

  def decode(self, z):
    h = self.nonlinearity( self.linear5(z) )
    h = h + self.nonlinearity( self.linear6(h) )
    h = h + self.nonlinearity( self.linear7(h) )
    return self.linear8(h)

  def forward(self, x):
    z = self.encode(x)
    x_hat = self.decode(z)
    return x_hat


class MLP_VAE(MLPAutoencoder):
  def __init__(self, input_dim, hidden_dim, latent_dim, nonlinearity='tanh'):
    super(MLP_VAE, self).__init__(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, nonlinearity=nonlinearity)

    # modify the last layer to output (mu, sigma)
    # linear4 is mu, linear5 is sigma
    # self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)
    self.linear5 = torch.nn.Linear(hidden_dim, latent_dim)

  def encode(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = h + self.nonlinearity( self.linear2(h) )
    h = h + self.nonlinearity( self.linear3(h) )
    return self.linear4(h), self.linear5(h)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
  
  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    x_hat = self.decode(z)

    return x_hat


# network definition
class ConvAutoencoderPair(nn.Module):
    def __init__(self):
        super(ConvAutoencoderPair, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, stride=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, 3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 2, 2)

        self.convt1 = nn.ConvTranspose2d(2, 64, 2)
        self.convt1_bn = nn.BatchNorm2d(64)
        self.convt2 = nn.ConvTranspose2d(64, 128, 4, stride=2)
        self.convt2_bn = nn.BatchNorm2d(128)
        self.convt3 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.convt3_bn = nn.BatchNorm2d(64)
        self.convt4 = nn.ConvTranspose2d(64, 2, 4, stride=2)

    def encode(self, x):
        # encode
        z = self.conv1(x.view(-1,2,28,28))
        z = F.relu(self.conv1_bn(z))
        z = self.conv2(z)
        z = F.relu(self.conv2_bn(z))
        z = self.conv3(z)
        z = F.relu(self.conv3_bn(z))
        z = self.conv4(z).view(-1, 2)
        return z

    def decode(self, z):
        # decode
        x = self.convt1(z.view(-1, 2, 1, 1))
        x = F.relu(self.convt1_bn(x))
        x = self.convt2(x)
        x = F.relu(self.convt2_bn(x))
        x = self.convt3(x)
        x = F.relu(self.convt3_bn(x))
        x = torch.tanh(self.convt4(x)).view(-1, 2*28*28)
        
        return x

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        
        return x


class ConvVAE(ConvAutoencoderPair):
  def __init__(self):
    super(ConvVAE, self).__init__()

    # embed to mu, sigma
    # letting self.conv4 embed to mu
    # letting self.conv5 embed to sigma
    self.conv5 = nnConv2d(64, 2, 2)

  def encode(self, x):
    z = self.conv1(x.view(-1,2,28,28))
    z = F.relu(self.conv1_bn(z))
    z = self.conv2(z)
    z = F.relu(self.conv2_bn(z))
    z = self.conv3(z)
    z = F.relu(self.conv3_bn(z))

    mu = self.conv4(z).squeeze()
    logvar = self.conv5(z).squeeze()
    return mu, logvar

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    x_hat = self.decode(z)

    return x_hat



class PSD(torch.nn.Module):
    '''A Neural Net which outputs a positive semi-definite matrix'''
    def __init__(self, input_dim, hidden_dim, diag_dim, nonlinearity='tanh'):
        super(PSD, self).__init__()
        self.diag_dim = diag_dim
        if diag_dim == 1:
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, diag_dim)

            for l in [self.linear1, self.linear2, self.linear3]:
                torch.nn.init.orthogonal_(l.weight) # use a principled initialization
            
            self.nonlinearity = choose_nonlinearity(nonlinearity)
        else:
            assert diag_dim > 1
            self.diag_dim = diag_dim
            self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear4 = torch.nn.Linear(hidden_dim, self.diag_dim + self.off_diag_dim)

            for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
                torch.nn.init.orthogonal_(l.weight) # use a principled initialization
            
            self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, q):
        if self.diag_dim == 1:
            h = self.nonlinearity( self.linear1(q) )
            h = self.nonlinearity( self.linear2(h) )
            h = self.nonlinearity( self.linear3(h) )
            return h*h + 0.1
        else:
            bs = q.shape[0]
            h = self.nonlinearity( self.linear1(q) )
            h = self.nonlinearity( self.linear2(h) )
            h = self.nonlinearity( self.linear3(h) )
            diag, off_diag = torch.split(self.linear4(h), [self.diag_dim, self.off_diag_dim], dim=1)
            # diag = torch.nn.functional.relu( self.linear4(h) )

            L = torch.diag_embed(diag)

            ind = np.tril_indices(self.diag_dim, k=-1)
            flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
            L = torch.flatten(L, start_dim=1)
            L[:, flat_ind] = off_diag
            L = torch.reshape(L, (bs, self.diag_dim, self.diag_dim))

            D = torch.bmm(L, L.permute(0, 2, 1))
            D[:, 0, 0] = D[:, 0, 0] + 0.1
            D[:, 1, 1] = D[:, 1, 1] + 0.1
            return D  
