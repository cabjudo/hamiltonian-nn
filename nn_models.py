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
    self.linear4a = torch.nn.Linear(hidden_dim, latent_dim)

  def encode(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = h + self.nonlinearity( self.linear2(h) )
    h = h + self.nonlinearity( self.linear3(h) )
    return self.linear4(h), self.linear4a(h)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
  
  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    x_hat = self.decode(z)

    return x_hat


# network definition: HGN 
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # The encoder network is a convolutional neural network with 8 layers, 
        # with 32 filters on the first layer, then 64 filters on each subsequent layer,
        self.conv1 = nn.Conv2d(2, 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.conv6_bn = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 64, 3)
        self.conv7_bn = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 48, 3)
        self.conv8_bn = nn.BatchNorm2d(48)
        
        # while in the last layer we have 48 filters. The final encoder transformer
        # network is a convolutional neural network with 3 layers and 64 filters on each layer. 
        self.conv9 = nn.Conv2d(48, 64, 3, stride=2) # added stride=2 because our embedding dimension is smaller
        self.conv9_bn = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64, 64, 3)
        self.conv10_bn = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(64, 2, 3)
        
        
        self.convt1 = nn.ConvTranspose2d(2, 64, 3)
        self.convt1_bn = nn.BatchNorm2d(64)
        self.convt2 = nn.ConvTranspose2d(64, 64, 3)
        self.convt2_bn = nn.BatchNorm2d(64)
        self.convt3 = nn.ConvTranspose2d(64, 48, 3)
        self.convt3_bn = nn.BatchNorm2d(48)
        
        self.convt4 = nn.ConvTranspose2d(48, 64, 2, stride=2)
        self.convt4_bn = nn.BatchNorm2d(64)
        self.convt5 = nn.ConvTranspose2d(64, 64, 3)
        self.convt5_bn = nn.BatchNorm2d(64)
        self.convt6 = nn.ConvTranspose2d(64, 64, 3)
        self.convt6_bn = nn.BatchNorm2d(64)
        self.convt7 = nn.ConvTranspose2d(64, 64, 3)
        self.convt7_bn = nn.BatchNorm2d(64)
        self.convt8 = nn.ConvTranspose2d(64, 64, 3)
        self.convt8_bn = nn.BatchNorm2d(64)
        self.convt9 = nn.ConvTranspose2d(64, 64, 3)
        self.convt9_bn = nn.BatchNorm2d(64)
        self.convt10 = nn.ConvTranspose2d(64, 64, 3)
        self.convt10_bn = nn.BatchNorm2d(64)
        self.convt11 = nn.ConvTranspose2d(64, 2, 3)

    def encode(self, x):
        # encode
        z = self.conv1(x.view(-1,2,28,28))
        z = F.relu(self.conv1_bn(z))
        # print(z.shape)
        
        z = self.conv2(z)
        z = F.relu(self.conv2_bn(z))
        # print(z.shape)
        
        z = self.conv3(z)
        z = F.relu(self.conv3_bn(z))
        # print(z.shape)
        
        z = self.conv4(z)
        z = F.relu(self.conv4_bn(z))
        # print(z.shape)
        
        z = self.conv5(z)
        z = F.relu(self.conv5_bn(z))
        # print(z.shape)
        
        z = self.conv6(z)
        z = F.relu(self.conv6_bn(z))
        # print(z.shape)
        
        z = self.conv7(z)
        z = F.relu(self.conv7_bn(z))
        # print(z.shape)
        
        z = self.conv8(z)
        z = F.relu(self.conv8_bn(z))
        # print(z.shape)
        
        z = self.conv9(z)
        z = F.relu(self.conv9_bn(z))
        # print(z.shape)
        
        z = self.conv10(z)
        z = F.relu(self.conv10_bn(z))
        # print(z.shape)
        
        z = self.conv11(z).view(-1, 2)
        # print(z.shape)
        return z

    def decode(self, z):
        # decode
        x = self.convt1(z.view(-1, 2, 1, 1))
        x = F.relu(self.convt1_bn(x))
        # print(x.shape)
        
        x = self.convt2(x)
        x = F.relu(self.convt2_bn(x))
        # print(x.shape)
        
        x = self.convt3(x)
        x = F.relu(self.convt3_bn(x))
        # print(x.shape)
        
        x = self.convt4(x)
        x = F.relu(self.convt4_bn(x))
        # print(x.shape)
        
        x = self.convt5(x)
        x = F.relu(self.convt5_bn(x))
        # print(x.shape)
        
        x = self.convt6(x)
        x = F.relu(self.convt6_bn(x))
        # print(x.shape)
        
        x = self.convt7(x)
        x = F.relu(self.convt7_bn(x))
        # print(x.shape)
        
        x = self.convt8(x)
        x = F.relu(self.convt8_bn(x))
        # print(x.shape)
        
        x = self.convt9(x)
        x = F.relu(self.convt9_bn(x))
        # print(x.shape)
        
        x = self.convt10(x)
        x = F.relu(self.convt10_bn(x))
        # print(x.shape)
        
        x = torch.tanh(self.convt11(x))
        # print(x.shape)
        x = x.view(-1, 2*28*28)
        # print(x.shape)
        
        return x

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        
        return x


class ConvVAE(ConvAutoencoder):
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
