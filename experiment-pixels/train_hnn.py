# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import torch, argparse

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from itertools import tee
from tqdm import tqdm

from nn_models import MLPAutoencoder, ConvAutoencoderPair, MLP
from hnn import HNN, PixelHNN
from data import get_dataset
from utils import L2_loss, get_model_parm_nums
from options import get_args

args = get_args()

'''The loss for this model is a bit complicated, so we'll
    define it in a separate function for clarity.'''
def pixelhnn_loss(x, x_next, model, return_scalar=True):
  # encode pixel space -> latent dimension
  z = model.encode(x)
  z_next = model.encode(x_next)

  # autoencoder loss
  x_hat = model.decode(z)
  ae_loss = ((x - x_hat)**2).mean(1)

  # hnn vector field loss
  noise = args.input_noise * torch.randn(*z.shape)
  z_hat_next = z + model.time_derivative(z + noise) # replace with rk4
  hnn_loss = ((z_next - z_hat_next)**2).mean(1)

  # canonical coordinate loss
  # -> makes latent space look like (x, v) coordinates
  w, dw = z.split(1,1)
  w_next, _ = z_next.split(1,1)
  cc_loss = ((dw-(w_next - w))**2).mean(1)

  # sum losses and take a gradient step
  loss = ae_loss + cc_loss + 1e-1 * hnn_loss
  if return_scalar:
    return loss.mean()
  return loss

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  if args.conv:
    autoencoder = ConvAutoencoderPair()
  else:
    autoencoder = MLPAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim, nonlinearity='relu')

  model = PixelHNN(args.latent_dim, args.hidden_dim,
                   autoencoder=autoencoder, nonlinearity=args.nonlinearity,
                   baseline=args.baseline)
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  num_parm = get_model_parm_nums(model)
  print('model contains {} parameters'.format(num_parm))

  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-5)

  # get dataset
  data = get_dataset('pendulum', args.save_dir, verbose=True, seed=args.seed)

  x = torch.tensor( data['pixels'], dtype=torch.float32)
  test_x = torch.tensor( data['test_pixels'], dtype=torch.float32)
  next_x = torch.tensor( data['next_pixels'], dtype=torch.float32)
  test_next_x = torch.tensor( data['test_next_pixels'], dtype=torch.float32)

  # vanilla ae train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in tqdm(range(args.total_steps+1)):
    
    # train step
    ixs = torch.randperm(x.shape[0])[:args.batch_size]
    loss = pixelhnn_loss(x[ixs], next_x[ixs], model)
    loss.backward() ; optim.step() ; optim.zero_grad()

    stats['train_loss'].append(loss.item())
    if args.verbose and step % args.print_every == 0:
      # run validation
      test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
      test_loss = pixelhnn_loss(test_x[test_ixs], test_next_x[test_ixs], model)
      stats['test_loss'].append(test_loss.item())

      print("step {}, train_loss {:.4e}, test_loss {:.4e}"
        .format(step, loss.item(), test_loss.item()))

  # this stuff was done because
  # the job kept being killed for memory use
  # the generators seem to kee that from happening
  # TODO: clean
  train_ind = list(range(0, x.shape[0], args.batch_size))
  train_ind.append(x.shape[0]-1)

  train_dist1, train_dist2 = tee( pixelhnn_loss(x[i].unsqueeze(0), next_x[i].unsqueeze(0), model).detach().numpy() for i in train_ind )
  train_avg = sum(train_dist1) / x.shape[0]
  train_std = sum( (v-train_avg)**2 for v in train_dist2 ) / x.shape[0]

  test_ind = list(range(0, test_x.shape[0], args.batch_size))
  test_ind.append(test_x.shape[0]-1)

  test_dist1, test_dist2 = tee( pixelhnn_loss(test_x[i].unsqueeze(0), test_next_x[i].unsqueeze(0), model).detach().numpy() for i in test_ind )
  test_avg = sum(test_dist1) / test_x.shape[0]
  test_std = sum( (v-test_avg)**2 for v in test_dist2 ) / test_x.shape[0]

  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_avg, train_std, test_avg, test_std))
  return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = 'baseline' if args.baseline else 'hnn'
    if args.conv: label = label + '-conv'
    path = '{}/{}-pixels-{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)
