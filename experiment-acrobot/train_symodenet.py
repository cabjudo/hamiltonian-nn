# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import torch

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLPAutoencoder, MLP
from symoden import PixelSymODEN_R
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

  u = torch.zeros_like(z[:,:2])
  z_noise = torch.cat((z + noise, u), -1)
  z = torch.cat((z, u), -1)

  z_hat_next = z + model.time_derivative(z_noise) # replace with rk4
  q_next, p_next, u_next = z_hat_next.split(2,1)
  z_hat_next = torch.cat((q_next, p_next), -1)
  hnn_loss = ((z_next - z_hat_next)**2).mean(1)

  # sum losses and take a gradient step
  loss = ae_loss + 1e-1 * hnn_loss
  if return_scalar:
    return loss.mean()
  return loss

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  autoencoder = MLPAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim,
                               nonlinearity='relu')
  model = PixelSymODEN_R(int(args.latent_dim/2), 
                         autoencoder=autoencoder, 
                         nonlinearity=args.nonlinearity,
                         dt=1e-3)
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-5)

  # get dataset
  data = get_dataset('acrobot', args.save_dir, verbose=True, seed=args.seed)

  x = torch.tensor( data['pixels'], dtype=torch.float32)
  test_x = torch.tensor( data['test_pixels'], dtype=torch.float32)
  next_x = torch.tensor( data['next_pixels'], dtype=torch.float32)
  test_next_x = torch.tensor( data['test_next_pixels'], dtype=torch.float32)

  # vanilla ae train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):
    
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

  train_dist = pixelhnn_loss(x, next_x, model, return_scalar=False)
  test_dist = pixelhnn_loss(test_x, test_next_x, model, return_scalar=False)
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))
  return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = 'baseline' if args.baseline else 'sym'
    path = '{}/{}-pixels-{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)
