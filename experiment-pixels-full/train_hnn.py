# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import torch, argparse
from torch.utils.tensorboard import SummaryWriter

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from itertools import tee
from tqdm import tqdm

from nn_models import MLPAutoencoder, ConvAutoencoder, MLP
from hnn import HNN, PixelHNN
from data import get_dataset
from utils import L2_loss, get_model_parm_nums
from options import get_args

import matplotlib.pylab as plt


def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
  writer = SummaryWriter()

  # init model and optimizer
  if args.conv:
    autoencoder = ConvAutoencoder().to(device)
  else:
    autoencoder = MLPAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim, nonlinearity='relu').to(device)

  model = PixelHNN(args.latent_dim, args.hidden_dim,
                   autoencoder, nonlinearity=args.nonlinearity,
                   baseline=args.baseline, device=device)
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  num_parm = get_model_parm_nums(model)
  print('model contains {} parameters'.format(num_parm))

  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-5)

  # get dataset
  data = get_dataset('pendulum', args.save_dir, verbose=True, seed=args.seed, max_angle=args.max_angle, timesteps=args.traj_len, trials=args.batch_size)

  x = torch.tensor( data['pixels'], dtype=torch.float32).to(device)
  test_x = torch.tensor( data['test_pixels'], dtype=torch.float32).to(device)
  next_x = torch.tensor( data['next_pixels'], dtype=torch.float32).to(device)
  test_next_x = torch.tensor( data['test_next_pixels'], dtype=torch.float32).to(device)

  criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
  # vanilla ae train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in tqdm(range(args.total_steps+1)):
    
    # train step
    ixs = torch.randperm(x.shape[0])[:args.batch_size]
    loss = model.compute_loss(args, x[ixs], next_x[ixs], criterion, device)
    loss.backward() ; optim.step() ; optim.zero_grad()

    # stats['train_loss'].append(loss.item())
    if step % args.print_every == 0:
      # run validation
      test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
      test_loss = model.compute_loss(args, test_x[test_ixs], test_next_x[test_ixs], criterion, device)
      # stats['test_loss'].append(test_loss.item())
      
      # show latent space
      fig = plt.figure()
      k = 1000
      z = model.encode(x[:k]).detach().cpu().numpy()
      plt.scatter(z[:,0], z[:,1], c=data['coords'][:k,0], cmap=plt.cm.viridis, s=2)
        
      writer.add_scalar('Loss/train', loss.item(), step)
      writer.add_scalar('Loss/test', test_loss.item(), step)
      writer.add_figure('Image/latents', fig, step)
      # print("step {}, train_loss {:.4e}, test_loss {:.4e}"
      #   .format(step, loss.item(), test_loss.item()))

  # this stuff was done because
  # the job kept being killed for memory use
  # the generators seem to keep that from happening
  # TODO: clean
  train_ind = list(range(0, x.shape[0], args.batch_size))
  train_ind.append(x.shape[0]-1)

  train_dist1, train_dist2 = tee( model.compute_loss(args, x[i].unsqueeze(0), next_x[i].unsqueeze(0), criterion, device).detach().cpu().numpy() for i in train_ind )
  train_avg = sum(train_dist1) / x.shape[0]
  train_std = sum( (v-train_avg)**2 for v in train_dist2 ) / x.shape[0]

  test_ind = list(range(0, test_x.shape[0], args.batch_size))
  test_ind.append(test_x.shape[0]-1)

  test_dist1, test_dist2 = tee( model.compute_loss(args, test_x[i].unsqueeze(0), test_next_x[i].unsqueeze(0), criterion, device).detach().cpu().numpy() for i in test_ind )
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
    path = '{}/{}-pixels-{}-{}.tar'.format(args.save_dir, args.name, label, datestring)
    torch.save(model.state_dict(), path)
    # add path to filename for dataset
