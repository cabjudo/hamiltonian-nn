# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from lagnet import Lagrangian
from lag_data import get_lag_dataset
from utils import L2_loss, rk4, get_model_parm_nums

from options import get_args

args = get_args()

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")
  S_net = MLP(int(args.input_dim/2), 140, int(args.input_dim/2)**2, args.nonlinearity)
  U_net = MLP(int(args.input_dim/2), 140, 1, args.nonlinearity)
  model = Lagrangian(int(args.input_dim/2), S_net, U_net, dt=1e-3)

  num_parm = get_model_parm_nums(model)
  print('model contains {} parameters'.format(num_parm))

  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

  # arrange data
  data = get_lag_dataset(seed=args.seed)
  x = torch.tensor( data['x'], requires_grad=False, dtype=torch.float32)
  # append zero control
  u = torch.zeros_like(x[:,0]).unsqueeze(-1)
  x = torch.cat((x, u), -1)

  test_x = torch.tensor( data['test_x'], requires_grad=False, dtype=torch.float32)
  # append zero control
  test_x = torch.cat((test_x, u), -1)

  dxdt = torch.Tensor(data['dx'])
  test_dxdt = torch.Tensor(data['test_dx'])

  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):
    
    # train step
    dq, dp, du = model.time_derivative(x).split(1,1)
    dxdt_hat = torch.cat((dq, dp), -1)
    loss = L2_loss(dxdt, dxdt_hat)
    loss.backward() ; optim.step() ; optim.zero_grad()
    
    # run test data
    dq_test, dp_test, du_test = model.time_derivative(test_x).split(1,1)
    test_dxdt_hat = torch.cat((dq_test, dp_test), -1)
    test_loss = L2_loss(test_dxdt, test_dxdt_hat)

    # logging
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    if args.verbose and step % args.print_every == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

  train_dq, train_dp, train_du = model.time_derivative(x).split(1,1)
  train_dxdt_hat = torch.cat((train_dq, train_dp), -1)
  train_dist = (dxdt - train_dxdt_hat)**2
  test_dq, test_dp, test_du = model.time_derivative(test_x).split(1,1)
  test_dxdt_hat = torch.cat((test_dq, test_dp), -1)
  test_dist = (test_dxdt - test_dxdt_hat)**2
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

  return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-baseline' if args.baseline else '-lag'
    label = '-rk4' + label if args.use_rk4 else label
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)
