# Symplectic ODE-Net | 2019
# Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty

# code structure follows the style of HNN by Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import torch
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP, PSD
from symoden import SymODEN_R
from data import get_dataset # , arrange_data
from utils import L2_loss, get_model_parm_nums # , to_pickle
from options import get_args

args = get_args()

def train(args):
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # reproducibility: set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init model and optimizer
    if args.verbose:
        print("Start training with num of points = {} and solver {}.".format(args.num_points, args.solver))
    
    if args.structure == False and args.baseline == True:
        nn_model = MLP(args.input_dim, 600, args.input_dim, args.nonlinearity)    
        model = SymODEN_R(args.input_dim, H_net=nn_model, device=device, baseline=True)
    elif args.structure == False and args.baseline == False:
        H_net = MLP(args.input_dim, 400, 1, args.nonlinearity)
        g_net = MLP(int(args.input_dim/2), 200, int(args.input_dim/2))
        model = SymODEN_R(args.input_dim, H_net=H_net, g_net=g_net, device=device, baseline=False)
    elif args.structure == True and args.baseline ==False:
        M_net = MLP(int(args.input_dim/2), 300, int(args.input_dim/2))
        V_net = MLP(int(args.input_dim/2), 50, 1)
        g_net = MLP(int(args.input_dim/2), 200, int(args.input_dim/2))
        model = SymODEN_R(args.input_dim, M_net=M_net, V_net=V_net, g_net=g_net, device=device, baseline=False, structure=True)
    else:
        raise RuntimeError('argument *baseline* and *structure* cannot both be true')

    num_parm = get_model_parm_nums(model)
    print('model contains {} parameters'.format(num_parm))

    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    data = get_dataset(seed=args.seed)

    # modified to use the hnn stuff
    x = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32) # [1125, 2] Bx2
    # append velocity for damping
    u = x[:,1].clone().detach().unsqueeze(-1)
    x = torch.cat((x, u), -1)
    
    test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32)
    # append velocity for damping
    test_x = torch.cat((test_x, u), -1)


    dxdt = torch.Tensor(data['dx'])  # [1125, 2] Bx2
    test_dxdt = torch.Tensor(data['test_dx'])

    # training loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps+1):
        # modified to match hnn
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
    label = '-baseline_ode' if args.baseline else '-sym'
    struct = '-struct' if args.structure else ''
    rad = '-rad' if args.rad else ''
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)
    # path = '{}/{}{}{}-{}-p{}-stats{}.pkl'.format(args.save_dir, args.name, label, struct, args.solver, args.num_points, rad)
    # to_pickle(stats, path)
