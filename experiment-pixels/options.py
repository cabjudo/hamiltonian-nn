import argparse
import numpy as np
from datetime import datetime
import hashlib
import json

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

def build_dataset_filename(args):
    # for the standard pendulum dataset; this also works for full(?)
    # <dataset name>_S<timesteps>_T<trials>_MA<max angle (replace '.' with '-')>
    max_angle_str = '{:.2f}'.format(args.max_angle).replace('.','-')
    return args.env_name + '_S' + str(args.timesteps) + '_T' + str(args.trials) + '_MA' + max_angle_str

def build_model_filename(args):
    # <dynamics type>_<dataset name>_<date string>
    # datetime_string = datetime.now().isoformat().replace('-','').replace(':','').replace('.','')
    # return args.dyn_network_type + '_' + build_dataset_filename(args) + '_' + datetime_string
    return args.dyn_network_type + args.ae_network_type + str(args.dyn_input_dim) + str(args.max_angle).replace('.','-')[:5] + args.env_name[:4] + '_' + hashlib.sha1(json.dumps(args.__dict__).encode('utf-8')).hexdigest()[:10]

def get_args():
    parser = argparse.ArgumentParser(description=None)
    # dynamics network architecture
    dyn_network_choices = ['HNN','LagNet', 'SymODEN']
    parser.add_argument('--dyn_network_type', default='HNN', choices=dyn_network_choices)
    parser.add_argument('--dyn_input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--dyn_hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    dyn_nonlinearity_choices = ['tanh', 'sigmoid']
    parser.add_argument('--dyn_nonlinearity', default='tanh', choices=dyn_nonlinearity_choices, help='neural net nonlinearity')
    
    # autoencoding network architecture
    parser.add_argument('--ae_input_dim', default=2*28**2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--ae_hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    # ae_latent_dim should equal dyn_input_dim
    parser.add_argument('--ae_latent_dim', default=2, type=int, help='latent dimension of autoencoder')
    ae_nonlinearity_choices = ['tanh', 'sigmoid', 'relu']
    parser.add_argument('--ae_nonlinearity', default='relu', choices=ae_nonlinearity_choices, help='neural net nonlinearity')
    ae_network_choices = ['conv', 'vae', 'mlp']
    parser.add_argument('--ae_network_type', default='mlp', choices=ae_network_choices, help='use CNN for autoencoder')
    
    # training parameters
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=200, type=int, help='batch size')
    parser.add_argument('--total_steps', default=10000, type=int, help='number of gradient steps')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--gpu', type=int, default=1)
    
    # loss function
    # autoencoder loss
    ae_loss_choices = ['bce', 'mse']
    parser.add_argument('--ae_loss_type', default='bce', choices=ae_loss_choices)
    parser.add_argument('--ae_weight', default=1., type=float)
    parser.add_argument('--vae_weight', default=1., type=float)
    # dynamics loss
    dyn_loss_choices = ['l1', 'mse']
    parser.add_argument('--dyn_loss_type', default='l1', choices=dyn_loss_choices)
    parser.add_argument('--dyn_weight', default=1e-1, type=float)
    # dynamics structure
    parser.add_argument('--structure_weight', default=1., type=float, help='encourage latent to look like \dot{q}')
    
    # dataset
    parser.add_argument('--env_name', default='pendulum')
    parser.add_argument('--timesteps', type=int, default=103)
    # should trials equal batch size???
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--max_angle', type=float, default=np.pi/6.)
    
    # debugging options
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--log_train', default=500, type=int, help='number of gradient steps between logs')
    parser.add_argument('--log_test', default=500, type=int, help='number of gradient steps between logs')
    parser.add_argument('--log_figure', default=500, type=int, help='number of gradient steps between logs')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    
    # save parameters
    parser.add_argument('--name', default='pixels', type=str, help='either "real" or "sim" data')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    
    # construct parameters
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    
    # generate filenames
    args.__dict__['dataset_filename'] = build_dataset_filename(args)
    args.__dict__['model_filename'] = build_model_filename(args)
    
    # resolve parameter conflicts
    # if dyn_network_type is LagNet, structure_weight = 0
    if args.dyn_network_type in ['LagNet']:
        args.structure_weight = 0.0
    
    return args

