# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import json
import seaborn
import pandas
import autograd
import autograd.numpy as np
import scipy.integrate
import scipy
solve_ivp = scipy.integrate.solve_ivp

import torch

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pylab as plt
from sklearn.decomposition import PCA

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from itertools import tee
from tqdm import tqdm

from nn_models import MLPAutoencoder, ConvAutoencoder, MLP_VAE
from hnn import PixelHNN
from lagnet import PixelLagrangian
from symoden import PixelSymODEN_R

from data import get_dataset as get_dataset_aux
from utils import get_model_parm_nums
from options import get_args

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = torch.log(torch.tensor(2. * np.pi))
  return torch.sum(
      -.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi),
      raxis)

def build_network(args):
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    # autoencoder
    if args.ae_network_type in ['conv']:
        autoencoder = ConvAutoencoder().to(device)
    if args.ae_network_type in ['vae']:
        autoencoder = MLP_VAE(args.ae_input_dim, 
                              args.ae_hidden_dim, 
                              args.dyn_input_dim, 
                              nonlinearity=args.ae_nonlinearity).to(device)
    if args.ae_network_type in ['mlp']:
        autoencoder = MLPAutoencoder(args.ae_input_dim, 
                                     args.ae_hidden_dim, 
                                     args.dyn_input_dim, 
                                     nonlinearity=args.ae_nonlinearity).to(device)
        
    # dynamics
    if args.dyn_network_type in ['HNN']:
        network = PixelHNN(args.dyn_input_dim, 
                           args.dyn_hidden_dim,
                           autoencoder, 
                           nonlinearity=args.dyn_nonlinearity,
                           baseline=args.baseline, device=device)
    if args.dyn_network_type in ['LagNet']:
        network = PixelLagrangian(int(args.dyn_input_dim/2), 
                                  hidden_dim=args.dyn_hidden_dim,
                                  autoencoder=autoencoder, 
                                  nonlinearity=args.dyn_nonlinearity,
                                  dt=1e-3,
                                  device=device)
    if args.dyn_network_type in ['SymODEN']:
        network = PixelSymODEN_R(int(args.dyn_input_dim/2), 
                                 autoencoder=autoencoder, 
                                 nonlinearity=args.dyn_nonlinearity,
                                 dt=1e-3, 
                                 device=device)
    return network


def build_loss_fcn(args):
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    # autoencoder
    if args.ae_loss_type in ['bce']:
        ae_loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    else: # args.ae_mse
        ae_loss_fcn = torch.nn.MSELoss(reduction='mean')
        
    # dynamics
    if args.dyn_loss_type in ['l1']:
        dyn_loss_fcn = torch.nn.L1Loss(reduction='mean')
    else: # args.dyn_mse
        dyn_loss_fcn = torch.nn.MSELoss(reduction='mean')
        
    # structure
    struct_loss_fcn_aux = torch.nn.MSELoss(reduction='mean')
        
    def structure_loss_fcn(z, z_next):
        if args.dyn_network_type in ['LagNet', 'SymODEN']:
            w, dw, _ = z.split(int(args.dyn_input_dim/2), 1)
        else:
            w, dw = z.split(int(args.dyn_input_dim/2), 1)
        w_next, _ = z_next.split(int(args.dyn_input_dim/2),1)
        return struct_loss_fcn_aux(dw, w_next - w)
        
    def loss_fcn(x, x_hat, z, z_next, z_hat_next):
        ae_loss = ae_loss_fcn(x_hat, x)
        dyn_loss = dyn_loss_fcn(z_hat_next, z_next)
        structure_loss = structure_loss_fcn(z, z_next)
        total_loss =  args.ae_weight * ae_loss + \
                      args.dyn_weight * dyn_loss + \
                      args.structure_weight * structure_loss
        return {'total': total_loss, 'ae': ae_loss, 'dyn': dyn_loss, 'structure': structure_loss}
    
    if args.dyn_network_type in ['HNN']:
        def loss_fcn_wrapper(x, x_next, model, args):
            if args.ae_network_type in ['mlp']:
                vae_loss = 0
            if args.ae_network_type in ['vae']:
                mu, logvar = model.autoencoder.encode_aux(x)
                z = model.autoencoder.reparameterize(mu, logvar)
                
                logpz = log_normal_pdf(z, torch.tensor(0.),torch.tensor(0.), raxis=1)
                print(logpz.shape)
                logqz_x = log_normal_pdf(z, mu, logvar, raxis=1)

                vae_loss = -logpz + logqz_x
                
            # encode pixel space -> latent dimension
            z = model.encode(x)
            z_next = model.encode(x_next)
            # reconstruct latent
            x_hat = model.decode(z)
            # apply dynamics
            z_hat_next = z + model.time_derivative(z) # replace with rk4
  
            losses = loss_fcn(x, x_hat, z, z_next, z_hat_next)
            losses['vae'] = vae_loss.mean() * args.vae_weight
            losses['total'] += vae_loss.mean() * args.vae_weight
  
            return losses

    if args.dyn_network_type in ['LagNet', 'SymODEN']:
        def loss_fcn_wrapper(x, x_next, model, args):
            if args.ae_network_type in ['mlp']:
                vae_loss = 0
            if args.ae_network_type in ['vae']:
                mu, logvar = model.autoencoder.encode_aux(x)
                z = model.autoencoder.reparameterize(mu, logvar)
                
                logpz = log_normal_pdf(z, torch.tensor(0.),torch.tensor(0.))
                logqz_x = log_normal_pdf(z, mu, logvar)

                vae_loss = -logpz + logqz_x
                
            # encode pixel space -> latent dimension
            z = model.encode(x)
            z_next = model.encode(x_next)
            # reconstruct latent
            x_hat = model.decode(z)
            # apply dynamics with zero control
            u = torch.zeros_like(z[:,:int(args.dyn_input_dim/2)]).reshape(-1,int(args.dyn_input_dim/2)).to(device)
            z = torch.cat((z, u), -1)
            z_hat_next = z + model.time_derivative(z) # replace with rk4
            q_next, p_next, u_next = z_hat_next.split(int(args.dyn_input_dim/2),1)
            z_hat_next = torch.cat((q_next, p_next), -1)
            
            losses = loss_fcn(x, x_hat, z, z_next, z_hat_next)
            losses['vae'] = vae_loss.mean() * args.vae_weight
            losses['total'] += vae_loss.mean() * args.vae_weight
  
            return losses # total_loss
    
    return loss_fcn_wrapper

    
def get_dataset(args):
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    # get dataset
    data = get_dataset_aux(args.env_name, 
                           args.save_dir, 
                           args.dataset_filename,
                           verbose=True, 
                           seed=args.seed,
                           timesteps=args.timesteps,
                           trials=args.trials,
                           max_angle=args.max_angle)

    x = torch.tensor( data['pixels'], dtype=torch.float32).to(device)
    test_x = torch.tensor( data['test_pixels'], dtype=torch.float32).to(device)
    next_x = torch.tensor( data['next_pixels'], dtype=torch.float32).to(device)
    test_next_x = torch.tensor( data['test_next_pixels'], dtype=torch.float32).to(device)
    
    return x, next_x, test_x, test_next_x, data['coords']


def evaluate_model(args, model):
    x, next_x, test_x, test_next_x, coords = get_dataset(args)
    loss_fcn = build_loss_fcn(args)
    
    # this stuff was done because
    # the job kept being killed for memory use
    # the generators seem to keep that from happening
    # TODO: clean
    train_ind = list(range(0, x.shape[0], args.batch_size))
    train_ind.append(x.shape[0]-1)

    train_dist1, train_dist2 = tee( loss_fcn(x[i].unsqueeze(0), next_x[i].unsqueeze(0), model, args)['total'].detach().cpu().numpy() for i in train_ind )
    train_avg = sum(train_dist1) / x.shape[0]
    train_std = sum( (v-train_avg)**2 for v in train_dist2 ) / x.shape[0]

    test_ind = list(range(0, test_x.shape[0], args.batch_size))
    test_ind.append(test_x.shape[0]-1)

    test_dist1, test_dist2 = tee( loss_fcn(test_x[i].unsqueeze(0), test_next_x[i].unsqueeze(0), model, args)['total'].detach().cpu().numpy() for i in test_ind )
    test_avg = sum(test_dist1) / test_x.shape[0]
    test_std = sum( (v-test_avg)**2 for v in test_dist2 ) / test_x.shape[0]

    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
      .format(train_avg, train_std, test_avg, test_std))


def train(args, model, loss_fcn, writer):
  device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
  writer.add_hparams(hparam_dict=args.__dict__, metric_dict={})
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=args.weight_decay)
  x, next_x, test_x, test_next_x, coords = get_dataset(args)

  for step in tqdm(range(args.total_steps + 1)):
    # train step
    ixs = torch.randperm(x.shape[0])[:args.batch_size]
    losses = loss_fcn(x[ixs], next_x[ixs], model, args)
    loss = losses['total']
    loss.backward() ; optim.step() ; optim.zero_grad()
    
    if step % args.log_train == 0: # there should be one for train test and figures
      writer.add_scalar('train_total_loss', losses['total'].item(), step)
      writer.add_scalar('train_ae_loss', losses['ae'].item(), step)
      writer.add_scalar('train_dyn_loss', losses['dyn'].item(), step)
      writer.add_scalar('train_structure_loss', losses['structure'].item(), step)
      writer.add_scalar('train_vae_loss', losses['vae'], step)
        
    if step % args.log_test == 0: # there should be one for train test and figures
      test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
      test_losses = loss_fcn(test_x[test_ixs], test_next_x[test_ixs], model, args) 
      
      writer.add_scalar('test_total_loss', test_losses['total'].item(), step)
      writer.add_scalar('test_ae_loss', test_losses['ae'].item(), step)
      writer.add_scalar('test_dyn_loss', test_losses['dyn'].item(), step)
      writer.add_scalar('test_structure_loss', test_losses['structure'].item(), step)
      writer.add_scalar('test_vae_loss', losses['vae'], step)

    if step % args.log_figure == 0: # there should be one for train test and figures
      k = 3000
      # latent space
      z = model.encode(x[:k])
      # dynamics
      if args.dyn_network_type in ['HNN']:
        z_hat_next = z + model.time_derivative(z)

      if args.dyn_network_type in ['LagNet', 'SymODEN']:
        u = torch.zeros_like(z[:,:int(args.dyn_input_dim/2)]).reshape(-1,int(args.dyn_input_dim/2)).to(device)
        z_ = torch.cat((z, u), -1)
        z_hat_next = z_ + model.time_derivative(z_) # replace with rk4
        q_next, p_next, u_next = z_hat_next.split(int(args.dyn_input_dim/2),1)
        z_hat_next = torch.cat((q_next, p_next), -1)
        
      # dimensionality reduction
      z = z.detach().cpu().numpy()
      z_hat_next = z_hat_next.detach().cpu().numpy()
    
      pca = PCA(n_components=2)
      z_2d = pca.fit_transform(z)
      z_hat_next_2d = pca.transform(z_hat_next)
        
      # seaborn pairplot figure
      pairplot_fig = seaborn.pairplot(pandas.DataFrame(z), markers="+", corner=True).fig
      writer.add_figure('pairplot', pairplot_fig, step)
      plt.close(pairplot_fig)
        
      # latent figure
      latent_fig = plt.figure()
      plt.scatter(z_2d[:,0], z_2d[:,1], c=coords[:k,0], cmap=plt.cm.viridis, s=2)
      writer.add_figure('latents', latent_fig, step)
      plt.close(latent_fig)
        
      # vector field figure, embedding vs dynamics
      latent_dynamics_fig = plt.figure()
      plt.quiver(z_2d[:-1,0], z_2d[:-1,1], z_2d[1:,0], z_2d[1:,1], color='k')
      plt.quiver(z_2d[:-1,0], z_2d[:-1,1], z_hat_next_2d[:-1,0], z_hat_next_2d[:-1,1], color='0.75')
      writer.add_figure('latent_dynamics', latent_dynamics_fig, step)
      plt.close(latent_dynamics_fig)
        
      # vector field error figure
      latent_dynamics_error_fig = plt.figure()
      latent_vec = z_2d[1:,:] - z_2d[:-1,:]
      dyn_vec = z_hat_next_2d[:-1,:] - z_2d[:-1,:]
      cos_sim = np.sum(latent_vec*dyn_vec, axis=1) # /(np.linalg.norm(latent_vec, axis=1)*np.linalg.norm(dyn_vec, axis=1))
      plt.scatter(z_2d[:-1,0], z_2d[:-1,1], c=cos_sim, cmap=plt.cm.viridis, s=2)
      writer.add_figure('latent_dynamics_error', latent_dynamics_error_fig, step)
      plt.close(latent_dynamics_error_fig)
    
      # extrapolated sequence
      k = 180
      z_hat_next = [ model.encode(x[0]).unsqueeze(0) ]
      for i in range(k-1):
        if args.dyn_network_type in ['HNN']:
            z_hat_next.append(z_hat_next[-1] + model.time_derivative(z_hat_next[-1])) 
        if args.dyn_network_type in ['LagNet', 'SymODEN']:
            u = torch.zeros_like(z_hat_next[-1][:,:int(args.dyn_input_dim/2)]).reshape(-1,int(args.dyn_input_dim/2)).to(device)
            z_ = torch.cat((z_hat_next[-1], u), -1)
            q_next, p_next, u_next = (z_ + model.time_derivative(z_)).split(int(args.dyn_input_dim/2),1)
            z_hat_next.append(torch.cat((q_next, p_next), -1))
            
      z_hat_next_array = np.asarray([ el.detach().squeeze().cpu().numpy() for el in z_hat_next ])
      # seaborn pairplot figure
      dynamics_pairplot_fig = seaborn.pairplot(pandas.DataFrame(z_hat_next_array), markers="+", corner=True).fig
      writer.add_figure('dynamics_pairplot', dynamics_pairplot_fig, step)
      plt.close(dynamics_pairplot_fig)
        
      # latent figure
      z_hat_next_2d = pca.transform(z_hat_next_array)
      dynamic_latent_fig = plt.figure()
      plt.scatter(z_hat_next_2d[:,0], z_hat_next_2d[:,1], c=coords[:k,0], cmap=plt.cm.viridis, s=2)
      writer.add_figure('dynamic latents', dynamic_latent_fig, step)
      plt.close(dynamic_latent_fig)
      
      # rendered figure
      z_hat_next = z_hat_next[0::5]
      # render
      x_hat_next = [ (model.decode(z).detach().cpu().numpy().reshape(2,28,28)[0].clip(-.5,.5) + .5) for z in z_hat_next ]
      x_hat_next_seq = [ np.concatenate(x_hat_next[(i*6):((i+1)*6)], 1) for i in range(6) ]
      x_hat_next_seq = np.concatenate(x_hat_next_seq, 0)
      # render gt
      x_next_seq = x[0:k:5, :28*28].detach().cpu().numpy().reshape(-1,28,28).transpose(0,2,1).reshape(-1,28).transpose(1,0)
      x_next_seq = [ x_next_seq[:,(i*6*28):((i+1)*6*28)] for i in range(6) ]
      x_next_seq = np.concatenate(x_next_seq, 0)
      # concatenate
      x_hat_next_seq = np.concatenate((x_hat_next_seq, x_next_seq), 1)
      # extrapolated sequence figure
      seq_fig = plt.figure()
      plt.imshow(x_hat_next_seq)
      writer.add_figure('sequence', seq_fig, step)
      plt.close(seq_fig)
      
  writer.close()

  return model


if __name__ == "__main__":
    args = get_args()
    # check if file exists
    from os import path
    
    print(args.__dict__['model_filename'])
    if path.exists('{}/{}.txt'.format(args.save_dir, args.__dict__['model_filename'])):
        exit()
    
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model = build_network(args)
    num_parm = get_model_parm_nums(model)
    if args.verbose:
        print('model contains {} parameters'.format(num_parm))
    
    writer = SummaryWriter(log_dir='runs/' + args.__dict__['model_filename'])
    loss_fcn = build_loss_fcn(args)
    
    model = train(args, model, loss_fcn, writer)
    evaluate_model(args, model)
    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    path = '{}/{}.tar'.format(args.save_dir, args.__dict__['model_filename'])
    torch.save(model.state_dict(), path)
    
    # write arguments to file <model_filename>.txt
    f = open('{}/{}.txt'.format(args.save_dir, args.__dict__['model_filename']), 'w')
    f.write(json.dumps(args.__dict__))
    f.close()

    
