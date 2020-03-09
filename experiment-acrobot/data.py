# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import gym
import scipy, scipy.misc

import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from PIL import Image
from utils import to_pickle, from_pickle

def get_theta(obs):
    '''
    In acrobot environment theta 0 is pointing down which is 
    consistent with defaults for hnn
    '''
    theta = np.arctan2(obs[1], obs[0])
    theta = theta + 2*np.pi if theta < -np.pi else theta
    theta = theta - 2*np.pi if theta > np.pi else theta
    return theta

def preproc(X, side):
    '''Crops, downsamples, desaturates, etc. the rgb pendulum observation.'''
    X = X[200:,100:400,0] - X[200:,100:400,1]
    im = Image.fromarray(X).resize((int(side), int(side)), Image.BICUBIC)
    im = np.asarray(im) / 255.
    return im

def sample_gym(seed=0, timesteps=103, trials=200, side=28, min_angle=0., max_angle=np.pi/6, 
              verbose=False, env_name='Acrobot-v1'):

    gym_settings = locals()
    if verbose:
        print("Making a dataset of acrobot pixel observations.")
        print("Edit 5/20/19: you may have to rewrite the `preproc` function depending on your screen size.")
    env = gym.make(env_name)
    env.reset()

    # the native reset function has high=0.1
    # which doesn't give sufficient dataset diversity
    def reset(env):
        high = max_angle
        env.env.state = np.random.uniform(low=-high, high=high, size=(4,))
        return env.env._get_ob()

    reset(env); env.seed(seed)

    canonical_coords, frames = [], []
    for step in range(trials*timesteps):

        if step % timesteps == 0:
            angle_ok = False

            while not angle_ok:
                env.reset()
                obs = reset(env)# env.reset()
                # only checks the first angle
                theta_init = np.abs(get_theta(obs[0:2]))
                if verbose:
                    print("\tCalled reset. Max angle= {:.3f}".format(theta_init))
                if theta_init > min_angle and theta_init < max_angle:
                    angle_ok = True
                  
            if verbose:
                print("\tRunning environment...")
                
        frames.append(preproc(env.render('rgb_array'), side))
        obs, _, _, _ = env.step(1)
        theta1, dtheta1 = get_theta(obs[0:2]), obs[-2] # theta1
        theta2, dtheta2 = get_theta(obs[2:4]), obs[-1] # theta2

        # The constant factor of 0.25 comes from saying plotting H = PE + KE*c
        # and choosing c such that total energy is as close to constant as
        # possible. It's not perfect, but the best we can do.
        canonical_coords.append( np.array([theta1, theta2, 0.25 * dtheta1, 0.25 * dtheta2]) )

    canonical_coords = np.stack(canonical_coords).reshape(trials*timesteps, -1)
    frames = np.stack(frames).reshape(trials*timesteps, -1)
    return canonical_coords, frames, gym_settings

def make_gym_dataset(test_split=0.2, **kwargs):
    '''Constructs a dataset of observations from an OpenAI Gym env'''
    canonical_coords, frames, gym_settings = sample_gym(**kwargs)
    
    coords, dcoords = [], [] # position and velocity data (canonical coordinates)
    pixels, dpixels = [], [] # position and velocity data (pixel space)
    next_pixels, next_dpixels = [], [] # (pixel space measurements, 1 timestep in future)

    trials = gym_settings['trials']
    for cc, pix in zip(np.split(canonical_coords, trials), np.split(frames, trials)):
        # calculate cc offsets
        cc = cc[1:]
        dcc = cc[1:] - cc[:-1]
        cc = cc[1:]

        # concat adjacent frames to get velocity information
        # now the pixel arrays have same information as canonical coords
        # ...but in a different (highly nonlinear) basis
        p = np.concatenate([pix[:-1], pix[1:]], axis=-1)
        
        dp = p[1:] - p[:-1]
        p = p[1:]

        # calculate the same quantities, one timestep in the future
        next_p, next_dp = p[1:], dp[1:]
        p, dp = p[:-1], dp[:-1]
        cc, dcc = cc[:-1], dcc[:-1]

        # append to lists
        coords.append(cc) ; dcoords.append(dcc)
        pixels.append(p) ; dpixels.append(dp)
        next_pixels.append(next_p) ; next_dpixels.append(next_dp)

    # concatenate across trials
    data = {'coords': coords, 'dcoords': dcoords,
            'pixels': pixels, 'dpixels': dpixels, 
            'next_pixels': next_pixels, 'next_dpixels': next_dpixels}
    data = {k: np.concatenate(v) for k, v in data.items()}

    # make a train/test split
    split_ix = int(data['coords'].shape[0]* test_split)
    split_data = {}
    for k, v in data.items():
      split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    data = split_data

    gym_settings['timesteps'] -= 3 # from all the offsets computed above
    data['meta'] = gym_settings

    return data

def get_dataset(experiment_name, save_dir, **kwargs):
  '''Returns a dataset bult on top of OpenAI Gym observations. Also constructs
  the dataset if no saved version is available.'''
  
  if experiment_name == "pendulum":
    env_name = "Pendulum-v0"
  elif experiment_name == "acrobot":
    env_name = "Acrobot-v1"
  else:
    assert experiment_name in ['pendulum']

  path = '{}/{}-pixels-dataset.pkl'.format(save_dir, experiment_name)

  try:
      data = from_pickle(path)
      print("Successfully loaded data from {}".format(path))
  except:
      print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
      data = make_gym_dataset(**kwargs)
      to_pickle(data, path)

  return data


### FOR DYNAMICS IN ANALYSIS SECTION ###
def hamiltonian_fn(coords):
  k = 1.9  # this coefficient must be fit to the data
  q, p = np.split(coords,2)
  H = k*(1-np.cos(q)) + p**2 # pendulum hamiltonian
  return H

def dynamics_fn(t, coords):
  dcoords = autograd.grad(hamiltonian_fn)(coords)
  dqdt, dpdt = np.split(dcoords,2)
  S = -np.concatenate([dpdt, -dqdt], axis=-1)
  return S

def lag_energy_fn(coords):
  k = 4.  # this coefficient must be fit to the data
  q, dq = np.split(coords,2)
  H = k*(1-np.cos(q)) + dq**2/2. # pendulum hamiltonian
  return H
