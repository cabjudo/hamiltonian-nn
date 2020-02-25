import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

from data import kinetic_energy
from data import potential_energy
from data import inertia_matrix
from data import dissipative_force

import constants

def lag_energy_fn(coords):
    ''' 
    change the data to (q, dq) 
    m = l = 1, g = 3
    dq = p / (m * l**2) = p
    '''
    q, dq = np.split(coords,2)
    T = kinetic_energy(dq)
    U = potential_energy(q)
    # L(q, dq) = (m * l**2 * dq**2)/2 - m * g * l (1 - cos q)
    # H = dq**2/2. + 3*(1 - np.cos(q)) # pendulum lagrangian
    H = T + U
    return H

def lagrangian_fn(coords):
    ''' 
    change the data to (q, dq) 
    m = l = 1, g = 3
    dq = p / (m * l**2) = p
    '''
    q, dq = np.split(coords,2)
    # L(q, dq) = (m * l**2 * dq**2)/2 - m * g * l (1 - cos q)
    # L = dq**2/2. - 3*(1 - np.cos(q)) # pendulum lagrangian
    T = kinetic_energy(dq)
    U = potential_energy(q)
    L = T - U
    return L

def lag_dynamics_fn(t, coords):
    q, dq = np.split(coords,2)
    dcoords = autograd.grad(lagrangian_fn)(coords)
    dLdq, dLddq = np.split(dcoords,2)

    I = inertia_matrix()
    D = dissipative_force(dq)

    # d/dt dLddq - dLdq = \tau
    dqdt = dLddq * I
    ddqdt = (D + dLdq) * 1./I
    # dqdt, ddqdt = dLddq, dLdq # dq = p = dLddq, d/dt dLddq = dp = dLdq
    # dqdt = dq
    
    S = np.concatenate([dqdt, ddqdt], axis=-1)
    return S

def get_lag_trajectory(t_span=[0,3], timescale=15, radius=None, y0=None, noise_std=0.1, **kwargs):
    '''
    copied get_trajectory but switched lag_dynamics_fcn in place of dynamics_fcn
    p and dp are appropriately changed to dq and ddq
    '''
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(2)*2.-1
    if radius is None:
        radius = np.random.rand() + 1.3 # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

    pend_ivp = solve_ivp(fun=lag_dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, dq = pend_ivp['y'][0], pend_ivp['y'][1]
    dydt = [lag_dynamics_fn(None, y) for y in pend_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, ddqdt = np.split(dydt,2)
    
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    dq += np.random.randn(*dq.shape)*noise_std
    return q, dq, dqdt, ddqdt, t_eval

def get_lag_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    '''
    copied get_dataset but use get_lag_trajectory in place of get_trajectory
    '''
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_lag_trajectory(**kwargs)
        xs.append( np.stack( [x, y]).T )
        dxs.append( np.stack( [dx, dy]).T )
        
    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

def get_lag_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [lag_dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field

