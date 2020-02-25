# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import constants

def inertia_matrix():
    '''
    I = m * l**2
    '''
    I = constants.m * constants.l**2
    return I

def dissipative_force(dq, momentum=False):
    I = inertia_matrix()    
    if momentum:
        # variable q_dot has the value for p instead
        # q_dot = p / (m * l**2)
        dq = dq / I

    return -constants.gamma * dq

def kinetic_energy(q_dot, momentum=False):
    '''
    T = m * l**2 * q_dot**2 / 2.
    '''
    I = inertia_matrix()
    if momentum:
        # variable q_dot has the value for p instead
        # q_dot = p / (m * l**2)
        q_dot = q_dot / I

    T = I * q_dot**2 / 2.
    return T

def potential_energy(q, g=3.):
    '''
    U = m * g * l * (1 - cos(q))
    '''
    U = constants.m * constants.g * constants.l * (1 - np.cos(q))
    return U

def hamiltonian_fn(coords):
    q, p = np.split(coords,2)
    T = kinetic_energy(p, momentum=True)
    U = potential_energy(q)
    H = T + U
    return H

def dynamics_fn(t, coords):
    q, p = np.split(coords,2)
    D = dissipative_force(p, momentum=True)

    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords,2)
    dqdt -= D

    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S

# def hamiltonian_fn(coords):
#     q, p = np.split(coords,2)
#     H = 3*(1-np.cos(q)) + p**2 # pendulum hamiltonian
#     return H

# def dynamics_fn(t, coords):
#     dcoords = autograd.grad(hamiltonian_fn)(coords)
#     dqdt, dpdt = np.split(dcoords,2)
#     S = np.concatenate([dpdt, -dqdt], axis=-1)
#     return S

def get_trajectory(t_span=[0,3], timescale=15, radius=None, y0=None, noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(2)*2.-1
    if radius is None:
        radius = np.random.rand() + 1.3 # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt,2)
    
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
    return q, p, dqdt, dpdt, t_eval

def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(**kwargs)
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

def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field
