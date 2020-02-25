import numpy as np

import torch
import torch.nn as nn
from nn_models import MLP

class Lagrangian(nn.Module):
    '''
    Learns Lagrangian of the system which is used
    to predict future states

    The momement of inertia matrix J is modelled as 
    S(q)S(q)^T + dim*I 
    where the matrix S is a neural network
    
    The potential energy is modelled as
    U(q)
    where U is a neural network
    
    The input is the current state (q, dq) 
    that is the current configuration and velocity
    The output is the update (dq, ddq)
    '''
    def __init__(self, dim=1, S_net=None, U_net=None, dt=1e-3):
        # , hidden_dim=140
        super(Lagrangian, self).__init__()
        self.dt = dt
        self.dim = dim
        self.delta_q = torch.eye(self.dim) * self.dt

        # mapping from configuration
        # to matrix used in computation of inertia matrix
        self.S = S_net # Bx1xD**2
        self.U = U_net # Bx1x1

        # initialization
        # used in HNN
        for m in self.S.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)

        for m in self.U.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)

    def J(self, q):
        '''
        Computes moment of inertia matrix
        The matrix is positive definite
        Matrix exponential is expensive
        Empirically the entries of S(q)S(q)^T are small
        Therefore,

        J = S(q)S(q)^T + dim * I

        is invertible and positive definite

        INPUTS: 
        q: state vector. 
           Bx1xD (batch, 1, dimension) 

        OUTPUT:
        J: moment of inertia matrix
           Bx1xDxD (batch, 1, dimension, dimension) 
        '''
        # Bx1xDxD
        A = self.S(q).view(-1, 1, self.dim, self.dim)
        I = torch.eye(self.dim).unsqueeze(0) * self.dim
        J = torch.einsum('blrc,blkc->brk', A, A) + I
        return J.unsqueeze(1) # Bx1xDxD

    def T(self, q, dq):
        '''
        Computes kinetic energy
        T = 1/2 * dq * J * dq

        INPUTS:
        q: state
           Bx1xD (batch, 1, dimension) 
        dq: velocity
           Bx1xD (batch, 1, dimension) 

        OUTPUTS:
        T: kinetic energy
           B (batch)
        '''
        J = self.J(q) # Bx1xDxD
        T = 0.5* torch.einsum('blrc,blc->blr', J, dq)
        T = torch.einsum('blr,blr->b', dq, T)
        return T

    def E(self, q, dq):
        '''
        Returns total energy
        E = T + U
        where T is the kinetic energy
        and U is the potential energy

        INPUTS:
        q: state
           Bx1xD (batch, 1, dimension) 
        dq: velocity
           Bx1xD (batch, 1, dimension) 

        OUTPUTS:
        E: total energy
           B (batch)
        '''
        # T has shape B
        # U has shape Bx1x1
        return self.T(q, dq) + self.U(q).squeeze()

    def coriolis(self, J, q, dq):
        '''
        Computes Coriolis terms
        Inputs:
        J: inertia matrix Bx1xDxD
        dq: velocity Bx1xD

        Outputs:
        C: coriolis terms Bx1xD
        '''
        # purturb each dimension of q for numerical gradient
        # DxBx1xDxD
        dJdqi = torch.stack([ self.J(q + dqi) - J for dqi in self.delta_q ]) / self.dt

        # dIdqi = \sum_{i,j} \frac{\partial J_kj}{\partial qi}
        #                    * dqi * dqj
        # the i-th derivative matrix is multiplied by
        # the i-th entry of dq
        dIdqi = torch.einsum('dblrc,bld->blrc', dJdqi, dq)
        # matrix vector product with dq
        dIdqi = torch.einsum('blrc,blc->blr', dIdqi, dq)

        # dIdqk = \sum_{i,j} \frac{\partial J_ij}{\partial qk}
        #                    * dqi * dqj
        dIdqk = torch.einsum('dblrc,blc->dblr', dJdqi, dq)
        dIdqk = torch.einsum('dblr,blr->bld', dIdqk, dq)

        # cent_cor = \sum_{i,j} ( \frac{\partial J_kj}{\partial qi}
        #            - 0.5 * \frac{\partial J_ij}{\partial qk})
        #            * dqi * dqj
        coriolis = dIdqi - 0.5 * dIdqk

        return coriolis

    def forward(self, y):
        q, dq = y.unsqueeze(1).split(self.dim, 2)

        return self.E(q, dq)

    def time_derivative(self, y):
        q, dq, u = y.unsqueeze(1).split(self.dim, 2)
        
        J = self.J(q)
        coriolis = self.coriolis(J, q, dq)

        # torch.inverse() works with (*, n, n) matrices where 
        # the first dimension is the batch size
        J_inv = J.inverse()

        U = self.U(q)
        # purturb each dimension of q for numerical gradient
        # Bx1xD
        dUdqi = torch.stack([ self.U(q + dqi) - U for dqi in self.delta_q ]).transpose(0,-1).squeeze(0) / self.dt

        # d/dt dL/ddq - dL/dq = \tau
        # assuming \tau = 0
        # ddq = J^{-1} (cent_cor + dUdqi)
        ddq = torch.einsum('blrc,blc->blr',-J_inv, coriolis + dUdqi)
        
        du = torch.zeros_like(ddq)
        dy = torch.cat((dq, ddq, du), -1).squeeze(1)

        return dy


class LagrangianFriction(Lagrangian):
    def __init__(self, dim=1, dt=1e-3):
        super(LagrangianFriction, self).__init__(dim=dim, dt=dt)

        self.friction = nn.Sequential(
            nn.Linear(self.dim, 1, bias=None),
        ) # Bx1x1

        # initialize weights
        for m in self.friction.modules():
            if isinstance(m, nn.Linear):
                # torch.nn.init.orthogonal_(m.weight)
                # torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
                torch.nn.init.constant_(m.weight.data, 0)

    def time_derivative(self, y):
        q, dq, u = y.unsqueeze(1).split(self.dim, 2)
        # q = y[:, :self.dim].unsqueeze(1)
        # dq = y[:, self.dim:].unsqueeze(1)

        J = self.J(q)
        coriolis = self.coriolis(J, q, dq)

        # torch.inverse() works with (*, n, n) matrices where 
        # the first dimension is the batch size
        J_inv = J.inverse()

        U = self.U(q)
        # purturb each dimension of q for numerical gradient
        # Bx1xD
        dUdqi = torch.stack([ self.U(q + dqi) - U for dqi in self.delta_q ]).transpose(0,-1).squeeze(0) / self.dt

        # Rayleigh dissipation function
        dFddq_i = torch.stack([ self.friction(dq * dqi) for dqi in self.delta_q ]).transpose(0,-1).squeeze(0) / self.dt

        # d/dt dL/ddq - dL/dq = \tau
        # assuming \tau = 0
        # ddq = J^{-1} (cent_cor + dUdqi)
        ddq = torch.einsum('blrc,blc->blr',-J_inv, coriolis + dUdqi + dFddq_i)

        du = torch.zeros_like(ddq)
        dy = torch.cat((dq, ddq, du), -1).squeeze(1)
        # dy = torch.cat((dq, ddq), -1)

        return dy


class PixelLagrangian(torch.nn.Module):
    def __init__(self, dim, hidden_dim, autoencoder, nonlinearity='tanh', dt=1e-3):
        super(PixelLagrangian, self).__init__()
        self.autoencoder = autoencoder

        S_net = MLP(dim, hidden_dim, dim**2, nonlinearity)
        U_net = MLP(dim, hidden_dim, 1, nonlinearity)
        self.lag = Lagrangian(dim, S_net, U_net, dt=1e-3)

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)

    def time_derivative(self, z):
        return self.lag.time_derivative(z)

    def forward(self, x):
        z = self.encode(x)
        z_next = z + self.time_derivative(z)
        return self.decode(z_next)
