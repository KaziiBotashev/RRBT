import numpy as np

class DynamicModel:
    def __init__(self, q_move, q_angle, r, dt, k = 0.1):
        self.Q = np.eye(3)*q_move
        self.Q[2,2] = q_angle
        self.R = np.eye(3)*r
        self.dt = dt
        self.A = np.eye(3) 
        self.C = np.eye(3)
        self.K = np.array([[1,1,0],
                           [0,0,1]])*k
    
    def B(self, state):
        return np.array([[self.dt*np.cos(state[2]), 0],
                         [self.dt*np.sin(state[2]), 0],
                         [0,               self.dt]])
        
    
    def predict(self, state, action, Sigma_state):
        self.state = state
        self.x_predicted = self.A@state+self.B(state)@action
        self.Sigma_predicted = self.A@Sigma_state@self.A.T+self.Q
        
    def update(self, Lambda_prev):
        self.z = np.random.multivariate_normal([0,0,0], self.R)
        self.S = self.C@self.Sigma_predicted@self.C.T + self.R
        self.L = self.Sigma_predicted@self.C.T@np.linalg.inv(self.S)
        z = np.random.multivariate_normal([0,0,0], self.R)
        self.x_updated = self.x_predicted + self.L@(z-self.C@self.x_predicted)
        self.Sigma_updated = self.Sigma_predicted - self.L@self.C@self.Sigma_predicted
        term = self.A-self.B(self.state)@self.K
        self.Lambda = term@Lambda_prev@term.T + self.L@self.C@self.Sigma_predicted
        
class Edge:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

class BeliefNode:
    def __init__(self, Sigma, Lambda, cost, parent = None, vertex = None, edge = None):
        self.Sigma = Sigma
        self.Lambda = Lambda
        self.parent = parent
        self.cost = cost
    def __repr__(self):
        return str((self.Sigma, self.Lambda, self.cost))