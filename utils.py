import numpy as np
from scipy.linalg import cholesky

class BeliefNode:
    def __init__(self, Sigma, Lambda, cost, parent = None, vertex = None, edge = None):
        self.Sigma = Sigma
        self.Lambda = Lambda
        self.parent = parent
        self.cost = cost
    def __repr__(self):
        return str((self.Sigma, self.Lambda, self.cost))
    
def get_cov_ellipse(mu, Sigma, color='k', nSigma=1, n_points = 10,legend=None):
    """
    Plots a 2D covariance ellipse given the Gaussian distribution parameters.
    The function expects the mean and covariance matrix to ignore the theta parameter.

    :param mu: The mean of the distribution: 2x1 vector.
    :param Sigma: The covariance of the distribution: 2x2 matrix.
    :param color: The border color of the ellipse and of the major and minor axes.
    :param nSigma: The radius of the ellipse in terms of the number of standard deviations (default: 1).
    :param legend: If not None, a legend label to the ellipse will be added to the plot as an attribute.
    """
    # print("get_cov_ellipse n_points",n_points,"nSigma",nSigma)
    mu = np.array(mu)
    assert mu.shape == (2,)
    Sigma = np.array(Sigma)
    assert Sigma.shape == (2, 2)

    A = cholesky(Sigma, lower=True)

    angles = np.linspace(0, 2 * np.pi, n_points)
    x_old = nSigma * np.cos(angles)
    y_old = nSigma * np.sin(angles)

    x_y_old = np.stack((x_old, y_old), 1)
    x_y_new = np.matmul(x_y_old, np.transpose(A)) + mu.reshape(1, 2) # (A*x)T = xT * AT
    
    return x_y_new

def linear_path(from_ ,to , n_points = 10):
    delta = (to - from_ )/n_points
    path =  [from_]
    for i in range(n_points):
        path.append(path[-1]+ delta)
    return path

def is_state_collision_free(env,mu, Sigma,n_points = 10, nSigma = 1,):
    """
    Checks collison of the ellipse with given the Gaussian distribution parameters.
    """
    # print("is_state_col n_points",n_points,"nSigma",nSigma)
    
    x_y_new = get_cov_ellipse(mu[:2],Sigma[:2,:2], n_points = n_points)
    for xy in x_y_new:
        check_state = tuple(xy.astype(int))
        if(check_state[0] < 0): return False
        if(check_state[0] > env.shape[0]-1): return False
        if(check_state[1] < 0): return False
        if(check_state[1] > env.shape[1]-1): return False
        if (is_point_in_collision(env,check_state,7)):
            return False
    
    return True
    
def path_length(states):
    assert (states.shape[1] == 2)
    length = 0
    for it in range (len(states)-1):
        length += np.linalg.norm(states[it+1] - states[it])

    return length

def is_state_observable(env,state, obs_val):
#     if(env[state] == obs_val):
    if(state[0] > 230): 
        return True
    else: 
        return False

def is_point_in_collision(env, state, collision_val):
    if(env[state] == collision_val): 
        return True
    else: 
        return False


def angle_difference(from_ang, to):
    delta_angle =  to - from_ang
    delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi
    return delta_angle

def get_distance(state_1,state_2, dis_weight = 1,ang_weight = 5):
    d_xy = np.linalg.norm(state_1[:2] - state_2[:2])
    d_theta = np.abs(angle_difference(state_1[2],state_2[2]))
    return dis_weight * d_xy + ang_weight * d_theta
   
def find_nearest(graph, random_point):
        min_distance = 1e7
        min_index = None
        
        for index, vert in enumerate(graph.nodes):
            vert = np.array(graph.nodes[vert]['val'])
            distance = get_distance(random_point, vert)
            if distance <= min_distance:
                min_distance = distance
                min_index = index
        
        return min_index

def steer_func(near_state, final_state,max_xy_step = 20.0, max_angle_step = np.pi/6):
    dir_xy = final_state[:2] - near_state[:2]
    angle = angle_difference(near_state[2],final_state[2])
    new_state = near_state.copy()
    if (np.linalg.norm(dir_xy) < max_xy_step):
        new_state[:2] = final_state[:2]
    else: 
        new_state[:2] += dir_xy * (max_xy_step / np.linalg.norm(dir_xy))
    
    if (np.abs(angle) < max_angle_step):
        new_state[2] += angle
    else: 
        new_state[2] += np.copysign(max_angle_step,angle)
    return new_state

def sample_state(x_rng, y_rng,theta_rng, x_dims,y_dims, goal_state = None):

    key = np.random.rand(1)    
    if (goal_state is not None and key >0.9):
        return goal_state
    else:
        x = x_rng.integers(low = x_dims[0], high = x_dims[1])
        y = y_rng.integers(low = y_dims[0], high = y_dims[1])
        theta =  2*np.pi * theta_rng.random() - np.pi
        return np.array([x,y,theta])
    
