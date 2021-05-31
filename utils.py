import numpy as np
import cv2
from typing import Union, Tuple
from scipy.linalg import cholesky

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
    
    return x_y_new #x_y_new[:, 0], x_y_new[:, 1]


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
    length = 0
    for it in range (len(states)-1):
        length += np.linalg.norm(states[it+1] - states[it])

    return length

def is_state_observable(env,state, obs_val):
#     if(env[state] == obs_val):
    if(state[0]> 230): 
        return True
    else: 
        return False

def is_point_in_collision(env, state, collision_val):
    if(env[state] == collision_val): 
        return True
    else: 
        return False

def load_data(path: str = "PS2_data.npz") -> Tuple[np.array, np.array, np.array, np.array]:
    """
    @param path: path to file with PS2 data
    @return:
        env - np.array, representing map of the environment. cells with value 0 - free space,
                                                             cells with value 255 - obstacles,
        obj - np.array, representing moving object (to be delivered from starting point to end point)
        start - np.array, starting point of moving object
        end - np.array, end position of moving object.
    """
    data = np.load(path)
    env = data["env"]
    obj = data["obj"]
    start = data["start"]
    stop = data["stop"]
    return env, obj, start, stop


def rotate_image(image: np.array, angle: Union[int, float]):
    """
    @param image: np array image (representing object)
    @param angle: angle in degrees
    @param return: np.array, 'image' rotated counter clockwise by 'angle', around image center
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def angle_difference(from_ang, to):
    delta_angle =  to - from_ang
    delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi
    return delta_angle

def get_distance(state_1,state_2, dis_weight = 1,ang_weight = 20):
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
    
def generate_cropped_images(env: np.array, obj: np.array, pose: np.array):
    """
    @param env: array representing environment
    @param obj: array representing moving object
    @param pose: position oj object in the environment
    @returns: tuple of intersected parts of env and obj,
                    and bool flag, showing if any non-empty part of object outside of env
    """
    dims = obj.shape
    dim_x = int((dims[0]) / 2)
    dim_y = int((dims[1]) / 2)

    x_min = pose[0] - dim_x
    cutted_x_min = 0
    if x_min < 0:
        cutted_x_min -= x_min
        x_min = 0

    x_max = pose[0] + dim_x
    cutted_x_max = obj.shape[0]
    if x_max > env.shape[0]:
        cutted_x_max -= x_max - (env.shape[0] - 1)
        x_max = env.shape[0] - 1

    y_min = pose[1] - dim_y
    cutted_y_min = 0
    if y_min < 0:
        cutted_y_min -= y_min
        y_min = 0

    y_max = pose[1] + dim_y
    cutted_y_max = obj.shape[1]
    if y_max > env.shape[1]:
        cutted_y_max -= y_max - (env.shape[1] - 1)
        y_max = env.shape[1] - 1

    real_object_in_outer_space = False
    if np.sum(obj) - np.sum(obj[cutted_x_min:cutted_x_max, cutted_y_min:cutted_y_max]) > 0:
        real_object_in_outer_space = True
    
    return env[x_min:x_max, y_min:y_max], obj[cutted_x_min:cutted_x_max, cutted_y_min:cutted_y_max], \
           real_object_in_outer_space


def check_collision_obj_image(env: np.array, obj: np.array, point1: np.array, point2: np.array,
                              threshold: float = 500.) -> bool:
    """
    @param env: numpy array representing an environment
    @param obj: numpy array representing an moving object
    @param point1: starting point
    @param point2: endpoint
    @return: bool, representing is point in collision or not
    """

    assert env.ndim == 2
    assert obj.ndim == 2
    assert point1.shape == point2.shape == (2,)

    # check start position
    env_cropped, obj_cropped, is_outside = generate_cropped_images(env, obj, point1)
    if is_outside:
        return True
    collision = (env_cropped + obj_cropped).max() > threshold
    if collision:
        return collision  # early exit if collision is found

    # calc num steps
    dist = np.linalg.norm(point2 - point1)
    step_dist = 2

    n_steps = int(dist / step_dist)

    # check intermediate collision
    if n_steps > 0:
        poses = np.linspace(point2, point1, num=n_steps)
        for pose in poses:
            env_cropped, obj_cropped, is_outside = generate_cropped_images(env, obj, pose.astype(int))
            if is_outside:
                return True
            collision = (env_cropped + obj_cropped).max() > threshold
            if collision:
                return collision  # early exit if collision is found

    # check end position
    env_cropped, obj_cropped, is_outside = generate_cropped_images(env, obj, point2)
    if is_outside:
        return True
    collision = (env_cropped + obj_cropped).max() > threshold
    return collision


def merge_images(img, obj, x):
    """
    @param img: original image in 2d
    @param obj, initial rocket object in 2d
    @param x, curent pose of the object
    @param angle: angle need to rotate
    @return: merged images of rotated object and passed env(img)
    """

    dims = obj.shape

    if dims[0] % 2 == 0:
        dim_x = int((dims[0]) / 2)
    else:
        dim_x = int((dims[0]) / 2) + 1
    if dims[1] % 2 == 0:
        dim_y = int((dims[1]) / 2)
    else:
        dim_y = int((dims[1]) / 2) + 1

    merged_img = np.copy(img)

    # if point close to border of env, check shapes
    x_min = int(x[0] - dim_y)
    x_min_cutted = -x_min if x_min < 0 else 0
    x_max = x_min + dims[0]  #int(x[1] + dim_y)
    x_max_cutted = obj.shape[0] - (x_max - img.shape[0]) if x_max > img.shape[0] else obj.shape[0]
    y_min = int(x[1]) - dim_x
    y_min_cutted = -y_min if y_min < 0 else 0
    y_max = y_min + dims[1]
    y_max_cutted = obj.shape[1] - (y_max - img.shape[1]) if y_max > img.shape[1] else obj.shape[1]
    
    merged_img[max(0, x_min): x_max, max(0, y_min):y_max] = obj
    merged_img = np.clip(merged_img, a_max=255, a_min=0)

    return merged_img

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_enviroment(img,obj,x):
    """
    img: original image in 2d
    obj, is the 3d array of different configurations
    x is the curent pose of the object
    """
    dims = obj.shape
    dim_x = int((dims[0]-1)/2)
    dim_y = int((dims[1]-1)/2)
    merged_img = np.copy(img)
    merged_img[ x[0]-dim_x:x[0]+dim_x+1, x[1]-dim_y:x[1]+dim_y+1 ] += obj[:,:,x[2]]*0.5
    return merged_img

def plotting_results(environment,obj,plan, weight=20, step=75):
    # plotting the result
    # ======================================
    fig = plt.figure()
    imgs = []
    for s in plan:
        
        im = merge_images(environment, obj, s[:2])
        plot = plt.imshow(im)
        imgs.append([plot])

    ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True)

    ani.save(f'rocket_solve_weight_{weight}_step_{step}.mp4')

    plt.show()



def trim_zeros(array):
    for axis in [0, 1]:
        mask = ~(array == 0).all(axis=axis)
        inv_mask = mask[::-1]
        start_idx = np.argmax(mask == True)
        end_idx = len(inv_mask) - np.argmax(inv_mask == True)
        if axis:
            array = array[start_idx:end_idx, :]
        else:
            array = array[:, start_idx:end_idx]
            
    return array

