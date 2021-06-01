import matplotlib.pyplot as plt
import numpy as np
from utils import get_cov_ellipse

def plot2dcov(mu, Sigma, color='k', nSigma=1, legend=None):
    x_y_new = get_cov_ellipse(mu, Sigma[:2,:2])
    
    plt.plot(x_y_new[:, 0], x_y_new[:, 1], color=color, label=legend)
    plt.scatter(mu[0], mu[1], color=color)
    
def plot_point_on_env(env,x = 150,y = 160):
    plt.figure(figsize=(10, 8))
    plt.scatter(y, x, linewidth=1, color='red')
    plt.imshow(env)

def path_plot(env,plan,nodes, local_planner = None,name ='tests', ellipse_step = 1,ellipse_points = 30,graph  = None):
    
    plt.figure(figsize=(12, 8))

    xaxis = [x[1] for x in plan]
    yaxis = [x[0] for x in plan]
    
    if (graph is not None):
        for node in graph.nodes:
            x, y, angle = graph.nodes[node]['val']
            plt.scatter(y,x, linewidth =0.01, color='white')

    for i in range(len(plan)-1):
        if (local_planner is not None):
            path = local_planner.dubins_path(plan[i], plan[i+1])
            dt = int(len(path)/20)
            it = 0
            while (it < 20):
                x, y = path[it*dt]
                plt.scatter(y,x, linewidth =0.01, color='blue')
                it +=1
            
        if not i % ellipse_step:
            mu = np.array([xaxis[i], yaxis[i]])
            matrix = nodes[i].Sigma[:2,:2]
            xy_ellipse = get_cov_ellipse(mu, matrix, n_points = ellipse_points)
            plt.plot(xy_ellipse[:,0], xy_ellipse[:,1], linewidth=2, color='magenta')
        
    plt.plot(xaxis, yaxis, color='blue', linewidth=2)
    plt.scatter(xaxis[0],yaxis[0],  linewidth=3, color='orange')
    plt.scatter(xaxis[-1], yaxis[-1],linewidth=3, color='red')
    
    plt.imshow(env)

    plt.axis('off')
    # plt.show()
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    
def plot_all_graph_ellipses(env,graph, ell_points = 10,additional_points = []):
    plt.figure(figsize=(10, 8))
    for node in graph.nodes:
        x, y, angle = graph.nodes[node]['val']
        belief_nodes = graph.nodes[node]['belief_nodes']
        
        plt.scatter(y, x, linewidth=1, color='white')
        x_y_new = get_cov_ellipse((y,x), belief_nodes[0].Sigma[:2,:2],n_points = ell_points)
        plt.plot(x_y_new[:, 0], x_y_new[:, 1])
        
    for point in additional_points:
        plt.scatter(point[1], point[0], linewidth=1, color='red')
    plt.imshow(env)
    plt.axis('off')
    plt.show()
    
def plot_all_graph_points(env,graph, additional_points = []):
    plt.figure(figsize=(10, 8))

    for node in graph.nodes:
        x, y, angle = graph.nodes[node]['val']
        plt.scatter(y, x, linewidth=1, color='white')
    for point in additional_points:
        plt.scatter(point[1], point[0], linewidth=1, color='red')
    plt.imshow(env)

def plot_dubins_path(env,path):
    plt.figure(figsize=(12, 8))
    for state in path:
        x, y = state
        plt.scatter(y,x, linewidth =0.01, color='red')
    plt.imshow(env)
    plt.show()