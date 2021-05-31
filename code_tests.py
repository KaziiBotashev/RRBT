# cell to test all the features of the code
from plot import *
from propagate import *
from utils import *
import networkx as nx

def test_path_vizualization(env):    
    test_plan = [[20,20,0], [100,50,0],[160,120,0]]
    test_cov_matrices = [BeliefNode(np.eye(2)*100,np.zeros((3,3)), 0),BeliefNode(np.eye(2)*500,np.zeros((3,3)),0),
                        BeliefNode(np.eye(2)*50,np.zeros((3,3)),0)]
    
    path_plot(env,test_plan,test_cov_matrices )


def check_sampling_and_steering(env,n_iter = 2, cov_matrix_diag = 200,step = 10):
    x_rng = np.random.default_rng(seed = 3)
    y_rng = np.random.default_rng(seed = 6)
    theta_rng = np.random.default_rng(seed = 10)
    x_dims = [2,env.shape[0]-2]
    y_dims = [2,env.shape[0]-2]
    new_vertex_index = -1

    G = nx.DiGraph()
    G.add_node(0, val = np.array([220.0, 50.0, 0.0]))
    for i in range(n_iter):
        if not i % 499:
            print('Iteration:',i, "vertices", len(G.nodes()))

        random_point = sample_state(x_rng, y_rng,theta_rng,x_dims,y_dims )
        nearest_index = find_nearest(G,random_point)
        nearest_state = G.nodes[nearest_index]['val']    
        vertex = steer_func(nearest_state,random_point,step)
        if (is_state_collision_free(env,vertex[:2],np.eye(2)*cov_matrix_diag)):
            new_vertex_index = len(G.nodes())
            G.add_node(new_vertex_index, val = vertex)
    plot_all_graph_points(env,G)     
    