import numpy as np

def AppendBelief(v, new_belief, eps):
    dominated = True
    for node in v.nodes:
        if np.linalg.norm(new_belief.Sigma) < np.linalg.norm(node.Sigma+np.eye(3)*eps) and
        np.linalg.norm(new_belief.Lambda) < np.linalg.norm(node.Lambda+np.eye(3)*eps) and
        new_belief.cost < node.cost:
            dominated = False
    if dominated:
        return False
    else:
        to_delete = []
        for node in v.nodes.copy():
            if np.linalg.norm(new_belief.Sigma) < np.linalg.norm(node.Sigma) and
            np.linalg.norm(new_belief.Lambda) < np.linalg.norm(node.Lambda) and
            new_belief.cost < node.cost:
                v.nodes.remove(node)
    v.nodes.append(new_belief)