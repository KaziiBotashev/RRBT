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