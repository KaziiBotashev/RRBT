# -*- coding: utf-8 -*-
"""
Created on Mon May 31 17:20:12 2021

@author: vikt-
"""

from dubins import Dubins
import numpy as np
# We initialize the planner with the turn radius and the desired distance between consecutive points
local_planner = Dubins(radius=10, point_separation= 1)

# We generate two points, x, y, psi
start = (0, 0, 0) # heading east
end = (100, 100, np.pi) # heading west

# We compute the path between them
path = local_planner.dubins_path(start, end)

print(path )
import matplotlib.pyplot as plt

plt.plot(path[:, 0], path[:, 1])