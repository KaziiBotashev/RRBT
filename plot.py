from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import numpy as np

def plot2dcov(mu, Sigma, color='k', nSigma=1, legend=None):
    """
    Plots a 2D covariance ellipse given the Gaussian distribution parameters.
    The function expects the mean and covariance matrix to ignore the theta parameter.

    :param mu: The mean of the distribution: 2x1 vector.
    :param Sigma: The covariance of the distribution: 2x2 matrix.
    :param color: The border color of the ellipse and of the major and minor axes.
    :param nSigma: The radius of the ellipse in terms of the number of standard deviations (default: 1).
    :param legend: If not None, a legend label to the ellipse will be added to the plot as an attribute.
    """
    mu = np.array(mu)
    assert mu.shape == (2,)
    Sigma = np.array(Sigma)
    assert Sigma.shape == (2, 2)

    n_points = 50

    A = cholesky(Sigma, lower=True)

    angles = np.linspace(0, 2 * np.pi, n_points)
    x_old = nSigma * np.cos(angles)
    y_old = nSigma * np.sin(angles)

    x_y_old = np.stack((x_old, y_old), 1)
    x_y_new = np.matmul(x_y_old, np.transpose(A)) + mu.reshape(1, 2) # (A*x)T = xT * AT

    plt.plot(x_y_new[:, 0], x_y_new[:, 1], color=color, label=legend)
    plt.scatter(mu[0], mu[1], color=color)
    
def plot_robot(state):
    """
    Plots a circle at the center of the robot and a line to depict the yaw.

    :param state: (x, y, theta)
    """

    assert isinstance(state, np.ndarray)
    assert state.shape == (3,)

    radius = 0.01
    robot = plt.Circle(state[:-1], radius, edgecolor='black', facecolor='cyan', alpha=0.25)
    orientation_line = np.array([[state[0], state[0] + (np.cos(state[2]) * (1000*radius * 1.5))],
                                 [state[1], state[1] + (np.sin(state[2]) * (1000*radius * 1.5))]])

    plt.gcf().gca().add_artist(robot)
    plt.plot(orientation_line[0], orientation_line[1], 'black')