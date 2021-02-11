import numpy as np
from Optimizer import GDProjOptimizer


def f(x_vec):
    x, y, z = x_vec[0], x_vec[1], x_vec[2]
    return x**2 + y**2 + z**2


def HyperplaneProj(a_vec, p, beta):
    """
    Function for projecting argument point on X: x + y + z = 1.
    Parameteres:
    -----------
    a_vec : point, that will be projected on X

    Returns: projX(a)
    """
    return a_vec + (beta - np.dot(p, a_vec))/(np.linalg.norm(p)**2) * p


# set optimizer parameteres and create it
opt_params = {
    'beta': 0.1,
    'lmb': 0.3,
    'tol': 1e-5,
    'max_iter': 100
}
opt = GDProjOptimizer(**opt_params)

# initial point
x0 = np.array([3, -6, 4])
hist = opt.minimize(f,
                    lambda x: HyperplaneProj(x, np.array([1, 1, 1]), 1),
                    x0)
print(hist)
