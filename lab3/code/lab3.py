import numpy as np
from Optimizer import GDProjOptimizer
from Plotter import Ellipsoid, Hyperplane, make_plane, make_ellipsoid
from scipy.optimize import minimize


def f1(x_vec):
    x, y, z = x_vec[0], x_vec[1], x_vec[2]
    return x**2 + y**2 + z**2

def f9(x_vec):
    x, y, z = x_vec[0], x_vec[1], x_vec[2]
    return x + 4*y + z

def HyperplaneProj(point, p, beta):
    """
    Function for projecting argument point on X: (p, x) = beta
    Parameters:
    -----------
    point   : point, that will be projected on X
    p, beta : hyperplane parameters

    Returns: projX(point)
    """
    return point + (beta - np.dot(p, point))/(np.linalg.norm(p)**2) * p

def EllipsoidProj(point, a, b, c, alpha=100):
    """
    Function for projecting argument point on X: a*x**2 + b*y**2 + c*z**2 <= 1
    Parameters:
    -----------
    point   : point, that will be projected on X
    a, b c  : ellipsoid parameters

    Returns: projX(point)
    """
    if a*point[0]**2 + b*point[1]**2 + c*point[2]**2 <= 1:
        return point
    else:
        def penalizer(vect):
            x, y, z = vect[0], vect[1], vect[2]
            return (x-point[0])**2 + (y-point[1])**2 + (z-point[2])**2 + alpha*(a*x**2 + b*y**2 + c*z**2 - 1)**2
        proj = minimize(penalizer, x0=np.random.randn(3)).x
        return proj

# set optimizer parameteres and create it
opt_params = {
    'beta': 0.1,
    'lmb': 0.5,
    'tol': 1e-5,
    'max_iter': 100
}
opt = GDProjOptimizer(**opt_params)

hist_var1 = opt.minimize(f1,
                    lambda x: HyperplaneProj(x, np.array([1, 1, 1]), 1),
                    x0=1000*np.array([1, 1, 1]))
x_min, x_max = np.min(hist_var1[:, 0])-0.5, np.max(hist_var1[:, 0])+0.5
y_min, y_max = np.min(hist_var1[:, 1])-0.5, np.max(hist_var1[:, 1])+0.5

make_plane((x_min, x_max), (y_min, y_max), 
           target_func=lambda x, y, z: Ellipsoid(1, 1, 1, x, y, z),
           p=np.array([1, 1, 1]), beta=1, points=hist_var1,
           fname='./lab3/latex/pics/res_var1_(1000).png')

hist_var9 = opt.minimize(f9,
                    lambda x: EllipsoidProj(x, 1, 3, 2),
                    x0=1000*np.array([1, 1, 1]))
x_min, x_max = np.min(hist_var9[:, 0])-0.5, np.max(hist_var9[:, 0])+0.5
y_min, y_max = np.min(hist_var9[:, 1])-0.5, np.max(hist_var9[:, 1])+0.5


make_ellipsoid(target_func=lambda x, y, z: Hyperplane([1, 4, 1], 0, x, y, z),
              a=1, b=3, c=2, points=hist_var9,
              fname='./lab3/latex/pics/res_var9_(1000).png')