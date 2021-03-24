import numpy as np
from Optimizer import GDProjOptimizer, NewtonOptimizer
from Optimizer import HyperplaneProj, EllipsoidProj
from Plotter import Ellipsoid, Hyperplane, make_plane, make_ellipsoid


def f1(x_vec):
    x, y, z = x_vec[0], x_vec[1], x_vec[2]
    return x**2 + y**2 + z**2

def f9(x_vec):
    x, y, z = x_vec[0], x_vec[1], x_vec[2]
    return x + 4*y + z

# set optimizer parameteres and create it
opt_params = {
    'beta': 1,
    'lmb': 0.5,
    'tol': 1e-5,
    'max_iter': 100
}
opt = GDProjOptimizer(**opt_params)

"""hist_var1 = opt.minimize(f1,
                    lambda x: HyperplaneProj(x, np.array([1, 1, 1]), 1),
                    x0=1/3*np.array([1, 1, 1]))
x_min, x_max = np.min(hist_var1[:, 0])-0.5, np.max(hist_var1[:, 0])+0.5
y_min, y_max = np.min(hist_var1[:, 1])-0.5, np.max(hist_var1[:, 1])+0.5

make_plane((x_min, x_max), (y_min, y_max), 
           target_func=lambda x, y, z: Ellipsoid(1, 1, 1, x, y, z),
           p=np.array([1, 1, 1]), beta=1, points=hist_var1,
           fname='./lab3/latex/pics/res_var1_(argmin).png')"""

proj_opt = NewtonOptimizer(**opt_params)
hist_var9 = opt.minimize(f9,
                    lambda x: EllipsoidProj(x, 1, 3, 2, proj_opt),
                    x0=1000*np.array([1, 1, 1]))
x_min, x_max = np.min(hist_var9[:, 0])-0.5, np.max(hist_var9[:, 0])+0.5
y_min, y_max = np.min(hist_var9[:, 1])-0.5, np.max(hist_var9[:, 1])+0.5

make_ellipsoid(target_func=lambda x, y, z: Hyperplane([1, 4, 1], 0, x, y, z),
              a=1, b=3, c=2, points=hist_var9,
              fname='./lab3/latex/pics/res_var9_(1000).png')