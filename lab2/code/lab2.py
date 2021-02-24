import numpy as np
from Optimizer import NewtonOptimizer
from Plotter import PlotContour, PlotSurface


# define target function
def f(x_vect):
    x, y = x_vect[0], x_vect[1]
    # return x**2 + 18*y**2 + 0.01*x*y + x - y
    return (y - x**2)**2 + (1 - x)**2


# set optimizer parameteres and create it
opt_params = {
    'beta': 1,
    'tol': 1e-5,
    'max_iter': 1000,
    'lmb': 0.5 # comment for classic
}
opt = NewtonOptimizer(**opt_params)

# initial point
x0 = np.array([10, 10], dtype=np.float64)
hist = opt.minimize(f, x0)

# save results
x_min, x_max = np.min(hist[:, 0])-0.5, np.max(hist[:, 0])+0.5
y_min, y_max = np.min(hist[:, 1])-0.5, np.max(hist[:, 1])+0.5
# PlotContour((x_min, x_max), (y_min, y_max), f,
#             fname='../latex/pics/contour_init.png')
# PlotSurface((x_min, x_max), (y_min, y_max), f,
#             fname='../latex/pics/surface_init.png')
PlotContour((x_min, x_max), (y_min, y_max), f, hist)
