import numpy as np
from Optimizer import GDOptimizer
from Plotter import PlotContour, PlotSurface

# define target function
def f(x_vect):
    x, y = x_vect[0], x_vect[1]
    return x**2 + 18*y**2 + 0.01*x*y + x - y

A = np.array([[1, 0.005], [0.005, 18]])
b = np.array([1, -1])

# set optimizer parameteres and create it
opt_params = {
    'tol': 1e-5,
    'max_iter': 100
}
opt = GDOptimizer(**opt_params)

# initial point
x0 = np.array([0, 0])
hist = opt.minimize(f, x0, beta = 1, lmb = 0.5)
#hist = opt.minimizeQuad(A, b, x0)

# save results
x_min, x_max = np.min(hist[:, 0])-0.5, np.max(hist[:, 0])+0.5
y_min, y_max = np.min(hist[:, 1])-0.5, np.max(hist[:, 1])+0.5
#PlotContour((x_min, x_max), (y_min, y_max), f)#, fname='../latex/pics/contour_init.png')
#PlotSurface((x_min, x_max), (y_min, y_max), f)#,fname='../latex/pics/surface_init.png')
PlotContour((x_min, x_max), (y_min, y_max), f, hist)#,fname='../latex/pics/contour_final.png')
#PlotSurface((x_min, x_max), (y_min, y_max), f, hist)#,fname='../latex/pics/surface_final.png')