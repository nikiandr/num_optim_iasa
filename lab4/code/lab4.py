import numpy as np
from Optimizer import QuadraticCG
from Plotter import PlotContour

A = 2*np.array([[1, 0.005], [0.005, 18]])
b = np.array([1, -1])

def f(x_vect):
    x, y = x_vect[0], x_vect[1]
    x = np.array([x, y])
    return (1/2 * x.T @ A @ x + b.T @ x)[0, 0]

hist = QuadraticCG(A, b)

# pictures
x_min, x_max = np.min(hist[:, 0])-0.5, np.max(hist[:, 0])+0.5
y_min, y_max = np.min(hist[:, 1])-0.5, np.max(hist[:, 1])+0.5
PlotContour((x_min, x_max), (y_min, y_max), f, hist)