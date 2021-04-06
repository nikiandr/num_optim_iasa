import numpy as np
from Optimizer import LinearizationOptimizer

#def f0(x_vect):
#    x, y = x_vect[0], x_vect[1]
#    return (x-2)**2 + (y-1)**2 

#def f1(x_vect):
#    x, y = x_vect[0], x_vect[1]
#    return -(-x**2 + y)

def f0(x_vect):
    x, y, z = x_vect[0], x_vect[1], x_vect[2]
    return x + 4*y + z

def f1(x_vect):
    x, y, z = x_vect[0], x_vect[1], x_vect[2]
    return x**2 + 3*y**2 + 2*z**2 - 1


x0 = np.random.randn(3)
hist = LinearizationOptimizer(f0, [f1], x0, eps=0.5, max_iter=1000)
print(f'argmin = {hist[-1, :-1]}, found in {len(hist)} iterations')