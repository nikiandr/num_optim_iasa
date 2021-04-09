import numpy as np
from Optimizer import LinearizationOptimizer

"""def f0(x_vect):
    x, y = x_vect[0], x_vect[1]
    return (x-2)**2 + (y-1)**2 

def f1(x_vect):
    x, y = x_vect[0], x_vect[1]
    return -(-x**2 + y)"""

"""def f0(x_vect):
    x, y, z = x_vect[0], x_vect[1], x_vect[2]
    return x + 4*y + z

def f1(x_vect):
    x, y, z = x_vect[0], x_vect[1], x_vect[2]
    return x**2 + 3*y**2 + 2*z**2 - 1"""

"""def f0(x_vect):
    x, y, z = x_vect[0], x_vect[1], x_vect[2]
    return x**2 + y**2 + z**2

def f1(x_vect):
    x, y, z = x_vect[0], x_vect[1], x_vect[2]
    return x + y + z - 1

def f2(x_vect):
    x, y, z = x_vect[0], x_vect[1], x_vect[2]
    return -(x + y + z - 1)"""

"""def f0(x_vect):
    x, y = x_vect[0], x_vect[1]
    return -x*y

def f1(x_vect):
    x, y = x_vect[0], x_vect[1]
    return -(x**2 + y**2)

def f2(x_vect):
    x, y = x_vect[0], x_vect[1]
    return x**2 + y**2 - 1

def f3(x_vect):
    x, y = x_vect[0], x_vect[1]
    return -x

def f4(x_vect):
    x, y = x_vect[0], x_vect[1]
    return -y"""

def f0(x_vect):
    x, y = x_vect[0], x_vect[1]
    return x**2 + y

def f1(x_vect):
    x, y = x_vect[0], x_vect[1]
    return x + y - 1

def f2(x_vect):
    x, y = x_vect[0], x_vect[1]
    return x**2 + y**2 - 9


res = []
for k in range(10):
    x0 = np.random.randn(2)
    hist = LinearizationOptimizer(f0, [f1, f2], x0, eps=0.5, max_iter=1000)
    res.append(len(hist))
    print(f'{k}\targmin = {hist[-1, :-1]}, found in {len(hist)} iterations')
print('average =', sum(res)/len(res))