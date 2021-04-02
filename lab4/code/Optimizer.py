import numpy as np

def D(f, x, h=0.005):
    return (f(x+h) - f(x-h))/(2*h)

def GradDesc(f, x0, beta=1, lmb=0.5, tol=1e-5, max_iter=1000, h=0.005):
    x = x0
    for i in range(max_iter):
        alpha = beta
        d = D(f, x, h)
        while f(x - alpha*d) >= f(x):
            alpha = alpha*lmb
        x = x - alpha*d
        if np.abs(D(f, x, h)) < tol:
            break
    return {'x': x, 'iterations': i+1}