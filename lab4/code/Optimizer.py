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


def dichotomy(f, a, b, delta, tol=1e-5, max_iter=1000):
    for i in range(max_iter):
        d = (b - a)/2 * delta
        x1 = (a + b)/2 - d
        x2 = (a + b)/2 + d
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        if abs(b-a) < 2*tol:
            break
    return {'x': (a+b)/2, 'iterations': i+1}


def golden_ratio(f, a, b, tol=1e-5, max_iter=1000):
    d = 4/(5**0.5 + 1) - 1
    return dichotomy(f, a, b, d, tol, max_iter)


