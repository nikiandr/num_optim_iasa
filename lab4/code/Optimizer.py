from typing import NoReturn
import numpy as np

def chad_fibonacci(i):
    a = 1
    b = 1
    for __ in range(i):
        a, b = b, a + b
    return a


def virgin_fibonacci(i):
    left = ((1 + (5 ** 0.5))/2)**(i+1)
    right = ((1 - (5 ** 0.5))/2)**(i+1)
    return (left - right)/ (5 ** 0.5)


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


def fibonacci_method(f, a, b, n):
    # Fn >= (b - a)/tol
    '''
    n = 0
    while chad_fibonacci(n) < (b - a) / tol:
        n += 1
    '''
    x1 = a + (chad_fibonacci(n-2)/chad_fibonacci(n))*(b - a)
    x2 = a + (chad_fibonacci(n-1)/chad_fibonacci(n))*(b - a)
    for k in range(1, n - 1):
        if f(x1) > f(x2):
            a = x1
            x1 = x2
            x2 = a + (chad_fibonacci(n-k-1)/chad_fibonacci(n-k))*(b - a)
        else:
            b = x2
            x2 = x1
            x1 = a + (chad_fibonacci(n-k-2)/chad_fibonacci(n-k))*(b - a)
    delta = 1e-3 # smoll boi
    a = 0.5*(b - a) + a
    b = (0.5 + delta)*(b - a) + a
    return {'x': (a+b)/2, 'iterations': n}
