import numpy as np
from scipy.optimize import minimize_scalar

def QuadraticCG(A, b):
    """
    Use conjugate gradients method to minimize
    1/2 * (Ax, x) + (b, x)
    A is symmetric non-negative defined n*n matrix,
    b is n-dimensional vector
    """
    def target(x):
        return 1/2 * np.dot(A @ x, x) + np.dot(b, x)
    def grad(x):
        return A @ x + b
    x = np.random.randn(len(b))
    history = []
    history.append([*x, target(x)])
    r, h = -grad(x), -grad(x)
    for _ in range(1, len(b)+1):
        alpha = np.linalg.norm(r)**2/np.dot(A @ h, h)
        x = x + alpha * h
        history.append([*x, target(x)])
        beta = np.linalg.norm(r - alpha*(A @ h))**2/np.linalg.norm(r)**2
        r = r - alpha*(A @ h)
        h = r + beta*h
    return np.array(history)