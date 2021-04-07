import numpy as np
from scipy.optimize import minimize_scalar

def dichotomy(f, a, b, delta, tol=1e-5, max_iter=100):
    for _ in range(max_iter):
        d = (b - a)/2 * delta
        x1 = (a + b)/2 - d
        x2 = (a + b)/2 + d
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        if abs(b-a) < 2*tol:
            break
    x = (a+b)/2
    return x


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


def ConjugateGradient(target_func, x0, renewal=True, grad_check=False, d=0.005, tol=1e-5, max_iter=100):
    """
    Minimize arbitrary function using conjugate gradients method
    Parameters
    -----------
    target_func : callable
                  function to minimize
    x0          : np.array of shape (n, )
                  initial point
    renewal     : bool
                  whether to make parameters renewal or not
                  default is True 
    grad_check  : bool
                  whether to check ||grad||^2 < tol or not
    d           : step for computing gradients,
                  default is 0.005
    max_iter    : int
                  maximum number of iterations, 
                  default is 100
    tol         : tolerance for algorithm stopping 
                  |f(x_prev) - f(x_new)| < tol
    -----------
    Returns history - np.array with shape (n_iter, n+1), n - number of variables;
            history[:, -1] - values of target functions
            history[-1, :-1] - solution
            history[-1, -1] - value of target function at solution point
    """
    def compute_grad(target_func, x, d):
            n = len(x)
            grad = np.zeros_like(x)
            for i in range(n):
                x_plus = np.copy(x)
                x_plus[i] += d
                x_minus = np.copy(x)
                x_minus[i] -= d
                grad[i] = (target_func(x_plus) - target_func(x_minus))/(2*d)
            return grad
    
    history = []
    x = x0.copy().astype('float')
    history.append([*x, target_func(x)])
    
    grad = compute_grad(target_func, x, d)
    if grad_check and np.linalg.norm(grad)**2 < tol:
        return np.array(history)
    h = -grad
    for k in range(max_iter):
        old_grad = grad
        
        alpha = dichotomy(lambda a: target_func(x + a*h), 0, 1, 0.001)
        x = x + alpha*h
        history.append([*x, target_func(x)])
        
        grad = compute_grad(target_func, x, d)
        if renewal and not k % 2:
            beta = 0
        else:
            beta = (grad @ (grad - old_grad)) / np.linalg.norm(old_grad)**2
        h = -grad + beta * h
        
        if k > 0 and np.abs(history[-1][1] - history[-2][1]) < tol:
            break
            
    return np.array(history)
        