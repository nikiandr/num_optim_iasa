import numpy as np

class GDOptimizer:
    def __init__(self, tol = 1e-5, max_iter = 100):
        """
        Parameters
        -----------
        max_iter  : int
                    maximum number of iterations, 
                    default is 100
        tol       : tolerance for algorithm stopping 
                    |f(x_prev) - f(x_new)| < tol
        """
        self.tol = tol
        self.max_iter = int(max_iter)
    
    def minimize(self, target_func, x0, beta = 0.1, lmb = 0.5, grad_check = False, h = 0.005):
        """
        Minimize arbitrary function with given initial point
        Parameters
        -----------
        target_func : callable
                      function to minimize
        x0          : np.array of shape (n, )
                      initial point
        beta, lmb   : float
                      parameters of step decay,
                      beta - initial step (default 0.1), 
                      lmb - decay rate (default 0.5)
        grad_check  : bool
                      whether to check ||grad||^2 < tol or not
        h           : step for computing gradients,
                      default is 0.005
        -----------
        Returns history - np.array with shape (n_iter, n+1), n - number of variables;
                history[:, -1] - values of target functions
                history[-1, :-1] - solution
                history[-1, -1] - value of target function at solution point
        """
        def compute_grad(target_func, x, h):
            n = len(x)
            grad = np.zeros_like(x).astype(float)
            for i in range(n):
                x_plus = np.copy(x)
                x_plus[i] += h
                x_minus = np.copy(x)
                x_minus[i] -= h
                grad[i] = (target_func(x_plus) - target_func(x_minus))/(2*h)
            return grad

        history = []
        x = np.copy(x0).astype(float)
        history.append([*x, target_func(x)])      
        for k in range(self.max_iter):
            grad = compute_grad(target_func, x, h)
            if grad_check and np.linalg.norm(grad)**2 < self.tol:
                break
            alpha = beta
            while target_func(x - alpha*grad) >= target_func(x):
                alpha = alpha*lmb
            x = x - alpha*grad
            history.append([*x, target_func(x)])
            if k > 0 and np.abs(history[-1][1] - history[-2][1]) < self.tol:
                break
        return np.array(history)

    def minimizeQuad(self, A, b, x0):
        """
        Minimize quadratic function (Ax, x) + (b, x)
        with given initial point
        Parameters
        -----------
        A, b    : np.array
                  parameters of target function
        x0      : np.array of shape (n, )
                  initial point
        -----------
        Returns history - np.array with shape (n_iter, n+1), n - number of variables;
                history[:, -1] - values of target functions
                history[-1, :-1] - solution
                history[-1, -1] - value of target function at solution point
        """
        target_func = lambda x: np.dot(A @ x, x) + np.dot(b, x)
        compute_grad = lambda x: 2*(A @ x) + b
        history = []
        x = np.copy(x0).astype(float)
        history.append([*x, target_func(x)])
        for k in range(self.max_iter):
            grad = compute_grad(x)
            alpha = np.dot(2*A @ x + b, grad)/(2*np.dot(A @ grad, grad))
            x = x - alpha*grad
            history.append([*x, target_func(x)])
            if k > 0 and np.abs(history[-1][1] - history[-2][1]) < self.tol:
                break
        return np.array(history)