import numpy as np

def OrthantProjector(a):
    return np.maximum(0, a)

class GDProjOptimizer:
    def __init__(self, beta=0.1, lmb=0.5, tol=1e-5, max_iter=100):
        """
        Parameteres
        -----------
        beta, lmb : float
                    parameteres of step decay,
                    beta - initial step (default 0.1),
                    lmb - decay rate (default 0.5)
        max_iter  : int
                    maximum number of iterations,
                    default is 100
        tol       : tolerance for algorithm stopping
                    |f(x_prev) - f(x_new)| < tol
        """
        self.beta = beta
        self.lmb = lmb
        self.tol = tol
        self.max_iter = int(max_iter)

    def minimize(self, target_func, proj_func, x0, exact_grad=None, h=0.005):
        """
        Parameteres
        -----------
        target_func : callable
                      function to minimize
        proj_func   : callable
                      function to calculate projection
        x0          : np.array of shape (n, )
                      initial point
        h           : step for computing gradients,
                      default is 0.005
        -----------
        Returns history - np.array with shape (n_iter, n+1),
                          n - number of variables;
                history[:, -1] - values of target functions
                history[-1, :-1] - solution
                history[-1, -1] - value of target function at solution point
        """
        if exact_grad is None:
            def compute_grad(target_func, x, h):
                n = len(x)
                grad = np.zeros_like(x)
                for i in range(n):
                    x_plus = np.copy(x)
                    x_plus[i] += h
                    x_minus = np.copy(x)
                    x_minus[i] -= h
                    grad[i] = (target_func(x_plus) - target_func(x_minus))/(2*h)
                return grad
        else:
            def compute_grad(target_func, x, h):
                return exact_grad(x)

        history = []
        x = np.copy(x0).astype(float)
        history.append([*x0, target_func(x0)])
        for k in range(self.max_iter):
            grad = compute_grad(target_func, x, h)
            alpha = self.beta
            while target_func(proj_func(x - alpha*grad)) >= target_func(x) and alpha > 1e-7:
                alpha = alpha*self.lmb
            x = proj_func(x - alpha*grad)
            history.append([*x, target_func(x)])
            if k > 0 and np.abs(history[-1][1] - history[-2][1]) < self.tol:
                break
        return np.array(history)