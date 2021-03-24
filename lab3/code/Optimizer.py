import numpy as np
import itertools

def HyperplaneProj(point, p, beta):
    """
    Function for projecting argument point on X: (p, x) = beta
    Parameters:
    -----------
    point   : point, that will be projected on X
    p, beta : hyperplane parameters

    Returns: projX(point)
    """
    return point + (beta - np.dot(p, point))/(np.linalg.norm(p)**2) * p

def EllipsoidUniform(a, b, c, seed=42):
    """
    Select point uniformly on ellipsoid a*x**2 + b*y**2 + c*z**2 = 1
    """
    np.random.seed(seed)
    theta = 2*np.pi*np.random.rand(1)
    phi = np.arccos(2*np.random.rand(1) - 1)
    x0 = np.array([1/np.sqrt(a) * np.cos(theta) * np.sin(phi),
                   1/np.sqrt(b) * np.sin(theta) * np.sin(phi),
                   1/np.sqrt(c) * np.cos(phi)]) # select initial point uniformly on Ellipsoid
    np.random.seed(None)
    return x0

def EllipsoidProj(point, a, b, c, opt, alpha=100, seed=42):
    """
    Function for projecting argument point on X: a*x**2 + b*y**2 + c*z**2 <= 1
    using penalty function method
    Parameters:
    -----------
    point       : point, that will be projected on X
    a, b, c     : ellipsoid parameters
    opt         : optimizer, callable

    Returns: projX(point)
    """
    if a*point[0]**2 + b*point[1]**2 + c*point[2]**2 <= 1:
        return point
    else:
        x0 = EllipsoidUniform(a, b, c, seed)
        def penalizer(vect):
            x, y, z = vect[0], vect[1], vect[2]
            return (x-point[0])**2 + (y-point[1])**2 + (z-point[2])**2 + alpha*(a*x**2 + b*y**2 + c*z**2 - 1)**2
        proj = opt.minimize(penalizer, x0)[-1, :-1]
        return proj.flatten().tolist()

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

    def minimize(self, target_func, proj_func, x0, h=0.005):
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

class NewtonOptimizer:
    def __init__(self, beta=1, lmb=None, tol=1e-5, max_iter=100):
        """
        Parameteres
        -----------
        beta      : float
                    initial step (default 1)
        lmb       : float or None
                    step decay parameter
                    if None - classic Newton method is used instead
        max_iter  : int
                    maximum number of iterations,
                    default is 100
        tol       : tolerance for algorithm stopping
                    |f(x_prev) - f(x_new)| < tol
        """
        self.beta = beta
        self.tol = tol
        self.max_iter = int(max_iter)
        self.lmb = lmb

    def minimize(self, target_func, x0, h=0.005):
        """
        Parameteres
        -----------
        target_func : callable
                      function to minimize
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

        def compute_hessian(target_func, x, h):
            n = len(x)
            hessian = np.empty((n, n))
            for k in range(n):
                for m in range(k+1):
                    dx = np.zeros_like(x)
                    dx[k] = h/2
                    dy = np.zeros_like(x)
                    dy[m] = h/2
                    hessian[k, m] = sum([i*j * target_func(x + i*dx + j*dy)
                                        for i, j in itertools.product([1, -1],
                                                                      repeat=2)
                                        ]) / h**2
                    hessian[m, k] = hessian[k, m]
            return hessian

        history = []
        x = np.copy(x0)
        history.append([*x0, target_func(x0)])
        for k in range(self.max_iter):
            grad = compute_grad(target_func, x, h)
            hessian = compute_hessian(target_func, x, h)
            step = np.linalg.pinv(hessian) @ grad

            alpha = self.beta
            if self.lmb is not None:
                while target_func(x - alpha*step) > target_func(x):  # > !!!
                    alpha = alpha * self.lmb
                    
            x = x - alpha * step
            history.append([*x, target_func(x)])
            if k > 0 and np.abs(history[-1][1] - history[-2][1]) < self.tol:
                break
        return np.array(history)