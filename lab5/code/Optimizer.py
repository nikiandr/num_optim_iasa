import numpy as np
from GradientOptimizer import GDProjOptimizer, OrthantProjector

def gradient(target_func, x, h=0.005):
    n = len(x)
    grad = np.zeros_like(x)
    for i in range(n):
        x_plus = np.copy(x)
        x_plus[i] += h
        x_minus = np.copy(x)
        x_minus[i] -= h
        grad[i] = (target_func(x_plus) - target_func(x_minus))/(2*h)
    return grad

def QuadraticOptimizer(C, p, A, b, N_avg=20):
    """
    Solve 1/2 (C*x, x) + (p, x) -> min given A*x >= b
    (C is symmetric matrix)
    using dual problem:
        -1/2 (C*u, u) + (b, v) -> max given
        A.T*v - C*u = p
        v >= 0
    solution in u == solution in x
    N_avg is number of averaged solutions from random points.
    """
    # A.T*v - C*u = p <=> A.T*v - p = C*u
    # u = C^{-1} * (A.T*v - p)
    C_inv = np.linalg.pinv(C)
    def dual_target(v):
        u = C_inv @ (A.T @ v - p)
        return 1/2 * np.dot(C @ u, u) - np.dot(b, v)
    # explicit form of dual_target gradient
    def dual_grad(v):
        return (A.T @ v - p).T @ (C_inv @ A.T) - b
    opt = GDProjOptimizer(beta=1, lmb=0.5, tol=1e-7, max_iter=1000)
    res_v = []
    for _ in range(N_avg):
        v0 = OrthantProjector(0.01*np.random.randn(len(b)))
        v = opt.minimize(dual_target, OrthantProjector, v0, exact_grad=dual_grad)[-1, :-1]
        res_v.append(v)
    v = np.array(res_v).mean(axis=0)
    u = C_inv @ (A.T @ v - p)
    return u

def LinearizationOptimizer(f0, constraints, x0, eps, N, tol=1e-5, max_iter=100):
    def F_N(x):
        return f0(x) + N*max([0] + [f(x) for f in constraints])
    history = []
    x = np.copy(x0).astype(float)
    history.append([*x0, f0(x0)])
    for k in range(max_iter):
        df0 = gradient(f0, x)
        d_constr = []
        constr_val = []
        for i in range(len(constraints)):
            d_constr.append(gradient(constraints[i], x))
            constr_val.append(constraints[i](x))
        d_constr = np.array(d_constr)
        constr_val = np.array(constr_val)
        p = QuadraticOptimizer(np.eye(len(df0)), df0, -d_constr, constr_val)
        alpha = 1
        while F_N(x + alpha*p) > F_N(x) - eps*alpha*np.linalg.norm(p)**2:
            alpha = alpha * 0.5
        x = x + alpha*p
        history.append([*x, f0(x)])
        if k > 0 and np.abs(history[-1][1] - history[-2][1]) < tol:
                break
    return np.array(history)


def f0(x_vect):
    x, y = x_vect[0], x_vect[1]
    return (x-2)**2 + (y-1)**2 

def f1(x_vect):
    x, y = x_vect[0], x_vect[1]
    return -(-x**2 + y)

x0 = np.array([1.2, -1.3]).astype(float)
hist = LinearizationOptimizer(f0, [f1], x0, eps=0.5, N=100, max_iter=100)
print(hist[-1, :-1])