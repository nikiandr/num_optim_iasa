import numpy as np

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

def NonnegativeQuadraticOptimizer(A, b):
    """
    Solve 1/2 * (Av, v) + (b, v) -> min under v >= 0
    """
    A_plus, A_minus = np.maximum(0, A), -np.minimum(0, A)
    v = np.random.rand(len(b))
    v_temp = v.copy()
    for _ in range(100):
        Ap_v, Am_v = A_plus @ v, A_minus @ v
        for i in range(len(v)):
            if np.abs(Ap_v[i]) >= 1e-8:
                v_temp[i] = v[i] * (-b[i] + np.sqrt(b[i]**2 + 4*Ap_v[i]*Am_v[i]))/(2*Ap_v[i])
            else:
                v = np.random.rand(len(b))
                break
        if np.linalg.norm(v - v_temp) < 1e-7:
            v = v_temp.copy()
            break
        v = v_temp.copy()
    return v

def GeneralQuadraticOptimizer(p, A, b):
    """
    Solve 1/2 * ||x||^2 + (p, x) -> min under A * x >= b
    using dual problem:
        1/2 * ||u||^2 - (b, v) -> min under
        A.T * v - u = p
        v >= 0
    solution in u == solution in x
    """
    # dual target: u = A.T @ v - p, so it equals
    # 1/2*(AA.T*v, v) - (A*p + b, v) + ||p||^2 -> min, v >= 0
    v = NonnegativeQuadraticOptimizer(A @ A.T, -A @ p - b)
    u = A.T @ v - p
    return u, v

def LinearizationOptimizer(f0, constraints, x0, eps, tol=1e-5, max_iter=100):
    """
    Solve f0(x) -> min under f_i(x) <= 0 for f_i in constraints
    using linearization method
    0 < eps < 1 is method parameter
    """
    N_k = 100
    def F_N(x, N):
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
        p, v = GeneralQuadraticOptimizer(df0, -d_constr, constr_val)
        alpha = 1
        while F_N(x + alpha*p, N_k) > F_N(x, N_k) - eps*alpha*np.linalg.norm(p)**2:
            alpha = alpha * 0.5
        x = x + alpha*p
        if v.sum() > N_k:
            N_k = 2 * v.sum()
        history.append([*x, f0(x)])
        if k > 0 and np.linalg.norm(p) < tol:
                break
    return np.array(history)