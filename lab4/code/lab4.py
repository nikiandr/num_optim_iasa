import numpy as np

from Optimizer import GradDesc, Newton, dichotomy, golden_ratio, fibonacci_method

f = lambda x: x**2 + x
f = lambda x: (x-1)**6
f = lambda x: np.sqrt(np.abs(x+3)) - np.cos(x)**4
f = lambda x: np.exp(x) - 0.33*x**3 + 2*x

print('Gradient\t', GradDesc(f, x0=0, beta=0.5))
print('Newton\t\t', Newton(f, x0=0, beta=0.5))
print('Dichotomy\t', dichotomy(f, -3, 0, 0.001))
print('Golden ratio\t', golden_ratio(f, -3, 0))
print('Fibonacci\t', fibonacci_method(f, -3, 0, 25))