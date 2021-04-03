import numpy as np
from Optimizer import GradDesc, dichotomy, golden_ratio

f = lambda x: np.exp(x) - 0.33*x**3 + 2*x

print(GradDesc(f, x0=2, beta=0.5))
print(dichotomy(f, -1, -2, 0.001))
print(golden_ratio(f, -1, -2))
