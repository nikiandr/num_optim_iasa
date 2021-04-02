import numpy as np
from Optimizer import GradDesc

f = lambda x: np.exp(x) - 0.33*x**3 + 2*x

print(GradDesc(f, x0=2, beta=0.5))