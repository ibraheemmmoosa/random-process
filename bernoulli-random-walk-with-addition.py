import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

rng = default_rng()
# p is probability of rise
m = 100
p = 0.5
r = np.array([1.0, -1.0])
n = 100000

def func(p, r, n):
    P = rng.choice(r, size=n, p=[p, 1 - p], replace=True)
    pos = np.cumsum(P)
    return pos

poses = [func(p, r, n) for i in range(m)]
pos = sum(poses) / m

subsample_every = 100
x = list(range(0, n, subsample_every))
for i in range(m):
    plt.plot(x, poses[i][::subsample_every])

plt.plot(x, pos[::subsample_every], linewidth=2.0, c='black')
plt.show()
