import numpy as np
import matplotlib.pyplot as plt

# p is probability of rise
m = 100
p = 0.5
r = np.array([1.0 + 1.1e-3, 1.0 - 1e-3])
n = 100000

def func(p, r, n):
    P = np.random.rand(n)
    P = (P < p).astype(int)
    D = 2 * P - 1

    zz = np.sum(np.convolve(D, [-1, -1], mode='valid') == 2)
    oz = np.sum(np.convolve(D, [1, -1], mode='valid') == 2)
    zo = np.sum(np.convolve(D, [-1, 1], mode='valid') == 2)
    oo = np.sum(np.convolve(D, [1, 1], mode='valid') == 2)
    tot = zz + oz + zo + oo

    pos = np.cumprod(r[P])

    return pos

poses = [func(p, r, n) for i in range(m)]
pos = sum(poses) / m

for i in range(m):
    plt.plot(poses[i])

plt.plot(pos, linewidth=2.0, c='black')
plt.show()
