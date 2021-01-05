import numpy as np
import matplotlib.pyplot as plt

# p is probability of rise
p = 0.5
r = np.array([1.0 + 1.1e-3, 1.0 - 1e-3])
n = 1000

def func(p, r, n):
    P = np.random.rand(n)
    P = (P < p).astype(int)
    D = 2 * P - 1

    zz = np.sum(np.convolve(D, [-1, -1], mode='valid') == 2)
    oz = np.sum(np.convolve(D, [1, -1], mode='valid') == 2)
    zo = np.sum(np.convolve(D, [-1, 1], mode='valid') == 2)
    oo = np.sum(np.convolve(D, [1, 1], mode='valid') == 2)
    tot = zz + oz + zo + oo

    print(zz/tot, oz/tot, zo/tot, oo/tot)

    pos = np.cumprod(r[P])

    return pos

pos1 = func(p, r, n)
pos2 = func(p, r, n)

pos = 0.5 * (pos1 + pos2)

plt.plot(pos)
plt.plot(pos1)
plt.plot(pos2)
plt.show()
