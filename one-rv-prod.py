import numpy as np
import matplotlib.pyplot as plt

# p is probability of rise
p = 0.75
r = 1.0 + 1e-3
n = 100

P = np.random.rand(n)
D = 2 * (P < p) - 1

zz = np.sum(np.convolve(D, [-1, -1], mode='valid') == 2)
oz = np.sum(np.convolve(D, [1, -1], mode='valid') == 2)
zo = np.sum(np.convolve(D, [-1, 1], mode='valid') == 2)
oo = np.sum(np.convolve(D, [1, 1], mode='valid') == 2)
tot = zz + oz + zo + oo

print(zz/tot, oz/tot, zo/tot, oo/tot)

pos = r ** np.cumsum(D)

plt.plot(pos)
plt.show()
