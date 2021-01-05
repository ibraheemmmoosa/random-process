import numpy as np
import matplotlib.pyplot as plt

# p is probability of rise
p = 0.5 - 1e-5
d = 1.0
n = 10000000

P = np.random.rand(n)
D = 2 * (P < p) - 1

zz = np.sum(np.convolve(D, [-1, -1], mode='valid') == 2)
oz = np.sum(np.convolve(D, [1, -1], mode='valid') == 2)
zo = np.sum(np.convolve(D, [-1, 1], mode='valid') == 2)
oo = np.sum(np.convolve(D, [1, 1], mode='valid') == 2)
tot = zz + oz + zo + oo

print(zz/tot, oz/tot, zo/tot, oo/tot)

pos = d * np.cumsum(D)

plt.plot(pos)
plt.show()
