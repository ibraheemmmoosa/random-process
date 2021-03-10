import numpy as np
import matplotlib.pyplot as plt

p = 0.5

m = [np.random.binomial(1, p, size=1000).mean() for i in range(1000000)]

plt.hist(m, bins=1000, density=True)
plt.show()
