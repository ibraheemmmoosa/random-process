import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0.0, 1.0

m = [np.random.normal(mu, sigma, 10).var() for i in range(1000000)]

plt.hist(m, 1000, density=True)
plt.show()
