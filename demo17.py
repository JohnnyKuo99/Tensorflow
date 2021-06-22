import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.1)
f = 1 / (1 + np.exp(-x))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, f)
plt.axhline(0.5, color='black')
plt.axvline(0, color='black')
plt.show()