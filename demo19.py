import matplotlib.pyplot as plt
import numpy as np

label = 'b=%.1f'

x = np.arange(-10, 10, 0.1)
w = 3
for b in [-10, -5, 0, 5, 10]:
    f = 1 / (1 + np.exp(-(x * w + b)))
    plt.plot(x, f, label=label % b)
plt.legend(loc=2)
plt.show()