import matplotlib.pyplot as plt
import numpy as np

label = 'w=%.1f'

x = np.arange(-10, 10, 0.1)

for w in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0]:
    f = 1 / (1 + np.exp(-x * w))
    plt.plot(x, f, label=label % w)
plt.legend(loc=2)
plt.show()