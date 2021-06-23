import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from pprint import pprint

X = np.r_[np.random.randn(5000, 2) + [2, 2],
          np.random.randn(5000, 2) + [0, -2],
          np.random.randn(5000, 2) + [-2, 2]]

inertias = []
r1 = range(1, 15)
for k in r1:
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

pprint(inertias)
plt.plot(r1, inertias)
plt.show()