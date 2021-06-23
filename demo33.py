import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X = np.r_[np.random.randn(100, 2) + [2, 2],
          np.random.randn(100, 2) + [0, -2],
          np.random.randn(100, 2) + [-2, 2]]
k = 4
kmeans = KMeans(n_clusters=k, n_init=1, verbose=True)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(f"inertia={kmeans.inertia_}")
print(kmeans.n_clusters)
print(np.unique(kmeans.labels_, return_counts=True))

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
markers = ['.', 'o', 's', '^', '*', 'x', 'v']
for i in range(k):
    dataX = X[kmeans.labels_ == i]
    plt.scatter(dataX[:, 0], dataX[:, 1], c=colors[i], marker=markers[i])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker="*", s=200, c='#FFC0EE')
plt.show()