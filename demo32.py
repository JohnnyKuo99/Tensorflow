from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy

X = np.r_[np.random.randn(50, 2) + [2, 2], np.random.randn(50, 2) + [0, -2],
          np.random.randn(50, 2) + [-2, 2]]
print(X.shape)
[plt.scatter(e[0], e[1], c='black', s=7) for e in X]

k = 3
C_x = np.random.uniform(np.min(X[:, 0]), np.max(X[:, 0]), size=k)
C_y = np.random.uniform(np.min(X[:, 1]), np.max(X[:, 1]), size=k)
plt.scatter(C_x, C_y, marker='*', s=200, c='#FFC0EE')
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)
plt.show()


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
delta = dist(C, C_old, None)
print(f"delta={delta}")


def plot_kmeans(current_cluster, d):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots()
    plt.title(f'detla is {d:.4f}')
    # get k cluster, draw with associate color
    for index1 in range(k):
        pts = np.array([X[j] for j in range(len(X)) if current_cluster[j] == index1])
        ax.scatter(pts[:, 0], pts[:, 1], s=7, c=colors[index1])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c="#0599FF")
    plt.show()


while delta != 0:
    print(f"iterate with delta={delta}")
    # assign dot to cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        clusters[i] = np.argmin(distances)
    # find new K centers
    C_old = deepcopy(C)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    delta = dist(C, C_old, None)
    # print(clusters)
    plot_kmeans(clusters, delta)