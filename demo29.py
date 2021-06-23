from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 0], [0, 1], [1, 2], [1, 4], [2, 0],
              [4, 2], [4, 4], [4, 0], [4, 6], [5, 7]])
kmeans = KMeans(n_clusters=2).fit(X)
print("label", kmeans.labels_)
print("centers", kmeans.cluster_centers_)
print("inertia=", kmeans.inertia_)

print("predict", kmeans.predict([[0, 0], [3, 0], [0, 3], [3, 3], [3.5, 3.5]]))