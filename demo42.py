from sklearn import datasets
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()

X = iris.data
species = iris.target

X_reduced = PCA(n_components=3).fit_transform(X)
print(X_reduced)
fig = plt.figure(1, figsize=(8, 8))
ax = Axes3D(fig, elev=-150, azim=180)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=species, cmap=plt.cm.Paired)
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")
plt.show()