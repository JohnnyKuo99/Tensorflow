import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import svm

iris = datasets.load_iris()

pca = PCA(n_components=2)
pca.fit(iris.data)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
data = pca.transform(iris.data)

print(data.shape)

datamax = data.max(axis=0) + 1
datamin = data.min(axis=0) - 1

n = 200
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))
svc = svm.SVC()
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])

plt.contour(X, Y, Z.reshape(X.shape), colors=['r', 'g', 'b'])

for i, c in zip([0, 1, 2], ['r', 'g', 'b']):
    d = data[iris.target == i]
    plt.scatter(d[:, 0], d[:, 1], c=c)
plt.show()