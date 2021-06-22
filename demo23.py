from sklearn.decomposition import PCA
import numpy as np
from sklearn import datasets, svm
from matplotlib import pyplot as plt

iris = datasets.load_iris()

pca = PCA(n_components=2)
data = pca.fit(iris.data).transform(iris.data)

print(data.shape)
print(data[0:5, ])
datamax = data.max(axis=0) + 0.5
datamin = data.min(axis=0) - 0.5
n = 1000
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))
# kerenl:linear ==> 0.966, c=10 ==> 0.97333
# kernel:poly ==> 0.946, c=10 ==> 0.96
# kernel:rbf ==> 0.96, C=10 ==> 0.966667
# kernel:sigmoid ==> 0.86, C=10 ==> 0.82666666
# svc = svm.SVC(kernel='linear', C=10)
# svc = svm.SVC(kernel='poly', C=10)
# svc = svm.SVC(kernel='rbf', C=10)
svc = svm.SVC(kernel='sigmoid', C=10)
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
vectors = svc.support_vectors_
plt.contour(X, Y, Z.reshape(X.shape))
print("score=", svc.score(data, iris.target))
for c, s in zip([0, 1, 2], ['o', 's', '.']):
    d = data[iris.target == c]
    plt.scatter(d[:, 0], d[:, 1], c='blue', marker=s)
plt.scatter(vectors[:, 0], vectors[:, 1], c='red', marker='*', alpha=0.3)
plt.show()