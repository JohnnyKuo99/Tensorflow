from numpy import array
from sklearn.decomposition import PCA

A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print(A)
pca = PCA(3)
pca.fit(A)
print(pca)

print("components", pca.components_)
print("variance", pca.explained_variance_)
print("variance ratio", pca.explained_variance_ratio_)

B = pca.transform(A)
print(B)