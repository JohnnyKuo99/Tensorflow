
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

# get iris dataset
iris = datasets.load_iris()

print(type(iris))
X = iris.data
species = iris.target
print(type(X), X.shape)
print(type(species), np.unique(species))
print(dir(iris))
# depend on feature, pair drawing
counter = 1
for i in range(0, 4):
    for j in range(i + 1, 4):
        plt.figure(counter, figsize=(8, 6))
        counter += 1
        xData = X[:, i]
        yData = X[:, j]
        x_min, x_max = xData.min() - 0.5, xData.max() + 0.5
        y_min, y_max = yData.min() - 0.5, yData.max() + 0.5
        plt.scatter(xData, yData, c=species, cmap=plt.cm.Paired)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel(iris['feature_names'][i])
        plt.ylabel(iris['feature_names'][j])
        plt.show()