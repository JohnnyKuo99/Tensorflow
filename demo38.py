import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
classifier = GaussianNB()
classifier.fit(X, Y)
print(classifier.predict([[0, 0], [0, 0.5], [0.5, -0.5], [-0.5, -0.5]]))

classifier2 = GaussianNB()
classifier2.partial_fit(X, Y, np.unique(Y))
print(classifier2.predict([[0, 0], [0, 0.5], [0.5, -0.5], [-0.5, -0.5]]))
classifier2.partial_fit([[0.3, -0.3]], [2])
print(classifier2.predict([[0, 0], [0, 0.5], [0.5, -0.5], [-0.5, -0.5]]))