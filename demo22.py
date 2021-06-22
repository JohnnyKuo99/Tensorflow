import numpy as np
from sklearn.svm import SVC

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [2, 2], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2, 2])
classifier1 = SVC()
classifier1.fit(X, y)
print(classifier1)
X2 = np.array([[5, 0], [-5, 0], [0, 5], [0, -5]])
print(f"predict x2= {classifier1.predict(X2)}")

print(f"support vector={classifier1.support_vectors_}")
print(f"support vector indices={classifier1.support_}")
print(f"support at each class={classifier1.n_support_}")