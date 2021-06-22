from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(dir(iris))
print(list(iris.keys()))
print(iris["feature_names"])
print(iris["target_names"])
## get feature (petal width)
X = iris["data"][:, 3:]
print(X)
##
y = (iris["target"] == 2).astype(np.int)
print(y)

regression1 = LogisticRegression()
regression1.fit(X, y)
print(regression1.coef_)
print(regression1.intercept_)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = regression1.predict_proba(X_new)

plt.plot(X, y, "gs")
plt.plot(X_new, y_prob[:, 1], "r-", label="Iris-virginica")
plt.plot(X_new, y_prob[:, 0], "g--", label="Not Iris-virginica")
# z=a*x+b, coef_[0]*x+intercept_
# verify by logistyic regression formula (1/(1+e^-x))
z = X_new * regression1.coef_[0] + regression1.intercept_
f = 1 / (1 + np.exp(-z))
plt.plot(X_new, f, "b--", label="verify")
plt.legend(loc=2)
plt.show()