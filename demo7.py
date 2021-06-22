import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import numpy

regressionData = datasets.make_regression(100, 1, noise=5)
print(type(regressionData))
print(regressionData[0].shape, regressionData[1].shape)
plt.scatter(regressionData[0], regressionData[1], c='red', marker='^')
plt.show()

regression1 = linear_model.LinearRegression()
regression1.fit(regressionData[0], regressionData[1])
print(f"coef={regression1.coef_}, intercept={regression1.intercept_}")

range1 = numpy.arange(regressionData[0].min() - 0.5,
                      regressionData[0].max() + 0.5, 1)
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_)
plt.scatter(regressionData[0], regressionData[1], c='red', marker='^')
print(f"score={regression1.score(regressionData[0], regressionData[1])}")
plt.show()