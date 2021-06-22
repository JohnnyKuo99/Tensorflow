from matplotlib import pyplot as plt
from sklearn import linear_model

regression1 = linear_model.LinearRegression()
features = [[1], [2], [3]]
values = [1, 4, 15]
plt.scatter(features, values, c='green')
plt.show()

regression1.fit(features, values)
print(f"coefficient={regression1.coef_}, intercept={regression1.intercept_}")
range1 = [-1, 3]
plt.plot(range1, regression1.coef_*range1+regression1.intercept_, c='gray')
plt.scatter(features, values, c='green')
plt.show()