from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest, f_regression

X, y = make_regression(n_samples=1000, n_features=10, n_informative=7)
print(X.shape)
r1 = LinearRegression()
r1.fit(X, y)
print(r1.coef_)
importance = r1.coef_
# iterate coef_
for i, v in enumerate(importance):
    print(f"feature{i:d}, score={v:.2f}")
pyplot.bar([x for x in range(len(importance))], importance)

kBest = SelectKBest(f_regression, k=7).fit(X, y)
print(kBest.get_support())
newX = kBest.fit_transform(X, y)
print(X[:1])
print(newX[:1])
pyplot.show()