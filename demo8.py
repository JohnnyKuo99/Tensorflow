from sklearn import datasets
from pprint import pprint

regressionData = datasets.make_regression(10, 6, noise=5)

regressionX = regressionData[0]
print(type(regressionX), regressionX.shape)
r1 = sorted(regressionX, key=lambda t: t[0])
pprint(r1)
r6 = sorted(regressionX, key=lambda t: t[5])
pprint(r6)