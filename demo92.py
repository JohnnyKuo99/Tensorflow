#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
#from sklearn import datasets
# import sklearn.datasets
import sklearn.datasets as ds

#regressionData1 = sklearn.datasets.make_regression(10, 6, noise=5)
regressionData1 = ds.make_regression(10, 6, noise=5)
for i in range(6):
    x = regressionData1[0][:, i]
    y = regressionData1[1]

    plt.scatter(x, y)
    plt.show()
