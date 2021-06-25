import numpy
from tensorflow.keras.datasets import imdb
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)
print(X.shape)
print(y.shape)

# how many classes
print(numpy.unique(y, return_counts=True))
# first comment
print(X[0])
print(X[1])

# how many symbol
print(len(numpy.unique(numpy.hstack(X))))

result = [len(x) for x in X]
print(f"imdb features average len:{numpy.mean(result)}, std:{numpy.std(result)}")

plt.subplot(121)
plt.boxplot(result)
plt.subplot(122)
plt.hist(result)
plt.show()