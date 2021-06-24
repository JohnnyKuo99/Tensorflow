import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

FILENAME = "data/diabetes.csv"
dataset1 = numpy.loadtxt(FILENAME, delimiter=",", skiprows=1)
print(type(dataset1))
print(dataset1.shape)
inputList = dataset1[:, :8]
resultList = dataset1[:, 8]


def create_model():
    m = Sequential()
    m.add(Dense(10, input_dim=8, activation=tf.nn.relu))
    m.add(Dense(8, activation=tf.nn.relu))
    m.add(Dense(1, activation=tf.nn.sigmoid))
    m.summary()
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=20, verbose=0)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model, inputList, resultList, cv=fiveFold)
print(f"mean={results.mean()}, std={results.std()}")