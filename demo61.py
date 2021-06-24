import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

FILENAME = "data/diabetes.csv"
dataset1 = numpy.loadtxt(FILENAME, delimiter=",", skiprows=1)
print(type(dataset1))
print(dataset1.shape)
inputList = dataset1[:, :8]
resultList = dataset1[:, 8]


def create_model(optimizer='adam', init='uniform'):
    m = Sequential()
    m.add(Dense(10, input_dim=8, kernel_initializer=init, activation=tf.nn.relu))
    m.add(Dense(8, activation=tf.nn.relu))
    m.add(Dense(1, activation=tf.nn.sigmoid))
    m.summary()
    m.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return m


model = KerasClassifier(build_fn=create_model, verbose=0)
#  epochs=200, batch_size=20,
optimizers = ['rmsprop', 'adam', 'sgd']
inits = ['normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 15]
parameter_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)
grid = GridSearchCV(estimator=model, param_grid=parameter_grid)
grid_result = grid.fit(inputList, resultList)