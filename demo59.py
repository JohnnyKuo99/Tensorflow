import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

FILENAME = "data/diabetes.csv"
dataset1 = numpy.loadtxt(FILENAME, delimiter=",", skiprows=1)
print(type(dataset1))
print(dataset1.shape)

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]

fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
totalScores = []


def create_model():
    m = Sequential()
    m.add(Dense(10, input_dim=8, activation=tf.nn.relu))
    m.add(Dense(8, activation=tf.nn.relu))
    m.add(Dense(1, activation=tf.nn.sigmoid))
    m.summary()
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


for train, test in fiveFold.split(inputList, resultList):
    model = create_model()
    model.fit(inputList[train], resultList[train], epochs=200, batch_size=20, verbose=0)
    scores = model.evaluate(inputList, resultList)
    print("score=", scores)
    totalScores.append(scores[1] * 100)

print(f"total score={totalScores}, mean={numpy.mean(totalScores)},"
      f"std={numpy.std(totalScores)}")