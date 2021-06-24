import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt

FILENAME = "data/diabetes.csv"
dataset1 = numpy.loadtxt(FILENAME, delimiter=",", skiprows=1)
print(type(dataset1))
print(dataset1.shape)

model = Sequential()
model.add(Dense(10, input_dim=8, activation=tf.nn.relu))
model.add(Dense(8, activation=tf.nn.relu))
model.add(Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
history = model.fit(inputList, resultList, epochs=200, batch_size=20, validation_split=0.25)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'], loc="upper right")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'], loc="upper right")
plt.show()

scores = model.evaluate(inputList, resultList)
print("score=", scores)
print("metrics:", model.metrics_names)
for s, n in zip(scores, model.metrics_names):
    print(f"{n} = {s}")