import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

FILENAME = "data/diabetes.csv"
dataset1 = numpy.loadtxt(FILENAME, delimiter=",", skiprows=1)
print(type(dataset1))
print(dataset1.shape)


def createModel():
    # global model
    m = Sequential()
    m.add(Dense(10, input_dim=8, activation=tf.nn.relu))
    m.add(Dense(8, activation=tf.nn.relu))
    m.add(Dense(1, activation=tf.nn.sigmoid))
    m.summary()
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


model = createModel()

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
model.fit(inputList, resultList, epochs=200, batch_size=20)
# make a directory model
tf.keras.models.save_model(model, 'model/demo57')

scores = model.evaluate(inputList, resultList, verbose=False)
print("score=", scores)
print("metrics:", model.metrics_names)
for s, n in zip(scores, model.metrics_names):
    print(f"{n} = {s}")

model2 = createModel()
scores2 = model2.evaluate(inputList, resultList, verbose=False)
print("[2]score=", scores2)
print("[2]metrics:", model2.metrics_names)
for s, n in zip(scores2, model2.metrics_names):
    print(f"{n} = {s}")

model3 = tf.keras.models.load_model('model/demo57')
scores3 = model3.evaluate(inputList, resultList, verbose=False)
print("[3]score=", scores3)
print("[3]metrics:", model3.metrics_names)
for s, n in zip(scores3, model3.metrics_names):
    print(f"{n} = {s}")