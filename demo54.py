import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

FILENAME = "data/diabetes.csv"
dataset1 = numpy.loadtxt(FILENAME, delimiter=",", skiprows=1)
print(type(dataset1))
print(dataset1.shape)

model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
model.fit(inputList, resultList, epochs=200, batch_size=20)

scores = model.evaluate(inputList, resultList)
print("score=", scores)
print("metrics:", model.metrics_names)
for s, n in zip(scores, model.metrics_names):
    print(f"{n} = {s}")