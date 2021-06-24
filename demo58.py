import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
feature_train, feature_test, label_train, label_test = \
    train_test_split(inputList, resultList, test_size=0.25, stratify=resultList)

# verify after train_test_split, will still keep ratio
for data in [resultList, label_train, label_test]:
    classes, counts = numpy.unique(data, return_counts=True)
    for cl, co in zip(classes, counts):
        print(f"{int(cl)}==>{co / sum(counts)}")

model.fit(feature_train, label_train, validation_data=(feature_test, label_test),
          epochs=200, batch_size=20)

scores = model.evaluate(feature_test, label_test)
print("score=", scores)
print("metrics:", model.metrics_names)
for s, n in zip(scores, model.metrics_names):
    print(f"{n} = {s}")