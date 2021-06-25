import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

FILENAME = "data/bmi.csv"
csv = pd.read_csv(FILENAME)
csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100
print(csv.describe())
# apply string to 1-hot encoding
encoder = LabelBinarizer()
transformedLabel = encoder.fit_transform(csv['label'])
print(csv['label'][:10])
print(transformedLabel[:10])

model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(2,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

tbCallback = callbacks.TensorBoard(log_dir="logs/demo74", histogram_freq=1)

train_csv = csv[:25000]
test_csv = csv[25000:]
train_pat = train_csv[['weight', 'height']]
test_pat = test_csv[['weight', 'height']]
train_ans = transformedLabel[:25000]
test_ans = transformedLabel[25000:]

model.fit(train_pat, train_ans, batch_size=50, epochs=50, verbose=1,
          validation_data=(test_pat, test_ans), callbacks=[tbCallback])
score = model.evaluate(test_pat, test_ans, verbose=0)
print('test loss', score[0])
print('test accuracy', score[1])
