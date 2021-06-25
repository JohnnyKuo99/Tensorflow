from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

FLATTEN_DIM = 28 * 28
TRAIN_SIZE = len(train_images)
TEST_SIZE = len(test_images)

trainImages = np.reshape(train_images, (TRAIN_SIZE, FLATTEN_DIM))
testImages = np.reshape(test_images, (TEST_SIZE, FLATTEN_DIM))
print(trainImages[0])

# transfer to float
trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
trainImages /= 255
testImages /= 200
print(trainImages[0])

NUM_DIGITS = 10

trainLabels = utils.to_categorical(train_labels, NUM_DIGITS)
testLabels = utils.to_categorical(test_labels, NUM_DIGITS)
print(trainLabels[0])

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(FLATTEN_DIM,)))
model.add(Dense(64, activation='relu',))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

tbCallback = TensorBoard(log_dir="logs/demo70", histogram_freq=0, write_graph=True,
                         write_images=True)
model.fit(trainImages, trainLabels, epochs=10, callbacks=[tbCallback])

predictLabels = model.predict_classes(testImages)
print("result=", predictLabels[:10])


import matplotlib.pyplot as plt
def plotTestImage(index):
    plt.title("The test image #%d marked as %d, predicted as %d" % (index, test_labels[index], predictLabels[index]))
    plt.imshow(test_images[index], cmap='binary')
    plt.show()
plotTestImage(0)
