import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data[0])
# get current max depend on num_words
print(max([max(sequence) for sequence in train_data]))

# convert id ==> words
# get words ==> ids
word_index = imdb.get_word_index()
print(type(word_index))
word_index_list = [(k, v) for k, v in word_index.items()]
print(word_index_list[:10])
# convert ids ==> words
reverse_word_index = dict([(v, k) for k, v in word_index.items()])
for i in range(5):
    decoded_review = ' '.join([reverse_word_index.get(j - 3, "?") for j in train_data[i]])
    print(f"[{i}][rank:{train_labels[i]}][{decoded_review}]")

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
print(np.unique(x_train[0], return_counts=True))
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print(type(x_train), type(x_test), type(y_train), type(y_test))

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=40, batch_size=128, validation_split=0.1)

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, loss, 'bo--', label="training loss")
plt.plot(epochs, val_loss, 'b*-', label='validation loss')
plt.title("train V.S. validation loss")
plt.xlabel("epocs")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(epochs, accuracy, 'ro--', label="training accuracy")
plt.plot(epochs, val_accuracy, 'r*-', label='validation accuracy')
plt.title("training V.S. validation accuracy")
plt.xlabel("epochs")
plt.ylabel('accuracy')
plt.legend()
plt.show()