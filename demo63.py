from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
print(mean)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
print(train_data.shape, test_data.shape)


def build_model():
    m = Sequential()
    m.add(Dense(32, activation='relu', input_shape=(train_data.shape[1],)))
    m.add(Dense(64, activation='relu'))
    m.add(Dense(1))
    m.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return m


model = build_model()
model.summary()