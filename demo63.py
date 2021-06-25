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
model.fit(train_data, train_targets, validation_split=0.1,
          epochs=100, batch_size=5, verbose=1)

for i, j in zip(test_data, test_targets):
    predict = model.predict(i.reshape(1, -1))
    predict_value = predict[0][0]
    print(f"predict as:{predict_value:.2f}, real as:{j:.2f}, diff is:{predict_value - j:.2f}")