from pandas import read_csv
from numpy import unique
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold
from numpy import mean, std

FILENAME = "data/iris.data"

df1 = read_csv(FILENAME, header=None)
print(df1.shape)
print(df1.describe())
dataset = df1.values
features = dataset[:, :4].astype(float)
labels = dataset[:, 4]
# print(features)
# print(unique(labels, return_counts=True))

# convert label to 1-hot encoding
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
# print(type(encoded_Y), encoded_Y[:10], unique(encoded_Y, return_counts=True))
dummy_y = to_categorical(encoded_Y)


# print(type(dummy_y), dummy_y[0, :], dummy_y[50, :], dummy_y[100, :])


def build_model():
    m = Sequential()
    m.add(Dense(8, input_dim=4, activation='relu'))
    m.add(Dense(3, activation="softmax"))
    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    m.summary()
    return m


e = KerasClassifier(build_fn=build_model, epochs=200, batch_size=10, verbose=0)
kfold = KFold(n_splits=5, shuffle=True)
result = cross_val_score(e, features, dummy_y, cv=kfold)
print(f"result={result},mean={mean(result)}, std={std(result)}")