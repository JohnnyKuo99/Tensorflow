import numpy as np
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
print(type(diabetes))
print("feature", diabetes.data.shape)
print("label", diabetes.target.shape)

dataForTest = -60
data_train = diabetes.data[:dataForTest]
target_train = diabetes.target[:dataForTest]
print("feature for training", data_train.shape)
print("label for training", target_train.shape)

data_test = diabetes.data[dataForTest:]
target_test = diabetes.target[dataForTest:]
print("feature for testing", data_test.shape)
print("label for testing", target_test.shape)

regression1 = linear_model.LinearRegression()
regression1.fit(data_train, target_train)
print(f"coef={regression1.coef_}")
print(f"intercept={regression1.intercept_}")

# calculate score
print(f"score={regression1.score(data_test, target_test)}")

# predict
for i in range(dataForTest, 0):
    data = data_test[i]
    # print(data.shape)
    data = data.reshape(1, -1)
    print(f"predict={regression1.predict(data)[0]:.2f}, actual={target_test[i]:.2f}")
# mean square error
MSE = np.mean((regression1.predict(data_test) - target_test) ** 2)
print(f"MSE={MSE}")