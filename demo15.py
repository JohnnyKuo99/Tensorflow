import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])
plt.plot(x, y)
plt.scatter(x, y)

# train original data
r1 = LinearRegression()
r1.fit(x, y)
# plot regression
# plt.plot(x, r1.coef_ * x + r1.intercept_)
# plot regression using predict
plt.plot(x, r1.predict(x))
plt.show()

print(f"linear regression, score={r1.score(x, y):.2f}")
x_seq = np.array(np.arange(5, 55, 0.1)).reshape(-1, 1)

# generate 2nd order feature
transformer = PolynomialFeatures(degree=7, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
# print(f"orig x shape= {x.shape}")
# print(f"new x shape={x_.shape}")
# print(f"orig x= {x}")
# print(f"new x={x_}")
# fit using linear regression (same)
r2 = LinearRegression().fit(x_, y)
print(f"2nd order score={r2.score(x_, y)}")
print(f"2nd order coef={r2.coef_}")
x_seq_ = transformer.transform(x_seq)
y_ = r2.predict(x_seq_)
plt.plot(x, y)
plt.scatter(x, y)
plt.plot(x_seq, y_)
plt.show()