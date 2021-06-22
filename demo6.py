from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 8], [3, 9]]
values = [1, 4, 5.5, 8]

regression1 = linear_model.LinearRegression()
regression1.fit(features, values)

print(f"coefficient={regression1.coef_}")
print(f"intercept={regression1.intercept_}")
print(f"score={regression1.score(features, values)}")

# predict
print(f"predict result={regression1.predict(features)}")