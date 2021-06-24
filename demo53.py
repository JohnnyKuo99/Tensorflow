import tensorflow as tf

t1 = [3.0, 4.0, 5.0]
t2 = [6.0, 6.0, 6.0]


def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


print(computeArea(tf.constant([t1, t2])).numpy())