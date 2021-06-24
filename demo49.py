import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
a = np.array([5, 3, 8])
b = np.array([3, -1, 2])
c = np.add(a, b)
print("array form, c=", c)

a2 = tf.constant(a)
b2 = tf.constant(b)
c2 = tf.add(a2, b2)
print("tensor form, c2=", c2)
with tf.compat.v1.Session() as session1:
    result = session1.run(c2)
    print("result=", result)
    print("result equal to c?", c == result)