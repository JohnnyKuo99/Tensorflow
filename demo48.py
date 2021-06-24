import tensorflow as tf
import numpy as np

a = np.array([5, 3, 8])
b = np.array([3, -1, 2])
c = np.add(a, b)
print("array form, c=", c)

a2 = tf.constant(a)
b2 = tf.constant(b)
c2 = tf.add(a2, b2)
print("tensor form, c2=", c2)
print("array form, c2=", c2.numpy())
print("c should equal to c2?", c == c2.numpy())