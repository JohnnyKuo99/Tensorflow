import tensorflow as tf

l1 = [1, 2, 3]
l2 = [4, 5, 6]


def add1(p, q):
    return tf.math.add(p, q)


@tf.function
def add2(p, q):
    return tf.math.add(p, q)

def add3(p, q):
    t1 = tf.constant(p)
    t2 = tf.constant(q)
    return t1 + t2


@tf.function
def add4(p, q):
    t1 = tf.constant(p)
    t2 = tf.constant(q)
    return t1 + t2


print(add1(l1, l2))
print(add2(l1, l2))
print(add3(l1, l2))
print(add4(l1, l2))