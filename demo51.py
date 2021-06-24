import tensorflow as tf

tf.compat.v1.disable_eager_execution()

l1 = [1, 2, 3]
l2 = [4, 5, 6]

a = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
b = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
c = tf.add(a, b)

with tf.compat.v1.Session() as session1:
    result = session1.run(c, feed_dict={a: l1, b: l2})
    print(result)