import tensorflow as tf

tf.compat.v1.disable_eager_execution()

t1 = tf.constant("hello tensorflow!")
print(t1)
session1 = tf.compat.v1.Session()
print("execute t1")
result = session1.run(t1)
print("result type=", type(result))
print(result)
session1.close()