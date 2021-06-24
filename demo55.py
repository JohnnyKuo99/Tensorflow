import tensorflow as tf

vectors = [3.0, -1.0, 2.4, 5.9, 0.001, 8.5, -4, 0.00000001, -0.00000001]
print(tf.nn.relu(vectors).numpy())
print(tf.nn.sigmoid(vectors).numpy())