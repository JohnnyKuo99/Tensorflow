import tensorflow as tf

l1 = [1, 1, 1, 0, 0, 0] * 6
image = tf.constant(l1, tf.float32)
tensor1 = tf.reshape(image, [1, 6, 6, 1])
print(tensor1[0, :, :, 0])