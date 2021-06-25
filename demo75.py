import tensorflow as tf

l1 = [1, 1, 1, 0, 0, 0] * 6
image = tf.constant(l1, tf.float32)
tensor1 = tf.reshape(image, [1, 6, 6, 1])
print(tensor1[0, :, :, 0].numpy())

#l2 = [1, 0, -1] * 3
l2 = [-1, 0, 1] * 3
filter = tf.constant(l2, tf.float32)
tensor2 = tf.reshape(filter, [3, 3, 1, 1])
print(tensor2[:, :, 0, 0].numpy())

result = tf.nn.conv2d(tensor1, tensor2, [1, 1, 1, 1], padding='VALID')
print(result.numpy()[0, :, :, 0])