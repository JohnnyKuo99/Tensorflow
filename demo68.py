import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("train shape={}".format(train_images.shape))
print("test shape={}".format(test_images.shape))
image1 = train_images[0]
print("train label shape={}".format(train_labels.shape))
print("test label shape={}".format(test_labels.shape))