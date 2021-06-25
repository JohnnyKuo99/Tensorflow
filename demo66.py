import numpy as np
import tensorflow as tf

scores = [3, 5, 4, 1, 2]


def mySoftMax(x):
    ax = np.array(x)
    return np.exp(ax) / np.sum(np.exp(ax), axis=0)


def normalRatio(x):
    ax = np.array(x)
    return ax / np.sum(ax, axis=0)


print("softmax by tensorflow result={}".format(tf.nn.softmax(np.array(scores, dtype=float))))
print("softmax result={}".format(mySoftMax(scores)))
print("normal result={}".format(normalRatio(scores)))