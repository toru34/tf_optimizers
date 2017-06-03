import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from utils import f_props, load_mnist, tf_log
from layers import Dense
from optimizers import *

random_state = 42

mnist_X, mnist_y = load_mnist()
train_X, valid_X, train_y, valid_y = train_test_split(mnist_X, mnist_y, test_size=0.1, random_state=random_state)

x = tf.placeholder(tf.float32, [None, 784])
t = tf.placeholder(tf.float32, [None, 10])

layers = [
    Dense(784, 256, tf.nn.relu),
    Dense(256, 256, tf.nn.relu),
    Dense(256, 10, tf.nn.softmax)
]

y, params = f_props(layers, x)

cost = -tf.reduce_mean(tf.reduce_sum(t*tf_log(y), axis=1))
updates = smorms3(cost, params)

train = tf.group(*updates)
test  = tf.argmax(y, axis=1)

n_epochs = 10
batch_size = 100
n_batches = train_X.shape[0]//batch_size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
        for i in range(n_batches):
            start = i*batch_size
            end = start + batch_size
            sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
        pred_y, valid_cost = sess.run([test, cost], feed_dict={x: valid_X, t: valid_y})
        print('EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')))
