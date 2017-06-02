import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, 1.0))


def f_props(layers, x):
    params = []
    for layer in layers:
        x = layer.f_prop(x)
        params += layer.params
    return x, params


def load_mnist(one_hot=True):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=one_hot)
    return mnist.train.images, mnist.train.labels
