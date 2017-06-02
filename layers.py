import numpy as np
import tensorflow as tf

rng = np.random.RandomState(1234)


class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        # Xavier initialization
        self.W = tf.Variable(rng.uniform(
            low=-np.sqrt(6/(in_dim + out_dim)),
            high=np.sqrt(6/(in_dim + out_dim)),
            size=(in_dim, out_dim)
        ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'), name='b')
        self.function = function
        self.params = [self.W, self.b]

    def f_prop(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)
