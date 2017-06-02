import numpy as np
import tensorflow as tf


def sgd(cost, params, lr=np.float32(0.01)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        updates.append(param.assign(param - lr*g_param))
    return updates


def sgd_clip(cost, params, lr=np.float32(0.01), thld=np.float32(1.0)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        g_param = tf.maximum(g_param, thld)
        updates.append(param.assign(param - lr*g_param))
    return updates


def momentum(cost, params, lr=np.float32(0.01), gamma=np.float32(0.9)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        v = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='v')
        updates.append(v.assign(gamma*v - lr*g_param))
        updates.append(param.assign(param + v))
    return updates


def nesterov_momentum(cost, params, lr=np.float32(0.01), gamma=np.float32(0.9)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        v = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='v')
        updates.append(v.assign(gamma*v - lr*g_param))
        updates.append(param.assign(param + (gamma**2)*v - (1 + gamma)*lr*g_param))
    return updates


# 微妙
def adagrad(cost, params, lr=np.float32(0.01), eps=np.float32(1)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        g2_sum = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='g2_sum')
        updates.append(g2_sum.assign(g2_sum + g_param**2))
        updates.append(param.assign(param - (lr/tf.sqrt(g2_sum + eps))*g_param))
    return updates


# costがnanになる
def adadelta(cost, params, gamma=np.float32(0.95), eps=np.float32(1e-8)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        ms_g = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='ms_g')
        ms_d = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='ms_d')
        d_param = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='d_param')
        updates.append(ms_g.assign(gamma*ms_g + (1 - gamma)*g_param**2))
        updates.append(d_param.assign((-tf.sqrt(ms_d + eps)/tf.sqrt(ms_g + eps))*g_param))
        updates.append(param.assign(param + d_param))
        updates.append(ms_d.assign(gamma*ms_d + (1 - gamma)*d_param*2))
    return updates


def rmsprop(cost, params, lr=np.float32(0.001), gamma=np.float32(0.9), eps=np.float32(1e-8)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        ms_g = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='ms_g')
        updates.append(ms_g.assign(gamma*ms_g + (1 - gamma)*g_param**2))
        updates.append(param.assign(param - lr/tf.sqrt(ms_g + eps)*g_param))
    return updates


def adam(cost, params, alpha=np.float32(0.001), beta_1=np.float32(0.9), beta_2=np.float32(0.999), eps=np.float32(1e-8)):
    g_params = tf.gradients(cost, params)
    t = tf.Variable(1.0, dtype=tf.float32, name='t')
    updates = []
    for param, g_param in zip(params, g_params):
        m = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='m')
        v = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='v')
        alpha_t = alpha*tf.sqrt(1 - beta_2**t)/(1 - beta_1**t)
        updates.append(m.assign(beta_1*m + (1 - beta_1)*g_param))
        updates.append(v.assign(beta_2*v + (1 - beta_2)*g_param**2))
        updates.append(param.assign(param - alpha_t*m/(tf.sqrt(v) + eps)))
    updates.append(t.assign(t + 1))
    return updates


def adamax(cost, params, alpha=np.float32(0.002), beta_1=np.float32(0.9), beta_2=np.float32(0.999), eps=np.float32(1e-8)):
    g_params = tf.gradients(cost, params)
    t = tf.Variable(1.0, dtype=tf.float32, name='t')
    updates = []
    for param, g_param in zip(params, g_params):
        m = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='m')
        u = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='u')
        updates.append(m.assign(beta_1*m + (1 - beta_1)*g_param))
        updates.append(u.assign(tf.maximum(beta_2*u, tf.abs(g_param))))
        updates.append(param.assign(param - (alpha/(1 - beta_1**t))*(m/(u + eps))))
    updates.append(t.assign(t + 1))
    return updates


# grad by grad
