import numpy as np
import tensorflow as tf


def gd(cost, params, lr=np.float32(0.01)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        updates.append(param.assign(param - lr*g_param))
    return updates


def gd_clip(cost, params, lr=np.float32(0.01), thld=np.float32(1.0)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        g_param = tf.where(tf.greater(tf.abs(g_param), thld), thld/tf.norm(g_param, ord=1)*g_param, g_param)
        updates.append(param.assign(param - lr*g_param))
    return updates


def momentum(cost, params, lr=np.float32(0.01), gamma=np.float32(0.9)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        v = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='v')
        updates.append(v.assign(gamma*v - lr*g_param))
        with tf.control_dependencies(updates):
            updates.append(param.assign(param + v))
    return updates


def nesterov_momentum(cost, params, lr=np.float32(0.01), gamma=np.float32(0.9)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        v = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='v')
        updates.append(v.assign(gamma*v - lr*g_param))
        with tf.control_dependencies(updates):
            updates.append(param.assign(param + (gamma**2)*v - (1 + gamma)*lr*g_param))
    return updates


def adagrad(cost, params, lr=np.float32(0.01), eps=np.float32(1e-8)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        g2_sum = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='g2_sum')
        updates.append(g2_sum.assign(g2_sum + g_param**2))
        with tf.control_dependencies(updates):
            updates.append(param.assign(param - lr/tf.sqrt(g2_sum + eps)*g_param))
    return updates


def adadelta(cost, params, gamma=np.float32(0.95), eps=np.float32(1e-8)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        ms_g = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='ms_g')
        ms_d = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='ms_d')
        d_param = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='d_param')
        updates.append(ms_g.assign(gamma*ms_g + (1. - gamma)*g_param**2))
        with tf.control_dependencies(updates):
            updates.append(d_param.assign(-tf.sqrt((ms_d + eps)/(ms_g + eps))*g_param))
        with tf.control_dependencies(updates):
            updates.append(param.assign(param + d_param))
        with tf.control_dependencies(updates):
            updates.append(ms_d.assign(gamma*ms_d + (1. - gamma)*d_param**2))
    return updates


def rmsprop(cost, params, lr=np.float32(0.001), gamma=np.float32(0.9), eps=np.float32(1e-8)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        ms_g = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='ms_g')
        updates.append(ms_g.assign(gamma*ms_g + (1. - gamma)*g_param**2))
        with tf.control_dependencies(updates):
            updates.append(param.assign(param - lr/tf.sqrt(ms_g + eps)*g_param))
    return updates


def adam(cost, params, alpha=np.float32(0.001), beta_1=np.float32(0.9), beta_2=np.float32(0.999), eps=np.float32(1e-8)):
    g_params = tf.gradients(cost, params)
    t = tf.Variable(0.0, dtype=tf.float32, name='t')
    updates = []
    updates.append(t.assign(t + 1))
    with tf.control_dependencies(updates):
        for param, g_param in zip(params, g_params):
            m = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='m')
            v = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='v')
            alpha_t = alpha*tf.sqrt(1. - beta_2**t)/(1. - beta_1**t)
            updates.append(m.assign(beta_1*m + (1. - beta_1)*g_param))
            with tf.control_dependencies(updates):
                updates.append(v.assign(beta_2*v + (1. - beta_2)*g_param**2))
            with tf.control_dependencies(updates):
                updates.append(param.assign(param - alpha_t*m/(tf.sqrt(v) + eps)))
    return updates


def adamax(cost, params, alpha=np.float32(0.002), beta_1=np.float32(0.9), beta_2=np.float32(0.999), eps=np.float32(1e-8)):
    g_params = tf.gradients(cost, params)
    t = tf.Variable(1.0, dtype=tf.float32, name='t')
    updates = []
    updates.append(t.assign(t + 1))
    with tf.control_dependencies(updates):
        for param, g_param in zip(params, g_params):
            m = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='m')
            u = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='u')
            updates.append(m.assign(beta_1*m + (1. - beta_1)*g_param))
            with tf.control_dependencies(updates):
                updates.append(u.assign(tf.maximum(beta_2*u, tf.abs(g_param))))
            with tf.control_dependencies(updates):
                updates.append(param.assign(param - (alpha/(1. - beta_1**t))*(m/(u + eps))))
    return updates


def smorms3(cost, params, lr=np.float32(0.001), eps=np.float32(1e-16)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        m = tf.Variable(np.ones(param.get_shape(), dtype='float32'), name='m')
        g = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='g')
        g2 = tf.Variable(np.zeros(param.get_shape(), dtype='float32'), name='g2')
        r = 1./(m + 1.)
        updates.append(g.assign((1. - r)*g + r*g_param))
        updates.append(g2.assign((1. - r)*g2 + r*g_param**2))
        with tf.control_dependencies(updates):
            x = g**2/(g2 + eps)
            updates.append(param.assign(param - tf.minimum(lr, x)/tf.sqrt(g2 + eps)*g_param))
            updates.append(m.assign(1. + (1. - x)*m))
    return updates
