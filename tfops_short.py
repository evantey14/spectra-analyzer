import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np

def default_initial_value(shape, std=0.05):
    return tf.random.normal(shape, 0., std)

def default_initializer(std=0.05):
    return tf.random_normal_initializer(0., std)

def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))

# wrapper tf.get_variable, augmented with 'init' functionality
# Get variable with data dependent init

@add_arg_scope
def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False, trainable=True):
    w = tf.compat.v1.get_variable(name, shape, dtype, None, trainable=trainable)
    if init:
        w = w.assign(initial_value)
        with tf.control_dependencies([w]):
            return w
    return w

# Activation normalization
@add_arg_scope
def actnorm(name, x, scale=1., logdet=None, hps=None, logscale_factor=1., batch_variance=False, reverse=False, init=False, trainable=True): #changed logscalefactor to 1?
    if arg_scope([get_variable_ddi], trainable=trainable):
        if not reverse:
            x = actnorm_center(name+"_center", x, reverse)
            x = actnorm_scale(name+"_scale", x, scale, logdet, hps,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
        else:
            x = actnorm_scale(name + "_scale", x, scale, logdet, hps,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
            x = actnorm_center(name+"_center", x, reverse)
        if logdet != None:
            return x, logdet
        return x

@add_arg_scope
def actnorm_center(name, x, reverse=False):
    # add a bias to x so it's centered around 0
    with tf.compat.v1.variable_scope(name):
        b = tf.get_variable("b", (1, 1, int_shape(x)[2]), initializer=tf.zeros_initializer())
        if not reverse:
            x += b
        else:
            x -= b
        return x

@add_arg_scope
def actnorm_scale(
    name, x, scale=1., logdet=None, hps=None, logscale_factor=3., batch_variance=False, reverse=False, init=False, trainable=True
):
    with tf.compat.v1.variable_scope(name), arg_scope([get_variable_ddi], trainable=trainable):
        logdet_factor = int_shape(x)[1]
        _shape = (1, 1, int_shape(x)[2])

        logs = tf.get_variable("logs", _shape, initializer=tf.zeros_initializer()) * logscale_factor
        if not reverse:
            x = x * tf.exp(logs)
        else:
            x = x * tf.exp(-logs)
        print(name, x.get_shape(), logs.get_shape(), logdet_factor)
        if logdet != None:
            dlogdet = tf.reduce_sum(logs) * logdet_factor
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x


@add_arg_scope
def conv1d(name, x, width, filter_size):
    with tf.compat.v1.variable_scope(name):
        n_in = int(x.get_shape()[2])

        stride_shape = [1, 1, 1]
        filter_shape = filter_size + [n_in, width]
        w = tf.compat.v1.get_variable("W", filter_shape, tf.float32,
                            initializer=default_initializer())

        x = tf.nn.conv1d(x, w, stride_shape, "SAME")
        
        #x = actnorm("actnorm", x)
        x += tf.get_variable("b", [1, 1, width],
                                 initializer=tf.zeros_initializer())

    return x


@add_arg_scope
def conv1d_zeros(name, x, width, filter_size, logscale_factor=3):
    with tf.compat.v1.variable_scope(name):
        n_in = int(x.get_shape()[2])
        stride_shape = [1, 1, 1]
        filter_shape = filter_size + [n_in, width]
        w = tf.compat.v1.get_variable("W", filter_shape, tf.float32,
                            initializer=tf.zeros_initializer())

        x = tf.nn.conv1d(x, w, stride_shape, "SAME")

        x += tf.compat.v1.get_variable("b", [1, 1, width],
                             initializer=tf.zeros_initializer())
        #x *= tf.exp(tf.compat.v1.get_variable("logs", [1, width], 
        #                            initializer=tf.zeros_initializer()) * logscale_factor)
    return x

def squeeze(x, factor):
    assert factor >= 1
    if factor == 1:
        return x
    batch_size, length, n_channels = int_shape(x)
    assert length % factor == 0
    x = tf.reshape(x, [-1, length // factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 2])
    x = tf.reshape(x, [-1, length // factor, n_channels * factor])
    return x

def unsqueeze(x, factor):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    length = int(shape[1])
    n_channels = int(shape[2])
    assert n_channels >= 2 and n_channels % 2 == 0
    x = tf.reshape(
        x, (-1, length, int(n_channels/factor), factor))
    x = tf.transpose(x, [0, 1, 3, 2])
    x = tf.reshape(x, (-1, int(length*factor), int(n_channels/factor)))
    return x

def flatten_sum(logps):
    if len(logps.get_shape()) == 3:
        return tf.reduce_sum(logps, [1, 2])
    else:
        raise Exception()

def standard_gaussian(shape):
    return gaussian_diag(tf.zeros(shape), tf.zeros(shape))

def gaussian_diag(mean, logsd):
    class o(object):
        pass
    o.mean = mean
    o.logsd = logsd
    o.eps = tf.random.normal(tf.shape(mean))
    o.sample = mean + tf.exp(logsd) * o.eps
    o.sample2 = lambda eps: mean + tf.exp(logsd) * eps
    o.logps = lambda x: -0.5 * \
        (np.log(2 * np.pi) + 2. * logsd + (x - mean) ** 2 / tf.exp(2. * logsd))
    o.logp = lambda x: flatten_sum(o.logps(x))
    o.get_eps = lambda x: (x - mean) / tf.exp(logsd)
    return o
