import tensorflow as tf
import numpy as np

def int_shape(x): # should probably do something about this
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))

# logs used to have a scale that was fixed to 3. I've removed it.

def actnorm(name, x, logdet):
    with tf.compat.v1.variable_scope(name):
        _, length, n_channels = int_shape(x)
        shape = (1, 1, n_channels)
        b = tf.get_variable("b", shape, initializer=tf.zeros_initializer())
        logs = tf.get_variable("logs", shape, initializer=tf.zeros_initializer())
        x += b
        x = x * tf.exp(logs)
        dlogdet = tf.reduce_sum(logs) * length
        return x, logdet + dlogdet

def actnorm_reverse(name, x):
    with tf.compat.v1.variable_scope(name):
        _, _, n_channels = int_shape(x)
        shape = (1, 1, n_channels)
        b = tf.get_variable("b", shape, initializer=tf.zeros_initializer())
        logs = tf.get_variable("logs", shape, initializer=tf.zeros_initializer())
        x = x * tf.exp(-logs)
        x -= b
        return x

def conv1d(name, x, width, channels_out, initializer=tf.random_normal_initializer(0, .05)):
    with tf.compat.v1.variable_scope(name):
        channels_in = int(x.get_shape()[2])
        stride_shape = 1 #[1, 1, 1]
        filter_shape = [width, channels_in, channels_out]
        w = tf.compat.v1.get_variable("W", filter_shape, tf.float32, initializer=initializer)
        x = tf.nn.conv1d(x, w, stride_shape, "SAME")
        # i don't know why the line below exists? but it seems like it adds the same bias after each
        # conv. in the original code it looks like there's an actnorm here.
        #x += tf.compat.v1.get_variable("b", [1, 1, width], initializer=tf.zeros_initializer())
        return x

def split(z):
    '''Split a (batch_size, length, n_channels) tensor in half across the channel dimension.'''
    assert int(z.get_shape()[2]) == 4
    return z[:, :, :2], z[:, :, 2:]

def unsplit(z1, z2):
    '''Recombine two tensors across the channel dimension.'''
    return tf.concat([z1, z2], 2)

def squeeze(x, factor):
    _, length, n_channels = int_shape(x)
    assert length % factor == 0
    x = tf.reshape(x, [-1, length // factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 2])
    x = tf.reshape(x, [-1, length // factor, n_channels * factor])
    return x

def unsqueeze(x, factor):
    _, length, n_channels = int_shape(x)
    assert n_channels >= 2 and n_channels % 2 == 0
    x = tf.reshape(x, [-1, length, n_channels // factor, factor])
    x = tf.transpose(x, [0, 1, 3, 2])
    x = tf.reshape(x, [-1, length * factor, n_channels // factor])
    return x

def gaussian_diag(mean, logsd):
    class o(object):
        pass
    o.mean = mean
    o.logsd = logsd
    o.sample = lambda: mean + tf.exp(logsd) * tf.random.normal(tf.shape(o.mean))
    o.logps = lambda x: -0.5 * \
        (np.log(2 * np.pi) + 2. * logsd + (x - mean) ** 2 / tf.exp(2. * logsd))
    o.logp = lambda x: tf.reduce_sum(o.logps(x), [1, 2])
    o.get_eps = lambda x: (x - mean) / tf.exp(logsd)
    return o
