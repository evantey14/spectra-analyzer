import tensorflow as tf
import numpy as np

def int_shape(z): # should probably do something about this
    return [-1]+list(map(int, z.get_shape()[1:]))

# logs used to have a scale that was fixed to 3. I've removed it.
def actnorm(name, z, logdet):
    with tf.compat.v1.variable_scope(name):
        _, length, n_channels = int_shape(z)
        shape = (1, 1, n_channels)
        b = tf.compat.v1.get_variable("b", shape, initializer=tf.zeros_initializer())
        logs = tf.compat.v1.get_variable("logs", shape, initializer=tf.zeros_initializer())
        z += b
        z = z * tf.exp(logs)
        dlogdet = tf.reduce_sum(logs) * length
        return z, logdet + dlogdet

def actnorm_reverse(name, z):
    with tf.compat.v1.variable_scope(name):
        _, _, n_channels = int_shape(z)
        shape = (1, 1, n_channels)
        b = tf.compat.v1.get_variable("b", shape, initializer=tf.zeros_initializer())
        logs = tf.compat.v1.get_variable("logs", shape, initializer=tf.zeros_initializer())
        z = z * tf.exp(-logs)
        z -= b
        return z

def conv1d(name, z, width, channels_out, initializer=tf.random_normal_initializer(0, .05)):
    with tf.compat.v1.variable_scope(name):
        channels_in = int(z.get_shape()[2])
        stride_shape = 1 #[1, 1, 1]
        filter_shape = [width, channels_in, channels_out]
        w = tf.compat.v1.get_variable("W", filter_shape, tf.float32, initializer=initializer)
        z = tf.nn.conv1d(z, w, stride_shape, "SAME")
        # i don't know why the line below exists? but it seems like it adds the same bias after each
        # conv. in the original code it looks like there's an actnorm here.
        #x += tf.compat.v1.get_variable("b", [1, 1, width], initializer=tf.zeros_initializer())
        return z

def f(name, z, channels_out):
    _, _, original_channels = int_shape(z)
    with tf.compat.v1.variable_scope(name):
        z = tf.nn.relu(conv1d('l_1', z, 101, channels_out))
        z = tf.nn.relu(conv1d('l_2', z, 1, channels_out))
        z = conv1d('l_last', z, 101, original_channels, initializer=tf.zeros_initializer())
        return z

def invertible_1x1_conv(name, z, logdet, reverse):
    with tf.compat.v1.variable_scope(name):
        batch_size, length, n_channels = int_shape(z)
        w_shape = [n_channels, n_channels]

        # Sample a random orthogonal matrix:
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
        w = tf.compat.v1.get_variable('W', dtype=tf.float32, initializer=w_init)

        #dlogdet = tf.linalg.LinearOperator(w).log_abs_determinant() * length
        dlogdet = tf.cast(tf.math.log(abs(tf.linalg.det(tf.cast(w, 'float64')))), 'float32') * length

        if not reverse:
            _w = tf.reshape(w, [1] + w_shape)
            #z = tf.nn.conv1d(z, _w, [1, 1, 1], 'SAME')
            z = tf.nn.conv1d(z, _w, 1, 'SAME')
            logdet += dlogdet
        else:
            _w = tf.reshape(tf.linalg.inv(w), [1]+w_shape)
            #z = tf.nn.conv1d(z, _w, [1, 1, 1], 'SAME')
            z = tf.nn.conv1d(z, _w, 1, 'SAME')
            logdet -= dlogdet

        return z, logdet

def split(z):
    '''Split a (batch_size, length, n_channels) tensor in half across the channel dimension.'''
    assert int(z.get_shape()[2]) == 4
    return z[:, :, :2], z[:, :, 2:]

def unsplit(z1, z2):
    '''Recombine two tensors across the channel dimension.'''
    return tf.concat([z1, z2], 2)

def squeeze(z, factor):
    _, length, n_channels = int_shape(z)
    assert length % factor == 0
    z = tf.reshape(z, [-1, length // factor, factor, n_channels])
    z = tf.transpose(z, [0, 1, 3, 2])
    z = tf.reshape(z, [-1, length // factor, n_channels * factor])
    return z

def unsqueeze(z, factor):
    _, length, n_channels = int_shape(z)
    assert n_channels >= 2 and n_channels % 2 == 0
    z = tf.reshape(z, [-1, length, n_channels // factor, factor])
    z = tf.transpose(z, [0, 1, 3, 2])
    z = tf.reshape(z, [-1, length * factor, n_channels // factor])
    return z

def gaussian_diag(mean, logsd):
    class o(object):
        pass
    o.mean = mean
    o.logsd = logsd
    o.sample = lambda: mean + tf.exp(logsd) * tf.random.normal(tf.shape(o.mean))
    o.logps = lambda z: -0.5 * \
        (np.log(2 * np.pi) + 2. * logsd + (z - mean) ** 2 / tf.exp(2. * logsd))
    o.logp = lambda z: tf.reduce_sum(o.logps(z), [1, 2])
    o.get_eps = lambda z: (z - mean) / tf.exp(logsd)
    return o
