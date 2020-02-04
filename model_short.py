import numpy as np
import tensorflow as tf

import tfops_short as Z
import optim_short as optim

class model:
    def __init__(self, sess, hps, train_iterator, data_init):
        # === Define session
        self.sess = sess

        # === Input tensors
        with tf.name_scope('input'):
            self.input_placeholder = tf.compat.v1.placeholder(tf.float32, [None, hps.n_bins, 1],
                                                              name='spectrum')
            self.lr = tf.compat.v1.placeholder(tf.float32, None, name='learning_rate')
            self.train_iterator = train_iterator
            self.z_placeholder = tf.compat.v1.placeholder(tf.float32, [None, hps.n_bins /
                2**(hps.n_levels + 1), 4], name='latent_rep') # after squeeze & n_levels-1 splits
            print(self.z_placeholder)

        # === Loss and optimizer
        self.loss, self.stats = self.f_loss(self.train_iterator, hps)
        all_params = tf.compat.v1.trainable_variables()
        gradients = tf.gradients(self.loss, all_params)
        train_op, polyak_swap_op, ema = optim.adamax(all_params, gradients, alpha=self.lr, hps=hps)
        
        self.train = lambda _lr: self.sess.run([train_op, self.stats], {self.lr: _lr})[1]
        self.polyak_swap = lambda: self.sess.run(polyak_swap_op)

        # === Encoding and decoding
        encode_op, _ = self._create_encoder(self.input_placeholder, hps)
        self.encode = lambda spectra: self.sess.run(encode_op, {self.input_placeholder: spectra})
        decode_op = self._create_decoder(self.z_placeholder, hps)
        self.decode = lambda z: self.sess.run(decode_op, {self.z_placeholder: z})
        reconstruct_op = self._create_decoder(encode_op, hps)
        self.reconstruct = lambda spectra: self.sess.run(reconstruct_op, {self.input_placeholder: spectra})

        # === Initialize
        # not sure if more initialization is needed?
        sess.run(tf.compat.v1.global_variables_initializer())

        # === Saving and restoring
        # not entirely sure what this does / if it works
        saver = tf.compat.v1.train.Saver()
        saver_ema = tf.compat.v1.train.Saver(ema.variables_to_restore())
        self.save_ema = lambda path: saver_ema.save(sess, path, write_meta_graph=False)
        self.save = lambda path: saver.save(sess, path, write_meta_graph=False)
        self.restore = lambda path: saver.restore(sess, path)

    def _create_encoder(self, x, hps):
        '''Set up encoder tensors to pipe input spectra x to a latent representation

        Args:
            x: input tensor with shape [?, n_bins, 1], either a placeholder or data stream
            hps: a config class

        Returns:
            z: output tensor, contains the fully compressed latent representation
            logpx: tensor with shape [?,], the log likelihood of each spectrum
        '''
        # Preprocess input x and set up log probability tensor where we'll accumulate log p(x) as x
        # gets transformed through the network.
        z = x - .5
        z = Z.squeeze(z, 4)
        logpx = tf.zeros_like(x, dtype='float32')[:, 0, 0]
        with tf.compat.v1.variable_scope('model', reuse=tf.AUTO_REUSE):
            # Compress latent representations with revnets and splits
            for i in range(hps.n_levels):
                z, logpx = revnet2d('revnet' + str(i), z, logpx, hps)
                if i < hps.n_levels - 1:
                    z, logpx, _ = split1d('pool' + str(i), z, objective=logpx)

            # Compute the prior on the final latent representation.
            hps.top_shape = Z.int_shape(z)[1:]
            logp, _, _ = prior('prior', hps.batch_size, hps)
            logpx += logp(z)
            return z, logpx

    def _create_decoder(self, z, hps):
        ''' Set up decoder tensors fo generate spectra from latent representation.
        
        Args:
            z: tensor where shape matches final latent representation.
            hps: a config class

        Returns:
            x: tensor with shape [?, n_bins, 1]
        '''
        with tf.compat.v1.variable_scope('model', reuse=tf.AUTO_REUSE):
            for i in reversed(range(hps.n_levels)):
                if i < hps.n_levels - 1:
                    z = split1d_reverse('pool' + str(i), z)
                z, _ = revnet2d('revnet' + str(i), z, 0, hps, reverse=True)
        z = Z.unsqueeze(z, 4)
        x = z + .5
        return x

    def f_loss(self, x, hps):
        z, logpx = self._create_encoder(x, hps)
        bits_x = -logpx / (np.log(2.) * int(x.get_shape()[1]) * int(x.get_shape()[2]))  # bits per subpixel

        local_loss = bits_x
        stats = tf.stack([tf.reduce_mean(local_loss)])

        return tf.reduce_mean(local_loss), stats

def revnet2d(name, z, logdet, hps, reverse=False):
    with tf.compat.v1.variable_scope(name):
        if not reverse:
            for i in range(hps.depth):
                z, logdet = revnet2d_step('revnetstep'+str(i), z, logdet, hps, reverse)
        else:
            for i in reversed(range(hps.depth)):
                z, logdet = revnet2d_step('revnetstep'+str(i), z, logdet, hps, reverse)
    return z, logdet

def revnet2d_step(name, z, logdet, hps, reverse):
    with tf.compat.v1.variable_scope(name):
        batch_size, length, n_channels = Z.int_shape(z)
        assert n_channels % 2 == 0
        if not reverse:
            z, logdet = Z.actnorm('actnorm', z, logdet=logdet, hps=hps)
            z, logdet = invertible_1x1_conv('invconv', z, logdet)
            z1 = z[:, :, :n_channels // 2]
            z2 = z[:, :, n_channels // 2:]
            z2 += f('f1', z1, hps.width)
            z = tf.concat([z1, z2], 2)
        else:
            z1 = z[:, :, :n_channels // 2]
            z2 = z[:, :, n_channels // 2:]
            z2 -= f('f1', z1, hps.width)
            z = tf.concat([z1, z2], 2)
            z, logdet = invertible_1x1_conv('invconv', z, logdet, reverse)
            z, logdet = Z.actnorm('actnorm', z, logdet=logdet, reverse=reverse)
            
    return z, logdet

def invertible_1x1_conv(name, z, logdet, reverse=False):
    with tf.compat.v1.variable_scope(name):
        batch_size, length, n_channels = Z.int_shape(z)
        w_shape = [n_channels, n_channels]

        # Sample a random orthogonal matrix:
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
        w = tf.compat.v1.get_variable('W', dtype=tf.float32, initializer=w_init)

        #dlogdet = tf.linalg.LinearOperator(w).log_abs_determinant() * length
        dlogdet = tf.cast(tf.math.log(abs(tf.linalg.det(tf.cast(w, 'float64')))), 'float32') * length

        if not reverse:
            _w = tf.reshape(w, [1] + w_shape)
            z = tf.nn.conv1d(z, _w, [1, 1, 1], 'SAME')
            logdet += dlogdet
        else:
            _w = tf.reshape(tf.matrix_inverse(w), [1]+w_shape)
            z = tf.nn.conv1d(z, _w, [1, 1, 1], 'SAME')
            logdet -= dlogdet
            
        return z, logdet
    
def f(name, h, width, n_out=None):
    n_out = n_out or int(h.get_shape()[2])
    with tf.compat.v1.variable_scope(name):
        h = tf.nn.relu(Z.conv1d('l_1', h, width, filter_size=[3]))
        h = tf.nn.relu(Z.conv1d('l_2', h, width, filter_size=[1]))
        h = Z.conv1d_zeros('l_last', h, n_out, filter_size=[3])
    return h

def split1d(name, z, objective=0.):
    with tf.compat.v1.variable_scope(name):
        n_z = Z.int_shape(z)[2]
        z1 = z[:, :, :n_z // 2]
        z2 = z[:, :, n_z // 2:]
        pz = split1d_prior(z1)
        objective += pz.logp(z2)
        
        z1 = Z.squeeze(z1, 2)
        eps = pz.get_eps(z2)
        return z1, objective, eps
    
def split1d_reverse(name, z, eps=None, eps_std=None):
    with tf.variable_scope(name):
        z1 = Z.unsqueeze(z, 2)
        pz = split1d_prior(z1)
        if eps is not None:
            # Already sampled eps
            z2 = pz.sample2(eps)
        elif eps_std is not None:
            # Sample with given eps_std
            z2 = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1]))
        else:
            # Sample normally
            z2 = pz.sample
        z = tf.concat([z1, z2], 2)
        return z
    
def split1d_prior(z):
    n_channels = int(z.get_shape()[2])
    h = Z.conv1d_zeros('conv', z, 2 * n_channels, filter_size=[3]) # again just for learning prior?

    mean = h[:, :, 0::2]
    logs = h[:, :, 1::2]
    return Z.gaussian_diag(mean, logs)

def prior(name, batch_size, hps):
    with tf.compat.v1.variable_scope(name):
        length, n_channels = hps.top_shape
        h = tf.zeros([batch_size, length, 2 * n_channels])
        #h = Z.conv1d_zeros('p', h, 2 * n_channels, filter_size=[3]) # for learning prior?
        pz = Z.gaussian_diag(h[:, :, :n_channels], h[:, :, n_channels:])

    def logp(z1):
        return pz.logp(z1)
        
    def sample(eps=None, eps_std=None):
        if eps is not None: # Already sampled eps. Don't use eps_std
            z = pz.sample2(eps)
        elif eps_std is not None: # Sample with given eps_std
            z = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1]))
        else: # Sample normally
            z = pz.sample
        return z

    def eps(z1):
        return pz.get_eps(z1)

    return logp, sample, eps
