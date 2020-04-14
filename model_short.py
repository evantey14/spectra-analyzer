import numpy as np
import tensorflow as tf

import tfops_short as Z

class model:
    def __init__(self, sess, hps, train_iterator, data_init):
        # === Define session
        self.sess = sess
        self.hps = hps

        # === Input tensors
        with tf.name_scope('input'):
            s_shape = [None, hps.n_bins, 1]
            self.s_placeholder = tf.compat.v1.placeholder(tf.float32, s_shape, name='spectra')

            self.lr_placeholder = tf.compat.v1.placeholder(tf.float32, None, name='learning_rate')

            self.train_iterator = train_iterator

            z_shape = [None, hps.n_bins/2**(hps.n_levels+1), 4]
            self.z_placeholder = tf.compat.v1.placeholder(tf.float32, z_shape, name='latent_rep')

            intermediate_z_shapes = [[None, hps.n_bins/2**(i+1), 2] for i in range(1, hps.n_levels)]
            self.intermediate_z_placeholders = [
                tf.compat.v1.placeholder(tf.float32, shape)
                for shape in intermediate_z_shapes
            ]

        # === Loss and optimizer
        self.optimizer, self.loss, self.stats = self._create_optimizer()

        # === Encoding and decoding
        self.z, self.logpx, self.intermediate_zs = self._create_encoder(self.s_placeholder)
        self.s = self._create_decoder(self.z_placeholder)
        self.s_from_intermediate_zs = self._create_decoder(self.z_placeholder,
                                                          self.intermediate_z_placeholders)

        # === Initialize
        sess.run(tf.compat.v1.global_variables_initializer()) # not sure if more initialization is needed?

        # === Saving and restoring
        with tf.device('/cpu:0'):
            saver = tf.compat.v1.train.Saver()
            self.save = lambda path: saver.save(sess, path, write_meta_graph=False)
            self.restore = lambda path: saver.restore(sess, path)

    def _create_optimizer(self):
        '''Set up optimizer to train on input train_iterator and learning rate.'''
        _, logpx, _ = self._create_encoder(self.train_iterator)
        bits_x = -logpx / (np.log(2.) * self.hps.n_bins)  # bits per subpixel

        with tf.compat.v1.variable_scope('optimizer', reuse=tf.compat.v1.AUTO_REUSE):
            loss = tf.reduce_mean(bits_x)
            stats = tf.stack([tf.reduce_mean(loss)])
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder).minimize(loss)

        return optimizer, loss, stats

    def _create_encoder(self, x):
        '''Set up encoder tensors to pipe input spectra x to a latent representation

        Args:
            x: input tensor with shape [?, n_bins, 1], either a placeholder or data stream

        Returns:
            z: output tensor, contains the fully compressed latent representation
            logpx: tensor with shape [?,], the log likelihood of each spectrum
            intermediate_zs: list of tensors, the components dropped after splits
        '''
        logpx = tf.zeros_like(x, dtype='float32')[:, 0, 0] # zeros tensor with shape (batch_size)
        intermediate_zs = []
        z = Z.squeeze(x - .5, 4) # preprocess the input
        with tf.compat.v1.variable_scope('model', reuse=tf.compat.v1.AUTO_REUSE):
            for i in range(self.hps.n_levels):
                for j in range(self.hps.depth):
                    z, logpx = self._flow_step('flow-level{}-depth{}'.format(i, j), z, logpx)
                if i < self.hps.n_levels - 1:
                    z1, z2 = self._split(z)
                    intermediate_prior = self._create_prior(z2)
                    logpx += intermediate_prior.logp(z2)
                    intermediate_zs.append(z2)
                    z = Z.squeeze(z1, 2)
            prior = self._create_prior(z)
            logpx += prior.logp(z)
            return z, logpx, intermediate_zs

    def _create_decoder(self, z, intermediate_zs=None):
        '''Set up decoder tensors to generate spectra from latent representation.

        Args:
            z: tensor where shape matches final latent representation.
            intermediate_zs: optional list of tensors, components removed during encoder splits.

        Returns:
            x: tensor with shape [?, n_bins, 1], spectra constructed from z.
        '''
        with tf.compat.v1.variable_scope('model', reuse=tf.compat.v1.AUTO_REUSE):
            for i in reversed(range(self.hps.n_levels)):
                if i < self.hps.n_levels - 1:
                    z1 = Z.unsqueeze(z, 2)
                    if intermediate_zs is None:
                        intermediate_prior = self._create_prior(z1)
                        z2 = intermediate_prior.sample()
                    else:
                        z2 = intermediate_zs[i]
                    z = self._unsplit(z1, z2)
                for j in reversed(range(self.hps.depth)):
                    z, _= self._reverse_flow_step('flow-level{}-depth{}'.format(i, j), z)
            x = Z.unsqueeze(z + .5, 4) # post-process spectra
            return x

    # move to tfops
    def _split(self, z):
        '''Split a (batch_size, length, n_channels) tensor in half across the channel dimension.'''
        assert int(z.get_shape()[2]) == 4
        return z[:, :, :2], z[:, :, 2:]

    # move to tfops
    def _unsplit(self, z1, z2):
        '''Recombine two tensors across the channel dimension.'''
        return tf.concat([z1, z2], 2)

    def _flow_step(self, name, z, logdet):
        with tf.compat.v1.variable_scope(name):
            z, logdet = Z.actnorm('actnorm', z, logdet=logdet, reverse=False)
            z, logdet = self._invertible_1x1_conv('invconv', z, logdet, reverse=False)
            z1, z2 = self._split(z)
            z2 += self._f('f1', z1, self.hps.width)
            z = self._unsplit(z1, z2)
            return z, logdet

    def _reverse_flow_step(self, name, z):
        with tf.compat.v1.variable_scope(name):
            z1, z2 = self._split(z)
            z2 -= self._f('f1', z1, self.hps.width)
            z = self._unsplit(z1, z2)
            z, logdet = self._invertible_1x1_conv('invconv', z, 0, reverse=True)
            z, logdet = Z.actnorm('actnorm', z, logdet=0, reverse=True)
            return z, logdet

    # move to tfops
    def _invertible_1x1_conv(self, name, z, logdet, reverse=False):
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
                #z = tf.nn.conv1d(z, _w, [1, 1, 1], 'SAME')
                z = tf.nn.conv1d(z, _w, 1, 'SAME')
                logdet += dlogdet
            else:
                _w = tf.reshape(tf.matrix_inverse(w), [1]+w_shape)
                #z = tf.nn.conv1d(z, _w, [1, 1, 1], 'SAME')
                z = tf.nn.conv1d(z, _w, 1, 'SAME')
                logdet -= dlogdet

            return z, logdet

    # move to tfops
    def _f(self, name, h, width, n_out=None):
        n_out = n_out or int(h.get_shape()[2])
        with tf.compat.v1.variable_scope(name):
            h = tf.nn.relu(Z.conv1d('l_1', h, width, filter_size=[3]))
            h = tf.nn.relu(Z.conv1d('l_2', h, width, filter_size=[1]))
            h = Z.conv1d_zeros('l_last', h, n_out, filter_size=[3])
        return h

    def _create_prior(self, z):
        '''Create a Gaussian object that makes the shape of z.'''
        mu = tf.zeros_like(z, dtype='float32')
        logs = tf.zeros_like(z, dtype='float32')
        return Z.gaussian_diag(mu, logs)

    def train(self, lr):
        '''Run one training batch to optimize the network.
        
        Args:
            lr: float, learning rate

        Returns:
            stats: statistics created in _create_optimizer. probably contains loss.
        '''
        _, stats = self.sess.run([self.optimizer, self.stats], {self.lr_placeholder: lr})
        return stats

    def encode(self, s):
        return self.sess.run([self.z, self.intermediate_zs], {self.s_placeholder: s})

    def decode(self, z, intermediate_zs=None):
        feed_dict = {self.z_placeholder: z}
        if intermediate_zs is None:
            return self.sess.run(self.s, feed_dict)
        else:
            for i in range(len(intermediate_zs)):
                feed_dict[self.intermediate_z_placeholders[i]] = intermediate_zs[i]
            return self.sess.run(self.s_from_intermediate_zs, feed_dict)

    def get_likelihood(self, s):
        return self.sess.run(self.logpx, {self.s_placeholder: s})
