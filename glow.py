import numpy as np
import tensorflow as tf

import tfops as Z

class model:
    def __init__(self, sess, hps, train_iterator):
        # === Define session
        self.sess = sess
        self.hps = hps

        # === Input tensors
        with tf.name_scope('input'):
            s_shape = [None, hps.n_bins, 1]
            self.s_placeholder = tf.compat.v1.placeholder(tf.float32, s_shape, name='spectra')

            self.lr_placeholder = tf.compat.v1.placeholder(tf.float32, None, name='learning_rate')

            self.train_iterator = train_iterator

            z_shape = [None, hps.n_bins//2**(hps.n_levels+1), 4]
            self.z_placeholder = tf.compat.v1.placeholder(tf.float32, z_shape, name='latent_rep')

            intermediate_z_shapes = [[None, hps.n_bins//2**(i+1), 2] for i in range(1, hps.n_levels)]
            self.intermediate_z_placeholders = [
                tf.compat.v1.placeholder(tf.float32, shape)
                for shape in intermediate_z_shapes
            ]

        # === Loss and optimizer
        self.training_op, self.training_ops, self.loss, self.stats = self._create_optimizer()

        # === Encoding and decoding
        self.z, self.logpx, self.intermediate_zs = self._create_encoder(self.s_placeholder)
        self.s_from_intermediate_zs = self._create_decoder(self.z_placeholder,
                                                           self.intermediate_z_placeholders)

        # === Initialize
        sess.run(tf.compat.v1.global_variables_initializer()) # not sure if more initialization is needed?

        # === Saving and restoring
        with tf.device('/cpu:0'):
            saver = tf.compat.v1.train.Saver()
            self.save = lambda path: saver.save(sess, path, write_meta_graph=False)
            self.restore = lambda path: saver.restore(sess, path)

        # === Description
        hps.spectra_shape = (hps.n_bins, 1)
        hps.level_shapes = [(hps.n_bins // 4, 4)]
        hps.intermediate_z_shapes = []
        for i in range(hps.n_levels - 1):
            previous_shape = hps.level_shapes[-1]
            hps.intermediate_z_shapes.append((previous_shape[0], 2))
            hps.level_shapes.append((previous_shape[0] // 2, 4))
        hps.latent_rep_shape = hps.level_shapes[-1]
        self.print_short_description()

    def _create_optimizer(self):
        '''Set up optimizer to train on input train_iterator and learning rate.'''
        _, logpx, _ = self._create_encoder(self.train_iterator)
        bits_x = -logpx / (np.log(2.) * self.hps.n_bins)  # bits per subpixel

        with tf.compat.v1.variable_scope('optimizer', reuse=tf.compat.v1.AUTO_REUSE):
            loss = tf.reduce_mean(bits_x)
            stats = tf.stack([tf.reduce_mean(loss)])
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr_placeholder)

            training_op = optimizer.minimize(loss)

            level_training_ops = []
            for i in range(self.hps.n_levels):
                scope = "model/level{}/".format(i)
                level_vars = tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope
                )
                level_training_ops.append(optimizer.minimize(loss, var_list=level_vars))

        return training_op, level_training_ops, loss, stats

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
        z = Z.squeeze(x, 4) # pre-process
        with tf.compat.v1.variable_scope('model', reuse=tf.compat.v1.AUTO_REUSE):
            for i in range(self.hps.n_levels):
                with tf.compat.v1.variable_scope('level{}'.format(i)):
                    for j in range(self.hps.level_depths[i]):
                        z, logpx = self._flow_step('depth{}'.format(j), z, logpx)
                    if i < self.hps.n_levels - 1:
                        z1, z2 = Z.split(z)
                        intermediate_prior = self._create_prior(z2)
                        logpx += intermediate_prior.logp(z2)
                        intermediate_zs.append(z2)
                        z = Z.squeeze(z1, 2)
            prior = self._create_prior(z)
            logpx += prior.logp(z)
            return z, logpx, intermediate_zs

    def _create_decoder(self, z, intermediate_zs):
        '''Set up decoder tensors to generate spectra from latent representation.

        Args:
            z: tensor where shape matches final latent representation.
            intermediate_zs: optional list of tensors, components removed during encoder splits.

        Returns:
            x: tensor with shape [?, n_bins, 1], spectra constructed from z.
        '''
        with tf.compat.v1.variable_scope('model', reuse=tf.compat.v1.AUTO_REUSE):
            for i in reversed(range(self.hps.n_levels)):
                with tf.compat.v1.variable_scope('level{}'.format(i)):
                    if i < self.hps.n_levels - 1:
                        z1 = Z.unsqueeze(z, 2)
                        z2 = intermediate_zs[i]
                        z = Z.unsplit(z1, z2)
                    for j in reversed(range(self.hps.level_depths[i])):
                        z = self._reverse_flow_step('depth{}'.format(j), z)
            x = Z.unsqueeze(z, 4) # post-process
            return x

    def _flow_step(self, name, z, logdet):
        with tf.compat.v1.variable_scope(name):
            z, logdet = Z.actnorm('actnorm', z, logdet)
            z, logdet = Z.invertible_1x1_conv('invconv', z, logdet, reverse=False)
            z1, z2 = Z.split(z)
            z2 += Z.f('f', z1, self.hps.window_size, self.hps.width)
            z = Z.unsplit(z1, z2)
            return z, logdet

    def _reverse_flow_step(self, name, z):
        with tf.compat.v1.variable_scope(name):
            z1, z2 = Z.split(z)
            z2 -= Z.f('f', z1, self.hps.window_size, self.hps.width)
            z = Z.unsplit(z1, z2)
            z, _ = Z.invertible_1x1_conv('invconv', z, 0, reverse=True)
            z = Z.actnorm_reverse('actnorm', z)
            return z

    def _create_prior(self, z):
        '''Create a unit normal Gaussian object with same shape as z.'''
        mu = tf.zeros_like(z, dtype='float32')
        logs = tf.zeros_like(z, dtype='float32')
        return Z.gaussian_diag(mu, logs)

    def create_peak_remover(self, window, original):
        # setup
        original_left, original_right = original[:, :window[0]], original[:, window[1]:]
        left, mid, right = (
            self.s_from_intermediate_zs[:, :window[0]],
            self.s_from_intermediate_zs[:, window[0]:window[1]],
            self.s_from_intermediate_zs[:, window[1]:],
        )
        gaussian_kernel = Z.create_gaussian_kernel(size=29, mean=0, std=7)
        derivative_kernel = tf.constant([[[-self.hps.n_bins / 2]], [[0]], [[self.hps.n_bins / 2]]])

        # outside window
        left_squared_error = tf.reduce_sum((left - original_left)**2)
        right_squared_error = tf.reduce_sum((right - original_right)**2)
        outside_cost = left_squared_error + right_squared_error

        # inside window
        smooth = tf.nn.conv1d(mid, gaussian_kernel, padding="SAME")
        first_derivative = tf.nn.conv1d(smooth, derivative_kernel, padding="SAME")
        smooth_first_derivative = tf.nn.conv1d(first_derivative, gaussian_kernel, padding="SAME")
        second_derivative = tf.nn.conv1d(smooth_first_derivative, derivative_kernel, padding="SAME")
        inside_cost = -tf.reduce_sum(second_derivative**2)

        # likelihood
        logp = -.5 * tf.reduce_sum(self.z_placeholder**2) # TODO: use true likelihood

        # maximize inside cost and likelihood. minimize outside cost
        cost = 1e-7 * inside_cost - 0 * outside_cost + 1.1e7 * logp

        self.remove_peak_spectrum = original
        _, self.remove_peak_intermediate_zs = self.encode(original)
        self.remove_peak_gradient = tf.gradients(cost, self.z_placeholder)

    def remove_peak(self, z, step_size):
        feed_dict = {self.z_placeholder: z}
        for i in range(len(self.remove_peak_intermediate_zs)):
            feed_dict[self.intermediate_z_placeholders[i]] = self.remove_peak_intermediate_zs[i]
        gradient = self.sess.run(self.remove_peak_gradient, feed_dict)[0]
        scale_factor = step_size / np.linalg.norm(gradient)
        new_z = z + scale_factor * gradient
        return new_z, gradient

    def train(self, lr, level=None):
        '''Run one training batch to optimize the network with learning rate lr.

        Returns:
            stats: statistics created in _create_optimizer. probably contains loss.
        '''
        training_op = self.training_op if level is None else self.training_ops[level]
        _, stats = self.sess.run([training_op, self.stats], {self.lr_placeholder: lr})
        return stats

    def encode(self, s):
        return self.sess.run([self.z, self.intermediate_zs], {self.s_placeholder: s})

    def decode(self, z, intermediate_zs=None):
        '''Decode a latent representation with optional intermediate components.

        Args:
            z: latent representation with shape [?, *self.hps.latent_rep_shape]
            intermediate_zs: list of intermediate zs. If a subset is provided (must be smaller
                             intermediate zs), the rest are randomly sampled.

        Returns:
            spectra, from z and intermediate zs. If no intermediate zs are provided, sample them
            randomly from unit normal distributions.
        '''
        feed_dict = {self.z_placeholder: z}
        num_input_zs = len(intermediate_zs) if intermediate_zs is not None else 0
        num_total_zs = len(self.hps.intermediate_z_shapes)
        for i in range(num_total_zs):
            if i < num_total_zs - num_input_zs:
                sample_shape = (z.shape[0], *self.hps.intermediate_z_shapes[i])
                sample = np.random.normal(0, 1, sample_shape)
                feed_dict[self.intermediate_z_placeholders[i]] = sample
            else:
                index = i - num_total_zs + len(intermediate_zs)
                feed_dict[self.intermediate_z_placeholders[i]] = intermediate_zs[index]
        return self.sess.run(self.s_from_intermediate_zs, feed_dict)

    def get_likelihood(self, s):
        return self.sess.run(self.logpx, {self.s_placeholder: s})

    def print_short_description(self):
        print(" in:", self.hps.spectra_shape)
        for i in range(self.hps.n_levels - 1):
            print("l{:2}:".format(i), self.hps.level_shapes[i], ">", self.hps.intermediate_z_shapes[i])
        print("l{:2}:".format(self.hps.n_levels - 1), self.hps.level_shapes[-1])
        print("fin:", self.hps.level_shapes[-1])

    def print_full_description(self):
        print("Input spectra shapes:", self.hps.spectra_shape)
        print("Initially squeeze to", self.hps.level_shapes[0])
        for i in range(self.hps.n_levels - 1):
            print("l{:2}:".format(i), "\t", self.hps.level_shapes[i])
            print("\t   v")
            print("\t", self.hps.level_depths[i], "flow steps")
            print("\t   v")
            print("\t split -> int rep:", self.hps.intermediate_z_shapes[i])
            print("\t   v")
            print("\t squeeze:", self.hps.intermediate_z_shapes[i], ">", self.hps.level_shapes[i+1])
            print("\t   v")
        print("l{:2}:".format(self.hps.n_levels - 1), "\t", self.hps.level_shapes[-1])
        print("\t   v")
        print("\t", self.hps.level_depths[-1], "flow steps")
        print("\t   v")
        print("fin rep:", self.hps.level_shapes[-1])
