import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

import tfops_short as Z
import optim_short as optim

def abstract_model(sess, hps, feed, train_iterator, data_init, lr, f_loss):
    # == Create class with static fields and methods
    class m(object):
        pass
    m.sess = sess
    m.X = feed['x']
    m.lr = lr

    # === Loss and optimizer
    loss_train, stats_train = f_loss(train_iterator, reuse=False)
    all_params = tf.compat.v1.trainable_variables()
    gs = tf.gradients(loss_train, all_params)

    train_op, polyak_swap_op, ema = optim.adamax(all_params, gs, alpha=lr, hps=hps)
    
    m.train = lambda _lr: sess.run([train_op, stats_train], {lr: _lr})[1]
    m.polyak_swap = lambda: sess.run(polyak_swap_op)

    # === Saving and restoring
    saver = tf.compat.v1.train.Saver()
    saver_ema = tf.compat.v1.train.Saver(ema.variables_to_restore())
    m.save_ema = lambda path: saver_ema.save(sess, path, write_meta_graph=False)
    m.save = lambda path: saver.save(sess, path, write_meta_graph=False)
    m.restore = lambda path: saver.restore(sess, path)
    
    # === Initialize
    with Z.arg_scope([Z.get_variable_ddi, Z.actnorm], init=True):
        results_init = f_loss(None, reuse=True)
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(results_init, {feed['x']: data_init})

    return m

def model(sess, hps, train_iterator, data_init):
    # Only for decoding/init, rest use iterators directly
    with tf.name_scope('input'):
        X = tf.compat.v1.placeholder(
            tf.float32, [None, 40000, 1], name='spectrum')
        lr = tf.compat.v1.placeholder(tf.float32, None, name='learning_rate')
        
    def encoder(z, objective):
        eps = []
        for i in range(hps.n_levels):
            hps.objectives.append(tf.reduce_mean(objective))
            hps.log.append("encoder level "+str(i))
            hps.z_means.append(tf.reduce_mean(z))
            hps.z_stds.append(tf.math.reduce_std(z))
            print("creating revnet level", i, z.get_shape())
            z, objective = revnet2d("revnet"+str(i), z, objective, hps)
            if i < hps.n_levels-1:
                print("splitting z", z.get_shape())
                z, objective, _eps = split1d("pool"+str(i), z, objective=objective)
                hps.objectives.append(tf.reduce_mean(objective))
                hps.log.append("end of encoder level "+str(i))
                eps.append(_eps)
        hps.z_means.append(tf.reduce_mean(z))
        hps.z_stds.append(tf.math.reduce_std(z))
        return z, objective, eps

    def decoder(z, eps=[None]*hps.n_levels, eps_std=None):
        for i in reversed(range(hps.n_levels)):
            if i < hps.n_levels-1:
                z = split1d_reverse("pool"+str(i), z, eps=eps[i], eps_std=eps_std)
            z, _ = revnet2d("revnet"+str(i), z, 0, hps, reverse=True)
        return z

    def f_loss(iterator, reuse):
        #x = iterator.get_next() if iterator is not None else X
        x = iterator if iterator is not None else X
        print("original shape", x.get_shape())
        with tf.compat.v1.variable_scope('model', reuse=reuse):
            objective = tf.zeros_like(x, dtype='float32')[:, 0, 0]
            z = x - .5 # map to centering around 0?
            hps.log = []
            hps.objectives = []
            hps.logpz = []
            hps.z_means = []
            hps.z_stds = []
            # Encode
            z = Z.squeeze(z, 4) # i guess we start out with a squeeze?
            print("encoding z", z.get_shape())
            z, objective, _ = encoder(z, objective)
            
            # Prior
            print("calculating prior", z.get_shape())
            hps.final_z = z
            hps.top_shape = Z.int_shape(z)[1:]
            logp, _, _ = prior("prior", hps.batch_size, hps)
            objective += logp(z)
            hps.logpz.append(tf.reduce_mean(logp(z)))
            hps.objectives.append(tf.reduce_mean(objective))
            hps.log.append("post prior")

            nobj = -objective
            bits_x = nobj / (np.log(2.) * int(x.get_shape()[1]) * int(x.get_shape()[2]))  # bits per subpixel

        local_loss = bits_x
        stats = [local_loss]
        global_stats = tf.stack([tf.reduce_mean(local_loss), -1, *hps.objectives, -1, *hps.logpz])
        #print(global_stats.get_shape(), global_stats)
        #print(hps.objectives.get_shape(), hps.objective)
#        global_stats = Z.allreduce_mean(
#            tf.stack([tf.reduce_mean(i) for i in stats])) # need to figure out what this is...
        
        return tf.reduce_mean(local_loss), global_stats

    feed = {'x': X}
    m = abstract_model(sess, hps, feed, train_iterator, data_init, lr, f_loss)
    
    def encode(x):
        with tf.variable_scope('model', reuse=True):
            objective = tf.zeros_like(x, dtype='float32')[:, 0, 0]
            z = x - .5
            z = Z.squeeze(z, 4)
            z, _, _, = encoder(z, objective)
            return z
    m.encode = encode

    def decode(z):
        with tf.variable_scope('model', reuse=True):
            z = decoder(z)
            z = Z.unsqueeze(z, 4)
            x = z + .5
            return x
    m.decode = decode

    # Sampling, encoding, and decoding (leave to implement later)
    # === Sampling function
    '''def f_sample(eps_std):
        with tf.variable_scope('model', reuse=True):
            _, sample, _ = prior("prior", 1, hps)
            z = sample(eps_std=eps_std)
            x = decoder(z, eps_std=eps_std)
            #z = Z.unsqueeze2d(z, 2)  # 8x8x12 -> 16x16x3
            #x = postprocess(z)
        return x

    m.eps_std = tf.placeholder(tf.float32, [None], name='eps_std')
    x_sampled = f_sample(m.eps_std)
    m.sample = lambda _eps_std: m.sess.run(x_sampled, {m.eps_std: _eps_std})'''
    
    return m

def checkpoint(z, logdet):
    # save z and logdet to checkpoints. i think they did a weird thing to maybe save memory??
    batch_size, length, n_channels = Z.int_shape(z)
    z_save = tf.reshape(z, [-1, length * n_channels])
    logdet_save = tf.reshape(logdet, [-1, 1])
    combined = tf.concat([z_save, logdet_save], axis=1)
    tf.compat.v1.add_to_collection('checkpoints', combined)

@add_arg_scope
def revnet2d(name, z, logdet, hps, reverse=False):
    with tf.compat.v1.variable_scope(name):
        if not reverse:
            for i in range(hps.depth):
                #checkpoint(z, logdet)
                z, logdet = revnet2d_step("revnetstep"+str(i), z, logdet, hps, reverse)
            #checkpoint(z, logdet)
        else:
            for i in reversed(range(hps.depth)):
                z, logdet = revnet2d_step("revnetstep"+str(i), z, logdet, hps, reverse)
    return z, logdet

@add_arg_scope
def revnet2d_step(name, z, logdet, hps, reverse):
    with tf.compat.v1.variable_scope(name):
        batch_size, length, n_channels = Z.int_shape(z)
        assert n_channels % 2 == 0
        if not reverse:
            z, logdet = Z.actnorm("actnorm", z, logdet=logdet, hps=hps)
            hps.objectives.append(tf.reduce_mean(logdet))
            hps.log.append(name+"post-actnorm")
            z, logdet = invertible_1x1_conv("invconv", z, logdet)
            hps.objectives.append(tf.reduce_mean(logdet))
            hps.log.append(name+"post-1x1conv")
            z1 = z[:, :, :n_channels // 2]
            z2 = z[:, :, n_channels // 2:]
            z2 += f("f1", z1, hps.width)
            z = tf.concat([z1, z2], 2)
        else:
            z1 = z[:, :, :n_channels // 2]
            z2 = z[:, :, n_channels // 2:]
            z2 -= f("f1", z1, hps.width)
            z = tf.concat([z1, z2], 2)
            z, logdet = invertible_1x1_conv("invconv", z, logdet, reverse)
            z, logdet = Z.actnorm("actnorm", z, logdet=logdet, reverse=reverse)
            
    return z, logdet

@add_arg_scope
def invertible_1x1_conv(name, z, logdet, reverse=False):
    with tf.compat.v1.variable_scope(name):
        batch_size, length, n_channels = Z.int_shape(z)
        w_shape = [n_channels, n_channels]

        # Sample a random orthogonal matrix:
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
        w = tf.compat.v1.get_variable("W", dtype=tf.float32, initializer=w_init)

        #dlogdet = tf.linalg.LinearOperator(w).log_abs_determinant() * length
        dlogdet = tf.cast(tf.math.log(abs(tf.linalg.det(tf.cast(w, 'float64')))), 'float32') * length

        if not reverse:
            _w = tf.reshape(w, [1] + w_shape)
            z = tf.nn.conv1d(z, _w, [1, 1, 1], "SAME")
            logdet += dlogdet
        else:
            _w = tf.reshape(tf.matrix_inverse(w), [1]+w_shape)
            z = tf.nn.conv1d(z, _w, [1, 1, 1], "SAME")
            logdet -= dlogdet
            
        return z, logdet
    
def f(name, h, width, n_out=None):
    n_out = n_out or int(h.get_shape()[2])
    with tf.compat.v1.variable_scope(name):
        h = tf.nn.relu(Z.conv1d("l_1", h, width, filter_size=[3]))
        h = tf.nn.relu(Z.conv1d("l_2", h, width, filter_size=[1]))
        h = Z.conv1d_zeros("l_last", h, n_out, filter_size=[3])
    return h

@add_arg_scope
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
    
@add_arg_scope
def split1d_reverse(name, z, eps, eps_std):
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

    
@add_arg_scope
def split1d_prior(z):
    n_channels = int(z.get_shape()[2])
    h = Z.conv1d_zeros("conv", z, 2 * n_channels, filter_size=[3]) # again just for learning prior?

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
