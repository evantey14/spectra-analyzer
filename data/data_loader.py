import numpy as np
import tensorflow as tf
import tftables

def add_noise(spectra):
    '''Add counting noise to a spectra batch.'''
    shape = tf.shape(spectra)
    sqrt = tf.sqrt(spectra)
    sums = tf.reduce_sum(spectra, axis=1)
    sqrtsums = tf.reduce_sum(sqrt, axis=1)
    As = .02 * sums / (np.sqrt(2 / 3.14) * sqrtsums)
    expanded_As = tf.tile(As, [shape[1]])
    reshaped_As = tf.reshape(expanded_As, [shape[0], shape[1]])
    noise = tf.random.normal(shape, stddev=reshaped_As) * sqrt
    return spectra + noise

def normalize(spectra):
    '''Normalize so spectra is zero mean unit variance.'''
    return (spectra - 448561800000000.0) / 370311580000000.0 # 0 mean, 1 std

def input_transform(tbl_batch):
    '''Define a transformation from table to batched data tensor.'''
    data = tbl_batch["spectrum"]
    mh_ratio, alpham_ratio = tbl_batch["MH_ratio"], tbl_batch["alphaM_ratio"]
    t_eff, log_g = tbl_batch["T_eff"], tbl_batch["log_g"]
    data_float = tf.cast(data, dtype=tf.float32)
    mh_ratio_float = tf.cast(mh_ratio, dtype=tf.float32)
    alpham_ratio_float = tf.cast(alpham_ratio, dtype=tf.float32)
    t_eff_float = tf.cast(t_eff, dtype=tf.float32)
    log_g_float = tf.cast(log_g, dtype=tf.float32)

    noisy_data = add_noise(data_float)
    normalized_data = normalize(noisy_data)
    reshaped_data = tf.expand_dims(normalized_data, 2) # add a channel dimension

    labels = tf.stack([t_eff_float, log_g_float, mh_ratio_float, alpham_ratio_float], axis=1)
    return reshaped_data, labels

def create_loader_from_hdf5(sess, batch_size, filename):
    loader = tftables.load_dataset(filename=filename,
                                   dataset_path="/spectra",
                                   input_transform=input_transform,
                                   batch_size=batch_size,
                                   cyclic=True,
                                   ordered=True)

    data_stream, metals_stream = loader.dequeue()

    def initialize_stream():
        '''tftables has no initialization, so this is an empty function'''
        pass

    loader.start(sess)
    return data_stream, metals_stream, initialize_stream

def create_loader_from_array(sess, batch_size, spectra, labels):
    '''Create an iterator and initializer from a data array with shape [n_data, n-bins].'''
    n_data, n_bins = spectra.shape
    spectra_placeholder = tf.compat.v1.placeholder(tf.float32, [n_data, n_bins, 1])
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices(spectra_placeholder).batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    input_stream = iterator.get_next()

    n_data, n_bins = labels.shape
    label_placeholder = tf.compat.v1.placeholder(tf.float32, [n_data, n_bins])
    label_dataset = tf.compat.v1.data.Dataset.from_tensor_slices(label_placeholder).batch(batch_size)
    label_iterator = label_dataset.make_initializable_iterator()
    label_stream = label_iterator.get_next()

    def initialize_stream():
        sess.run([iterator.initializer, label_iterator.initializer], 
                 feed_dict={spectra_placeholder: spectra[:, :, np.newaxis],
                            label_placeholder: labels})

    return input_stream, label_stream, initialize_stream
