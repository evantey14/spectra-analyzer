import numpy as np
import tensorflow as tf
import tftables

def input_transform(tbl_batch):
    '''Define a transformation from table to batched data tensor.'''
    data = tbl_batch["spectrum"]
    mh_ratio, alpham_ratio = tbl_batch["MH_ratio"], tbl_batch["alphaM_ratio"]
    data_float = tf.to_float(data)
    mh_ratio_float, alpham_ratio_float = tf.to_float(mh_ratio), tf.to_float(alpham_ratio)

    data_slice = data_float[:, 700000:740000] # section chosen by hand because it has features in it
    data_max = tf.reduce_max(data_slice, axis=1)
    normalized_data = tf.divide(data_slice, tf.expand_dims(data_max, axis=1))
    reshaped_normalized_data = tf.expand_dims(normalized_data, 2) # add a channel dimension

    metals = tf.stack([mh_ratio_float, alpham_ratio_float], axis=1)
    return reshaped_normalized_data, metals

def create_loader_from_hdf5(sess, batch_size, filename):
    loader = tftables.load_dataset(filename=filename,
                                   dataset_path="/spectra",
                                   input_transform=input_transform,
                                   batch_size=batch_size,
                                   cyclic=True,
                                   ordered=True)

    data_stream, metals_stream = loader.dequeue()

    def initialize_input_stream():
        '''tftables has no initialization, so this is an empty function'''
        pass

    loader.start(sess)
    return data_stream, initialize_input_stream

def create_loader_from_array(sess, batch_size, data):
    '''Create an iterator and initializer from a data array with shape [n_data, n-bins].'''
    n_data, n_bins = data.shape
    placeholder_data = tf.compat.v1.placeholder(tf.float32, [n_data, n_bins, 1])
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices(placeholder_data)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    input_stream = iterator.get_next()

    def initialize_input_stream():
        sess.run(iterator.initializer, feed_dict={placeholder_data: data[:, :, np.newaxis]})

    return iterator.get_next(), initialize_input_stream
