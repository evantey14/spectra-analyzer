import numpy as np
import tensorflow as tf
import tftables

def _input_transform(tbl_batch):
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

def create_data_loader(sess, batch_size, n_data, n_bins, filename="/mnt/raid0/gabrielc/sample_8k.h5"):
    assert n_bins == 40000
    assert n_data == 8000
    loader = tftables.load_dataset(filename=filename,
                                   dataset_path="/spectra",
                                   input_transform=_input_transform,
                                   batch_size=batch_size,
                                   cyclic=True,
                                   ordered=True)

    data_stream, metals_stream = loader.dequeue()

    def initialize_input_stream():
        '''tftables has no initialization, so this is an empty function'''
        pass

    loader.start(sess)
    data_init = sess.run(data_stream)
    return data_stream, initialize_input_stream, data_init 
