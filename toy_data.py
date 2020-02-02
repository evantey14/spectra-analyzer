import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def planck(wavelength, T):
    '''Return an output intensity for a star with temperature T at some wavelength in nm.'''
    h = 6.626e-34
    c = 3e8
    k = 1.38e-23
    return 2 * h * c**2 / wavelength**5 * 1 / (np.exp(h * c / (wavelength * k * T)) - 1)


def generate_spectrum(bins, T, A, mu, sigma):
    '''Generate a normalized spectrum for star with temperature T and gaussian feature A*N(mu, sigma).'''
    wavelengths = np.linspace(0, 3e-6, num=bins+1)[1:] # don't use 0 wavelength
    spectrum = planck(wavelengths, T)
    norm_spectrum = spectrum / np.max(spectrum)
    
    noise = np.random.normal(0, .01, bins) 
    feature = 1 - A * np.exp(-(wavelengths - mu)**2 / 2 / sigma**2)
    return feature * norm_spectrum + noise

def sample_default_prior():
    '''Sample a point from test prior.'''
    T = np.random.uniform(2000, 7000)
    A = np.random.normal(.8, .1) # size of the dip should be fairly large
    mu = np.random.normal(1.5e-6, .01e-6) # dip should vaguely appear in the same place
    sigma = np.random.normal(.01e-6, .01e-6) # dip width should be narrow
    return T, A, mu, sigma

def create_data_stream(sess, batch_size, n_data, bins, sample_prior=sample_default_prior):
    '''Create an iterator, initializer, and data_init for a toy dataset. Must be called in a valid tf session.'''
    data = np.array([generate_spectrum(bins, *sample_prior()) for _ in range(n_data)])[:, :, np.newaxis]
    placeholder_data = tf.compat.v1.placeholder(tf.float32, [n_data, bins, 1])
    dataset = tf.data.Dataset.from_tensor_slices(placeholder_data)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()

    def initialize_iterator():
        sess.run(iterator.initializer, feed_dict={placeholder_data: data})

    initialize_iterator()
    data_init = sess.run(iterator.get_next())
    return iterator, initialize_iterator, data_init
