import numpy as np
import tensorflow as tf

from vae import VariationalAutoencoder


class ConvolutionalVAE(VariationalAutoencoder):
    def _create_encoder(self):
        encoder = {}
        
        batch_size = self.batch_size
        input_size = self.network_architecture["input_size"]
        self.conv_filter = tf.get_variable("encoder-filter", shape=(200, 1, 1), dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer)
        first_layer = tf.squeeze(tf.nn.conv1d(tf.expand_dims(self.input, -1), self.conv_filter, 1,
                                              "SAME"))

        encoder_layer_sizes = self.network_architecture["encoder_layer_sizes"]
        weights, biases, layers = self._generate_layers_from_architecture(
            "encoder", first_layer, encoder_layer_sizes, tf.nn.relu
        )

        encoder["weights"], encoder["biases"], encoder["layers"] = weights, biases, layers
        self.encoder = encoder

    def encode(self, spectrum):
        s = np.copy(spectrum)
        conv_filter = self.conv_filter.eval()
        
        # numpy's convolution is a signals convolution (not neural network)
        s = np.convolve(s, np.flip(conv_filter.flatten()), mode="same")

        return super().encode(s)
