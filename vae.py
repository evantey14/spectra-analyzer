import numpy as np
import tensorflow as tf
import tftables


class VariationalAutoencoder:
    def __init__(self, sess, network_architecture, input_stream, label_stream, learning_rate, batch_size):
        """Set up the VAE.

        Args:
            network_architecture: dict, containing
                                  {
                                      "input_size": length of a single spectrum,
                                      "latent_representation_size": number of latent variables,
                                      "encoder_layer_sizes": [int], 
                                      "decoder_layer_sizes": [int],
                                      "label_predictor_layer_sizes": [int]
                                  }
            learning_rate: float, learning rate for the vae optimizer
            batch_size: int, number of spectra being processed at any given time.
        """
        # Store metadata
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Set up the network
        self.input = input_stream[:, :, 0] # Ignore channel dimension
        self.label = label_stream
        self._create_network()
        self._create_optimizers()

        # Create Saver
        self.saver = tf.compat.v1.train.Saver()
    
        # Initialize values
        self.sess = sess
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def _create_network(self):
        """Create the tensor network.
      
        VAE:
            1) Connect the encoder layers based on the network arhitecture
            2) Create the sampling step from the latent distribution
            3) Connect the encoder layers
            4) Normalize the resultant spectra with the max value

        Metal Predictor:
            1) Create a simple fully connected NN
        """
        with tf.compat.v1.variable_scope("vae", reuse=tf.compat.v1.AUTO_REUSE):
            self._create_encoder()

            latent_representation_size = self.network_architecture["latent_representation_size"]
            self.latent_mean = self.encoder["layers"][-1][:, :latent_representation_size]
            self.latent_sigma = tf.exp(self.encoder["layers"][-1][:, latent_representation_size:])
            eps = tf.compat.v1.random_normal((self.batch_size, latent_representation_size), 0, 1)
            self.latent_representation = self.latent_mean + eps * self.latent_sigma

            self._create_decoder()

            self.reconstruction = self.decoder["layers"][-1]

        with tf.compat.v1.variable_scope("label_predictor", reuse=tf.compat.v1.AUTO_REUSE):
            self._create_label_predictor()
            # paper says metallicity ranges are a subset of [-4, 4].
            self.label_prediction = 4 * self.label_predictor["layers"][-1]

    def _create_encoder(self):
        encoder = {}

        first_layer = self.input
        encoder_layer_sizes = self.network_architecture["encoder_layer_sizes"]
        activation_fns = [tf.nn.relu] * len(encoder_layer_sizes)
        activation_fns[-1] = lambda x: x # linear activation for final layer
        weights, biases, layers = self._generate_layers_from_architecture(
            "encoder", first_layer, encoder_layer_sizes, activation_fns
        )

        encoder["weights"], encoder["biases"], encoder["layers"] = weights, biases, layers
        self.encoder = encoder

    def _create_decoder(self):
        decoder = {}
    
        first_layer = self.latent_representation
        decoder_layer_sizes = self.network_architecture["decoder_layer_sizes"]
        activation_fns = [tf.nn.relu] * len(decoder_layer_sizes)
        activation_fns[-1] = tf.nn.sigmoid
        weights, biases, layers = self._generate_layers_from_architecture(
            "decoder", first_layer, decoder_layer_sizes, activation_fns
        )

        decoder["weights"], decoder["biases"], decoder["layers"] = weights, biases, layers
        self.decoder = decoder

    def _create_label_predictor(self):
        label_predictor = {}

        first_layer = self.latent_representation
        label_predictor_layer_sizes = self.network_architecture["label_predictor_layer_sizes"]
        activation_fns = [tf.nn.relu] * len(label_predictor_layer_sizes)
        activation_fns[-1] = tf.nn.tanh

        weights, biases, layers = self._generate_layers_from_architecture(
            "label_predictor", first_layer, label_predictor_layer_sizes, activation_fns
        )

        label_predictor["weights"] = weights
        label_predictor["biases"] = biases
        label_predictor["layers"] = layers

        self.label_predictor = label_predictor

    def _generate_layers_from_architecture(self, name, first_layer, layer_sizes, activation_fns):
        """Generate weight, bias, and layer tensors given a list of layer sizes.

        Args:
            name: str, what the variables should be named
            first_layer: Tensor, the first layer in the architecture 
            layer_sizes: [int], indicating the size of each fully connected layer
            activation_fns: [function], activation functions for each neuron

        Returns:
            weights, biases, layers. lists of Tensors.
        """
        weights = []
        biases = []
        layers = [first_layer]
        for i in range(1, len(layer_sizes)):
            n_in, n_out = layer_sizes[i-1], layer_sizes[i]
            previous_layer = layers[-1]

            weights.append(tf.compat.v1.get_variable("{}-weight-{}".format(name, i),
                                                     shape=(n_in, n_out),
                                                     dtype=tf.float32,
                                                     initializer=tf.compat.v1.orthogonal_initializer))
            biases.append(tf.compat.v1.get_variable("{}-bias-{}".format(name, i), shape=(n_out),
                                                    dtype=tf.float32,
                                                    initializer=tf.zeros_initializer))
            layers.append(activation_fns[i](tf.matmul(previous_layer, weights[-1]) + biases[-1]))
        return weights, biases, layers 

    def _create_optimizers(self):
        """Define the cost functions.

        VAE:
            For the reconstruction cost, assume Gaussian distribution hence an L2 loss.
            The latent cost comes from the KL divergence to a N(0, 1) distribution.

        Metal Predictor:
            Use a simple L2 loss.
        """
        self.reconstruction_cost = tf.reduce_sum(tf.square(self.input - self.reconstruction))
        self.latent_cost = -0.5 * tf.reduce_sum(1 + tf.math.log(1e-5 + tf.square(self.latent_sigma))
                                                - tf.square(self.latent_mean)
                                                - tf.square(self.latent_sigma))

        self.l1_loss = tf.reduce_mean(tf.abs(self.input - self.reconstruction)) # just for reporting
        self.cost = tf.reduce_mean(self.reconstruction_cost + self.latent_cost)
        
        self.vae_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.cost,
            var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="vae")
        )

        self.label_cost = tf.reduce_mean(tf.square(self.label - self.label_prediction))
        self.label_predictor_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(
                self.label_cost, var_list=tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="label_predictor"
            )
        )

    def optimize(self):
        """Run one optimization iteration."""
        return self.sess.run([self.vae_optimizer, self.cost,
                              self.l1_loss, self.reconstruction_cost, self.latent_cost,
                              self.label_predictor_optimizer, self.label_cost])

    def optimize_label_predictor(self):
        """Run one optimization iteration just for the label predictor."""
        return self.sess.run([self.label_predictor_optimizer, self.label_cost])

    def encode(self, spectrum):
        """Map spectrum to latent representation."""
        s = np.copy(spectrum)
        weights, biases = self.encoder["weights"], self.encoder["biases"]

        for i in range(len(weights)):
            if i == len(weights)-1:
                s = weights[i].eval().T @ s + biases[i].eval()
            else:
                s = np.clip(weights[i].eval().T @ s + biases[i].eval(), 0, None)

        latent_representation_size = self.network_architecture["latent_representation_size"]
        latent_representation = np.random.normal(s[:latent_representation_size],
                                                 np.exp(s[latent_representation_size:]),
                                                 (latent_representation_size,))
        return latent_representation

    def decode(self, latent_representation):
        """Generate spectrum from latent representation."""
        s = np.copy(latent_representation)
        weights, biases = self.decoder["weights"], self.decoder["biases"]

        for i in range(len(self.decoder["weights"])):
            if i == len(weights) - 1:
                s = 1 / (1 + np.exp(-(weights[i].eval().T @ s + biases[i].eval())))
            else:
                s = np.clip(weights[i].eval().T @ s + biases[i].eval(), 0, None)
        return s

    def reconstruct(self, spectrum):
        """Reconstruct spectrum."""
        return self.decode(self.encode(spectrum))
    
    def save(self, checkpoint_name):
        self.saver.save(self.sess, checkpoint_name)
        print("saved to", checkpoint_name)

    def restore(self, checkpoint_name):
        self.saver.restore(self.sess, checkpoint_name)
        print("loaded model weights from", checkpoint_name)

    def close(self):
        self.loader.stop(self.sess)
        self.sess.close() 
