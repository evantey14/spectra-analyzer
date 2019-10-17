import numpy as np
import tensorflow as tf
import tftables


class VariationalAutoencoder:
    def __init__(self, network_architecture, data_file, learning_rate=0.001, batch_size=50):
        """Set up the VAE.

        Args:
            network_architecture: dict, containing
                                  {
                                      "input_size": length of a single spectrum,
                                      "latent_representation_size": number of latent variables,
                                      "encoder_layer_sizes": [int], 
                                      "decoder_layer_sizes": [int],
                                      "metals_predictor_layer_sizes": [int]
                                  }
            learning_rate: float, learning rate for the vae optimizer
            batch_size: int, number of spectra being processed at any given time.
        """
        # Store metadata
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Set up the network
        tf.reset_default_graph()
        self._create_data_pipe(data_file)
        self._create_network()
        self._create_optimizers()

        # Create Saver
        self.saver = tf.train.Saver()
    
        # Initialize values
        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.loader.start(self.sess)
        self.sess.run(init)

    def _create_data_pipe(self, data_file):
        """Create a loader via tftables that will load data."""
        self.loader = tftables.load_dataset(filename=data_file,
                                       dataset_path="/spectra",
                                       input_transform=self._input_transform,
                                       batch_size=self.batch_size,
                                       cyclic=True,
                                       ordered=True)

        data_batch, metals_batch = self.loader.dequeue()
        self.input, self.metals = data_batch, metals_batch

    def _input_transform(self, tbl_batch):
        """Function that transforms hdf5 files to usable tensors via tftables."""
        data = tbl_batch["spectrum"]
        mh_ratio, alpham_ratio = tbl_batch["MH_ratio"], tbl_batch["alphaM_ratio"]

        data_float = tf.to_float(data)
        mh_ratio_float, alpham_ratio_float = tf.to_float(mh_ratio), tf.to_float(alpham_ratio)

        data_slice = data_float[:, 700000:750000] # let's try just training on a portion of the spectrum
        data_max = tf.reduce_max(data_slice, axis=1)
        normalized_data = tf.divide(data_slice, tf.expand_dims(data_max, axis=1))

        metals = tf.stack([mh_ratio_float, alpham_ratio_float], axis=1)
        return normalized_data, metals

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
        with tf.variable_scope("vae"):
            self._create_encoder()

            latent_representation_size = self.network_architecture["latent_representation_size"]
            self.latent_mean = self.encoder["layers"][-1][:, :latent_representation_size] 
            self.latent_sigma = self.encoder["layers"][-1][:, latent_representation_size:]
            eps = tf.random_normal((self.batch_size, latent_representation_size), 0, 1)
            self.latent_representation = self.latent_mean + eps * self.latent_sigma

            self._create_decoder()

            maxes = tf.reduce_max(self.decoder["layers"][-1], axis=1)
            self.reconstruction = tf.divide(self.decoder["layers"][-1],
                                            tf.expand_dims(maxes, axis=1))

        with tf.variable_scope("metals_predictor"):
            self._create_metals_predictor()
            # paper says metallicity ranges are a subset of [-4, 4].
            self.metals_prediction = 4 * self.metals_predictor["layers"][-1]

    def _create_encoder(self):
        encoder = {}

        first_layer = self.input
        encoder_layer_sizes = self.network_architecture["encoder_layer_sizes"]
        weights, biases, layers = self._generate_layers_from_architecture(
            "encoder", first_layer, encoder_layer_sizes, tf.nn.relu
        )

        encoder["weights"], encoder["biases"], encoder["layers"] = weights, biases, layers
        self.encoder = encoder

    def _create_decoder(self):
        decoder = {}
    
        first_layer = self.latent_representation
        decoder_layer_sizes = self.network_architecture["decoder_layer_sizes"]
        weights, biases, layers = self._generate_layers_from_architecture(
            "decoder", first_layer, decoder_layer_sizes, tf.nn.relu
        )

        decoder["weights"], decoder["biases"], decoder["layers"] = weights, biases, layers
        self.decoder = decoder

    def _create_metals_predictor(self):
        metals_predictor = {}

        first_layer = self.latent_representation
        metals_predictor_layer_sizes = self.network_architecture["metals_predictor_layer_sizes"]
        weights, biases, layers = self._generate_layers_from_architecture(
            "metals_predictor", first_layer, metals_predictor_layer_sizes, tf.nn.tanh
        )

        metals_predictor["weights"] = weights
        metals_predictor["biases"] = biases
        metals_predictor["layers"] = layers

        self.metals_predictor = metals_predictor

    def _generate_layers_from_architecture(self, name, first_layer, layer_sizes, activation_fn):
        """Generate weight, bias, and layer tensors given a list of layer sizes.

        Args:
            name: str, what the variables should be named
            first_layer: Tensor, the first layer in the architecture 
            layer_sizes: [int], indicating the size of each fully connected layer
            activation_fn: function, activation function for each neuron

        Returns:
            weights, biases, layers. lists of Tensors.
        """
        weights = []
        biases = []
        layers = [first_layer]
        for i in range(1, len(layer_sizes)):
            n_in, n_out = layer_sizes[i-1], layer_sizes[i]
            previous_layer = layers[-1]

            weights.append(tf.get_variable("{}-weight-{}".format(name, i), shape=(n_in, n_out),
                                           dtype=tf.float32, initializer=tf.orthogonal_initializer))
            biases.append(tf.get_variable("{}-bias-{}".format(name, i), shape=(n_out),
                                          dtype=tf.float32, initializer=tf.zeros_initializer))
            layers.append(activation_fn(tf.matmul(previous_layer, weights[-1]) + biases[-1]))
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
        self.latent_cost = -0.5 * tf.reduce_sum(1 + tf.log(1e-5 + tf.square(self.latent_sigma))
                                                - tf.square(self.latent_mean)
                                                - tf.square(self.latent_sigma))

        self.l1_loss = tf.reduce_mean(tf.abs(self.input - self.reconstruction)) # just for reporting
        self.cost = tf.reduce_mean(self.reconstruction_cost + self.latent_cost)
        
        self.vae_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.cost, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="vae")
        )

        self.metals_cost = tf.reduce_mean(tf.square(self.metals - self.metals_prediction))
        self.metals_predictor_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.metals_cost, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="metals_predictor"
            )
        )

    def optimize(self):
        """Run one optimization iteration."""
        return self.sess.run([self.vae_optimizer, self.cost,
                              self.l1_loss, self.reconstruction_cost, self.latent_cost,
                              self.metals_predictor_optimizer, self.metals_cost])

    def optimize_metals_predictor(self):
        """Run one optimization iteration just for the metals predictor."""
        return self.sess.run([self.metals_predictor_optimizer, self.metals_cost])

    def encode(self, spectrum):
        """Map spectrum to latent representation."""
        s = np.copy(spectrum)
        weights, biases = self.encoder["weights"], self.encoder["biases"]

        for i in range(len(weights)):
            s = np.clip(weights[i].eval().T @ s + biases[i].eval(), 0, None)

        latent_representation_size = self.network_architecture["latent_representation_size"]
        latent_representation = np.random.normal(s[:latent_representation_size],
                                                 s[latent_representation_size:],
                                                 (latent_representation_size,))
        return latent_representation

    def decode(self, latent_representation):
        """Generate spectrum from latent representation."""
        s = np.copy(latent_representation)
        weights, biases = self.decoder["weights"], self.decoder["biases"]

        for i in range(len(self.decoder["weights"])):
            s = np.clip(weights[i].eval().T @ s + biases[i].eval(), 0, None)

        return s / np.max(s)

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
