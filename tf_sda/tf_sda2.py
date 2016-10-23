import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utilities
from util import *
from tensorflow.examples.tutorials.mnist import input_data

__version__ = '0.1'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', 'data', 'data dir')
flags.DEFINE_integer('drop_out_rate', 0.5, 'the drop out rate')
flags.DEFINE_string('activations', 'relu', 'Type of activations. ["sigmoid", "tanh", "relu"]')
flags.DEFINE_string('x_train_path', 'data/MNIST', 'Train file path')
flags.DEFINE_string('batch_method', 'random', 'How to generate data')
flags.DEFINE_integer('epochs', 100, 'Train file path')
flags.DEFINE_integer('log_step', 100, 'tensorflow writer log data')
flags.DEFINE_integer('batch_size', 100, 'training batch size')
flags.DEFINE_boolean('debug', True, 'debug the system')
flags.DEFINE_string('log_dir', 'model', 'the log dir')

"""
###################
### TENSORBOARD ###
###################
"""


def attach_variable_summaries(var, name, summ_list):
    """Attach statistical summaries to a tensor for tensorboard visualization."""
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        summ_mean = tf.scalar_summary("mean/" + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(var, mean))))
        summ_std = tf.scalar_summary('stddev/' + name, stddev)
        summ_max = tf.scalar_summary('max/' + name, tf.reduce_max(var))
        summ_min = tf.scalar_summary('min/' + name, tf.reduce_min(var))
        summ_hist = tf.histogram_summary(name, var)
    summ_list.extend([summ_mean, summ_std, summ_max, summ_min, summ_hist])


def attach_scalar_summary(var, name, summ_list):
    """Attach scalar summaries to a scalar."""
    summ = tf.scalar_summary(tags=name, values=var)
    summ_list.append(summ)


"""
############################
### TENSORFLOW UTILITIES ###
############################
"""


def layer_parameter(shape, scope):
    with tf.name_scope(scope):
        weight = tf.Variable(tf.random_uniform(shape=shape,  minval=-4*np.sqrt(6./(shape[0]+shape[1])),
                            maxval=4*np.sqrt(6./(shape[0]+shape[1]))),  dtype=tf.float32, name='weights')
        hbias = tf.Variable(tf.zeros(shape=[shape[1]]), dtype=tf.float32, name='h_bias')
        vbias = tf.Variable(tf.zeros(shape=[shape[0]]), dtype=tf.float32, name='v_bias')
    return weight, hbias, vbias


def weight_variable(input_dim, output_dim, name=None, stretch_factor=1, dtype=tf.float32):
    """Creates a weight variable with initial weights as recommended by Bengio.
    Reference: http://arxiv.org/pdf/1206.5533v2.pdf. If sigmoid is used as the activation
    function, then a stretch_factor of 4 is recommended."""
    limit = np.sqrt(6 / (input_dim + output_dim))
    initial = tf.random_uniform(shape=[input_dim, output_dim],
                                minval=-(stretch_factor * limit),
                                maxval=stretch_factor * limit,
                                dtype=dtype)
    return tf.Variable(initial, name=name)


def bias_variable(dim, initial_value=0.0, name=None, dtype=tf.float32):
    """Creates a bias variable with an initial constant value."""
    return tf.Variable(tf.constant(value=initial_value, dtype=dtype, shape=[dim]), name=name)


def corrupt(tensor, corruption_level=0.05):
    """Uses the masking noise algorithm to mask corruption_level proportion
    of the input.
    :param tensor: A tensor whose values are to be corrupted.
    :param corruption_level: An int [0, 1] specifying the probability to corrupt each value.
    :return: The corrupted tensor.
    """
    total_samples = tf.reduce_prod(tf.shape(tensor))
    corruption_matrix = tf.multinomial(tf.log([[corruption_level, 1 - corruption_level]]), total_samples)
    corruption_matrix = tf.cast(tf.reshape(corruption_matrix, shape=tf.shape(tensor)), dtype=tf.float32)
    return tf.mul(tensor, corruption_matrix)

"""
############################
### NEURAL NETWORK LAYER ###
############################
"""


class NNLayer:
    """A container class to represent a hidden layer in the autoencoder network."""

    def __init__(self, input_dim, output_dim, name="hidden_layer", activation=None, weights=None, biases=None):
        """Initializes an NNLayer with empty weights/biases (default). Weights/biases
        are meant to be updated during pre-training with set_wb. Also has methods to
        transform an input_tensor to an encoded representation via the weights/biases
        of the layer.
        :param input_dim: An int representing the dimension of input to this layer.
        :param output_dim: An int representing the dimension of the encoded output.
        :param activation: A function to transform the inputs to this layer (sigmoid, etc.).
        :param weights: A tensor with shape [input_dim, output_dim]
        :param biases: A tensor with shape [output_dim]
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.activation = activation
        self.weights = weights      # Evaluated numpy array, static
        self.biases = biases        # Evaluated numpy array, static
        self._weights = None        # Weights Variable, dynamic
        self._biases = None         # Biases Variable, dynamic

    @property
    def is_pretrained(self):
        return self.weights is not None and self.biases is not None

    def set_wb(self, weights, biases):
        """Used during pre-training for convenience."""
        self.weights = weights      # Evaluated numpy array
        self.biases = biases        # Evaluated numpy array

        print("Set weights of layer with shape", weights.shape)
        print("Set biases of layer with shape", biases.shape)

    def set_wb_variables(self, summ_list):
        """This function is called at the beginning of supervised fine tuning to create new
        variables with initial values based on their static parameter counterparts. These
        variables can then all be adjusted simultaneously during the fine tune optimization."""
        assert self.is_pretrained, "Cannot set Variables when not pretrained."
        with tf.name_scope(self.name):
            self._weights = tf.Variable(self.weights, dtype=tf.float32, name="weights")
            self._biases = tf.Variable(self.biases, dtype=tf.float32, name="biases")
            attach_variable_summaries(self._weights, name=self._weights.name, summ_list=summ_list)
            attach_variable_summaries(self._biases, name=self._biases.name, summ_list=summ_list)
        print("Created new weights and bias variables from current values.")

    def update_wb(self, sess):
        """This function is called at the end of supervised fine tuning to update the static
        weight and bias values to the newest snapshot of their dynamic variable counterparts."""
        assert self._weights is not None and self._biases is not None, "Weights and biases Variables not set."
        self.weights = sess.run(self._weights)
        self.biases = sess.run(self._biases)
        print("Updated weights and biases with corresponding evaluated variable values.")

    def get_weight_variable(self):
        return self._weights

    def get_bias_variable(self):
        return self._biases

    def encode(self, input_tensor, use_variables=False):
        """Performs this layer's encoding on the input_tensor. use_variables is set to true
        during the fine-tuning stage, when all parameters of each layer need to be adjusted."""
        assert self.is_pretrained, "Cannot encode when not pre-trained."
        if use_variables:
            return self.activate(tf.matmul(input_tensor, self._weights) + self._biases)
        else:
            return self.activate(tf.matmul(input_tensor, self.weights) + self.biases)

    def activate(self, input_tensor, name=None):
        """Applies the activation function for this layer based on self.activation."""
        if self.activation == "sigmoid":
            return tf.nn.sigmoid(input_tensor, name=name)
        if self.activation == "tanh":
            return tf.nn.tanh(input_tensor, name=name)
        if self.activation == "relu":
            return tf.nn.relu(input_tensor, name=name)
        else:
            print("Activation function not valid. Using the identity.")
            return input_tensor

"""
#####################################
### STACKED DENOISING AUTOENCODER ###
#####################################
"""


class SDAutoencoder(object):
    def check_assertions(self):
        assert 0 <= self.noise <= 1, "Invalid noise value given: %s" % self.noise

    def __init__(self, dims, activations, sess, noise=0.0, pretrain_lr=0.001, finetune_lr=0.001,
                 batch_size=100, print_step=100, loss='cross-entropy'):
        self.dims = dims
        self.input_dim = dims[0]  # The dimension of the raw input
        self.output_dim = dims[-1]  # The output dimension of the last layer: fully encoded input
        self.noise = noise
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.batch_size = batch_size
        self.print_step = print_step
        self.activations = activations
        self.loss = loss

        self.hidden_layers = self.create_new_layers()
        self.sess = sess

        self.check_assertions()
        print("Initialized SDA network with dims %s, activations %s, noise %s, "
              "pretraining learning rate %s, finetuning learning rate %s, and batch size %s."
              % (dims, activations, self.noise, self.pretrain_lr, self.finetune_lr, self.batch_size))

    @property
    def is_pretrained(self):
        """Returns whether the whole autoencoder network (all layers) is pre-trained."""
        return all([layer.is_pretrained for layer in self.hidden_layers])

    ##########################
    # VARIABLE CONFIGURATION #
    ##########################

    def get_all_variables(self, additional_layer=None):
        all_variables = []
        for layers in self.hidden_layers:
            all_variables.extend([layers.get_weight_variable(), layers.get_bias_variable()])
        if additional_layer:
            all_variables.extend(additional_layer)
        return all_variables

    def setup_all_variables(self, summ_list):
        """See NNLayer.set_wb_variables. Performs layer method on all hidden layers."""
        for layer in self.hidden_layers:
            layer.set_wb_variables(summ_list)

    def finalize_all_variables(self):
        """See NNLayer.finalize_all_variables. Performs layer method on all hidden layers."""
        for layer in self.hidden_layers:
            layer.update_wb(self.sess)

    def save_variables(self, filepath):
        """Saves all Tensorflow variables in the desired filepath."""
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, filepath)
        print("Model saved in file: %s" % save_path)

    ################
    # WRITING DATA #
    ################
    def write_encoded_input(self, filepath, x_test_path):
        """Reads from x_test_path and encodes the input through the entire model. Then
        writes the encoded result to filepath. Call this function after pretraining and
        fine-tuning to get the newly learned features.
        """
        x_test = get_batch_generator(x_test_path, self.batch_size)
        self.write_encoded_input_gen(filepath, x_test_gen=x_test)

    def write_encoded_input_gen(self, filepath, x_test_gen):
        """Get encoded feature representation and writes to filepath.
        :param filepath: A string specifying the file path/name to write the encoded input to.
        :param x_test_gen: A generator that iterates through the x-test values.
        :return: None
        """
        sess = self.sess
        x_input = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        x_encoded = self.get_encoded_input(x_input, depth=-1, use_variables=False)

        print("Beginning to write to file.")
        for x_batch in x_test_gen:
            self.write_data(sess.run(x_encoded, feed_dict={x_input: x_batch}), filepath)
        print("Written encoded input to file %s" % filepath)

    def write_encoded_input_with_ys(self, filepath_x, filepath_y, xy_test_gen):
        """For use in testing MNIST. Writes the encoded x values along with their corresponding
        y values to file.
        :param filepath_x: A string, the filepath to store the encoded x values.
        :param filepath_y: A string, the filepath to store the y values.
        :param xy_test_gen: A generator that yields tuples of x and y test values.
        :return: None
        """
        sess = self.sess
        x_input = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        x_encoded = self.get_encoded_input(x_input, depth=-1, use_variables=False)

        print("Beginning to write to file encoded x with ys.")
        for x_batch, y_batch in xy_test_gen:
            self.write_data(sess.run(x_encoded, feed_dict={x_input: x_batch}), filepath_x)
            self.write_data(y_batch, filepath_y)
        print("Written encoded input to file %s and test ys to %s" % (filepath_x, filepath_y))

    ###################
    # GENERAL UTILITY #
    ###################

    def get_encoded_input(self, input_tensor, depth, use_variables=False):
        depth = len(self.hidden_layers) if depth == -1 else depth
        for i in range(depth):
            input_tensor = self.hidden_layers[i].encode(input_tensor, use_variables=use_variables)
        return input_tensor

    def get_loss(self, labels, values, epsilon=1e-10):
        return tf.reduce_mean(-tf.reduce_sum(labels * tf.log(values + epsilon) + (1 - labels) *
                                             tf.log(1 - values + epsilon), reduction_indices=[1]))

    def create_new_layers(self):
        assert len(self.dims) >= 2 and len(self.activations) > 1, "Invalid number of layers given by `dims` and `activations`."
        assert len(self.dims) == len(self.activations)+1, 'Incorrect number of layers/activations.'
        return [NNLayer(self.dims[i], self.dims[i+1], 'hidden_layer'+str(i), self.activations[i]) for i in range(len(self.activations))]

    ###############
    # PRETRAINING #
    ###############
    def pre_train_network(self):
        print 'Starting to pretrain autoencoder network.'
        data = input_data.read_data_sets("data/MNIST", one_hot=True)
        for i in range(len(self.hidden_layers)):
            if FLAGS.batch_method == "random":
                # x_train = get_random_batch_generator(self.batch_size, FLAGS.x_train_path, repeat=FLAGS.epochs - 1)
                # data = input_data.read_data_sets("data/MNIST", one_hot=True)
                x_train = data.train.images
                np.random.shuffle(x_train)
                x_train = [_ for _ in utilities.gen_batches(x_train, FLAGS.batch_size)]
            else:
                # x_train = get_batch_generator(FLAGS.x_train_path, self.batch_size, repeat=FLAGS.epochs-1)
                # data = input_data.read_data_sets("data/MNIST", one_hot=True)
                x_train = data.train.images
                x_train = [_ for _ in utilities.gen_batches(x_train, FLAGS.batch_size)]
            self.pre_train_layer(i, x_train)
        print 'Finished pretraining of autoencoder network.'

    def pre_train_layer(self, depth, data):
        sess = self.sess
        print 'Starting to pretrain layer %d.' % depth

        hidden_layer = self.hidden_layers[depth]
        summary_list = []

        with tf.name_scope(hidden_layer.name):
            input_dim, output_dim = hidden_layer.input_dim, hidden_layer.output_dim

            with tf.name_scope("x_values"):
                x_original = tf.placeholder(tf.float32, shape=[None, self.input_dim])
                x_latent = self.get_encoded_input(x_original, depth, use_variables=False)
                x_corrupt = corrupt(x_latent, corruption_level=self.noise)

            with tf.name_scope('encoding_vars'):
                stretch_factor = 4 if self.activations[depth] == "sigmoid" else 1
                encode = {
                    "weights": weight_variable(input_dim, output_dim, name="weights", stretch_factor=stretch_factor),
                    "biases": bias_variable(output_dim, initial_value=0, name="biases")
                }
                attach_variable_summaries(encode["weights"], encode["weights"].name, summ_list=summary_list)
                attach_variable_summaries(encode["biases"], encode["biases"].name, summ_list=summary_list)

            with tf.name_scope("decoding_vars"):
                decode = {
                    "weights": tf.transpose(encode["weights"], name="transposed_weights"),  # Tied weights
                    "biases": bias_variable(input_dim, initial_value=0, name="decode_biases")
                }
                attach_variable_summaries(decode["weights"], decode["weights"].name, summ_list=summary_list)
                attach_variable_summaries(decode["biases"], decode["biases"].name, summ_list=summary_list)

            with tf.name_scope("encoded_and_decoded"):
                encoded = hidden_layer.activate(tf.matmul(x_corrupt, encode["weights"]) + encode["biases"])
                decoded = hidden_layer.activate(tf.matmul(encoded, decode["weights"]) + decode["biases"])
                attach_variable_summaries(encoded, "encoded", summ_list=summary_list)
                attach_variable_summaries(decoded, "decoded", summ_list=summary_list)

            # Reconstruction loss
            with tf.name_scope("reconstruction_loss"):
                loss = self.get_loss(x_latent, decoded)
                attach_scalar_summary(loss, "%s_loss" % self.loss, summ_list=summary_list)

            trainable_vars = [encode["weights"], encode["biases"], decode["biases"]]
            # Only optimize variables for this layer ("greedy")
            with tf.name_scope("train_step"):
                train_op = tf.train.AdamOptimizer(learning_rate=self.pretrain_lr).minimize(
                    loss, var_list=trainable_vars)
            sess.run(tf.initialize_all_variables())

            # Merge summaries and get a summary writer
            merged = tf.merge_summary(summary_list)
            pretrain_writer = tf.train.SummaryWriter("model/" + hidden_layer.name, sess.graph)

            step = 0
            for batch_x_original in data:
                print batch_x_original.shape
                sess.run(train_op, feed_dict={x_original: batch_x_original})

                if step % self.print_step == 0:
                    loss_value = sess.run(loss, feed_dict={x_original: batch_x_original})
                    print("Step %s, batch %s loss = %s" % (step, self.loss, loss_value))

                if step % FLAGS.log_step == 0:
                    summary = sess.run(merged, feed_dict={x_original: batch_x_original})
                    pretrain_writer.add_summary(summary, global_step=step)

                # Break for debugging purposes
                if FLAGS.debug and step > 5:
                    break
                step += 1
            # Set the weights and biases of pretrained hidden layer
            hidden_layer.set_wb(weights=sess.run(encode["weights"]), biases=sess.run(encode["biases"]))
            print("Finished pretraining of layer %d. Updated layer weights and biases." % depth)

    ##############
    # FINETUNING #
    ##############
    def finetune_parameters(self, output_dim, epochs=1, batch_method="random"):
        """Performs fine tuning on all parameters of the neural network plus two additional softmax
        variables. Call this method after `pretrain_network` is complete. Y values should be represented
        in one-hot format.
        :param x_train_path: A string, the path to the x train values.
        :param y_train_path: A string, the path to the y train values.
        :param output_dim: An int, the number of classes in the target classification problem. Ex: 10 for MNIST.
        :param epochs: An int, the number of iterations to tune through the entire dataset.
        :param batch_method: A string, either 'random' or 'sequential', to indicate how batches are retrieved.
        :return: The tuned softmax parameters (weights and biases) of the classification layer.
        """
        data = input_data.read_data_sets("data/MNIST", one_hot=True)
        if batch_method == "random":
            x_train = data.train.images
            y_label = data.train.labels
            shuff = zip(x_train, y_label)
            np.random.shuffle(shuff)
            xy_train = [_ for _ in utilities.gen_batches(shuff, FLAGS.batch_size)]
        else:
            x_train = data.train.images
            y_label = data.train.labels
            shuff = zip(x_train, y_label)
            xy_train = [_ for _ in utilities.gen_batches(shuff, FLAGS.batch_size)]

        return self.finetune_parameters_gen(xy_train_gen=xy_train, output_dim=output_dim)

    def finetune_parameters_gen(self, xy_train_gen, output_dim):
        """An implementation of finetuning to support data feeding from generators."""
        sess = self.sess
        summary_list = []

        print("Starting to fine tune parameters of network.")
        with tf.name_scope("finetuning"):
            self.setup_all_variables(summary_list)

            with tf.name_scope("inputs"):
                x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="raw_input")
                with tf.name_scope("fully_encoded"):
                    x_encoded = self.get_encoded_input(x, depth=-1, use_variables=True)  # Full depth encoding

            """Note on W below: The difference between self.output_dim and output_dim is that the former
            is the output dimension of the autoencoder stack, which is the dimension of the new feature
            space. The latter is the dimension of the y value space for classification. Ex: If the output
            should be binary, then the output_dim = 2."""
            with tf.name_scope("softmax_variables"):
                W = weight_variable(self.output_dim, output_dim, name="weights")
                b = bias_variable(output_dim, initial_value=0, name="biases")
                attach_variable_summaries(W, W.name, summ_list=summary_list)
                attach_variable_summaries(b, b.name, summ_list=summary_list)

            with tf.name_scope("outputs"):
                y_logits = tf.matmul(x_encoded, W) + b
                with tf.name_scope("predicted"):
                    y_pred = tf.nn.softmax(y_logits, name="y_pred")
                    attach_variable_summaries(y_pred, y_pred.name, summ_list=summary_list)
                with tf.name_scope("actual"):
                    y_actual = tf.placeholder(tf.float32, shape=[None, output_dim], name="y_actual")
                    attach_variable_summaries(y_actual, y_actual.name, summ_list=summary_list)

            with tf.name_scope("cross_entropy"):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_logits, y_actual))
                attach_scalar_summary(cross_entropy, "cross_entropy", summ_list=summary_list)

            trainable_vars = self.get_all_variables(additional_layer=[W, b])
            # trainable_vars = self.get_all_variables()
            # print trainable_vars
            for var in trainable_vars:
                print var.name
            with tf.name_scope("train_step"):
                train_step = tf.train.AdamOptimizer(learning_rate=self.finetune_lr).minimize(
                    cross_entropy, var_list=trainable_vars)
                print trainable_vars

            with tf.name_scope("evaluation"):
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                attach_scalar_summary(accuracy, "finetune_accuracy", summ_list=summary_list)

            sess.run(tf.initialize_all_variables())

            # Merge summaries and get a summary writer
            merged = tf.merge_summary(summary_list)
            train_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/train/finetune", sess.graph)

            step = 0
            for batch in xy_train_gen:
                batch_xs, batch_ys = zip(*batch)
                if step % self.print_step == 0:
                    print("Step %s, batch accuracy: " % step,
                          sess.run(accuracy, feed_dict={x: batch_xs, y_actual: batch_ys}))

                # For debugging predicted y values
                if step % (self.print_step * 10) == 0:
                    print("Predicted y-value:", sess.run(y_pred, feed_dict={x: batch_xs})[0])
                    print("Actual y-value:", batch_ys[0])

                if step % FLAGS.log_step == 0:
                    summary = sess.run(merged, feed_dict={x: batch_xs, y_actual: batch_ys})
                    train_writer.add_summary(summary, global_step=step)

                # For debugging, break early.
                if FLAGS.debug and step > 5:
                    break

                sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})
                step += 1

            self.finalize_all_variables()
            print("Completed fine-tuning of parameters.")
            tuned_params = {"weights": sess.run(W), "biases": sess.run(b)}

            return tuned_params


if __name__ == '__main__':
    # Start a TensorFlow session
    sess = tf.Session()

    # Initialize an unconfigured autoencoder with specified dimensions, etc.
    sda = SDAutoencoder(dims=[784, 256, 64, 32],
                        activations=["relu", "relu", "relu"],
                        sess=sess,
                        noise=0.1)

    # Pretrain weights and biases of each layer in the network.
    sda.pre_train_network()

    # Read in test y-values to softmax classifier.
    sda.finetune_parameters(epochs=10, output_dim=10)

    # Write to file the newly represented features.
    # sda.write_encoded_input(filepath="data/transformed.csv", x_test_path=FLAGS.x_train_path)





