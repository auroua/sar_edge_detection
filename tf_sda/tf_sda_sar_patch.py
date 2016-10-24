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
flags.DEFINE_boolean('debug', False, 'debug the system')
flags.DEFINE_string('log_dir', 'model_sar_patch', 'the log dir')

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
    limit = np.sqrt(6.0 / (input_dim + output_dim))
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

    def __init__(self, input_dim, output_dim, name="hidden_layer", activation=None):
        """Initializes an NNLayer with empty weights/biases (default). Weights/biases
        are meant to be updated during pre-training with set_wb. Also has methods to
        transform an input_tensor to an encoded representation via the weights/biases
        of the layer.
        :param input_dim: An int representing the dimension of input to this layer.
        :param output_dim: An int representing the dimension of the encoded output.
        :param activation: A function to transform the inputs to this layer (sigmoid, etc.).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.activation = activation

        stretch_factor = 4.0
        self.weights = weight_variable(input_dim, output_dim, name="weights", stretch_factor=stretch_factor)
        self.biases = bias_variable(output_dim, initial_value=0, name="encode_biases")
        self.decode_biases = bias_variable(input_dim, initial_value=0, name="decode_biases")

    def visible_variables(self, summ_list):
        with tf.name_scope(self.name):
            attach_variable_summaries(self.weights, name=self.weights.name, summ_list=summ_list)
            attach_variable_summaries(self.biases, name=self.biases.name, summ_list=summ_list)
        print("Created new weights and bias variables from current values.")

    def get_weight_variable(self):
        return self.weights

    def get_bias_variable(self):
        return self.biases

    def encode(self, input_tensor):
        return self.activate(tf.matmul(input_tensor, self.weights) + self.biases)

    def decode(self, encode_tensor):
        return self.activate(tf.matmul(encode_tensor, tf.transpose(self.weights)) + self.decode_biases)

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
                 batch_size=100, print_step=100):
        self.dims = dims
        self.input_dim = dims[0]  # The dimension of the raw input
        self.output_dim = dims[-1]  # The output dimension of the last layer: fully encoded input
        self.noise = noise
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.batch_size = batch_size
        self.print_step = print_step
        self.activations = activations
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

    def save_variables(self, filepath):
        """Saves all Tensorflow variables in the desired filepath."""
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, filepath)
        print("Model saved in file: %s" % save_path)

    ###################
    # GENERAL UTILITY #
    ###################

    def get_encoded_input(self, input_tensor, depth):
        depth = len(self.hidden_layers) if depth == -1 else depth
        for i in range(depth):
            print 'get encode input hidden layers ', i
            input_tensor = self.hidden_layers[i].encode(input_tensor)
        return input_tensor

    def get_loss(self, labels, values, epsilon=1e-10):
        return tf.reduce_mean(-tf.reduce_sum(labels * tf.log(values + epsilon) + (1 - labels) *
                                             tf.log(1 - values + epsilon), reduction_indices=[1]))

    def get_l2_loss(self, labels, values):
        return tf.nn.l2_loss(labels - values)/tf.cast(tf.shape(values)[0], tf.float32)

    def create_new_layers(self):
        assert len(self.dims) >= 2 and len(self.activations) > 1, "Invalid number of layers given by `dims` and `activations`."
        assert len(self.dims) == len(self.activations)+1, 'Incorrect number of layers/activations.'
        layers = [NNLayer(self.dims[i], self.dims[i+1], 'hidden_layer'+str(i), self.activations[i]) for i in range(len(self.activations))]
        print 'total layer size ====', len(layers)
        return layers
    ###############
    # PRETRAINING #
    ###############
    def pre_train_network(self, epochs):
        print 'Starting to pretrain autoencoder network.'
        data = input_data.read_data_sets("data/MNIST", one_hot=True)
        x_train = data.train.images
        for i in range(len(self.hidden_layers)):
            # if FLAGS.batch_method == "random":
            #     x_train = data.train.images
            #     np.random.shuffle(x_train)
            #     x_train = [_ for _ in utilities.gen_batches(x_train, FLAGS.batch_size)]
            # else:
            #     x_train = data.train.images
            #     x_train = [_ for _ in utilities.gen_batches(x_train, FLAGS.batch_size)]
            self.pre_train_layer(i, x_train, epochs)
        print 'Finished pretraining of autoencoder network.'

    def pre_train_layer(self, depth, data, epoch):
        sess = self.sess
        print 'Starting to pretrain layer %d.' % depth
        hidden_layer = self.hidden_layers[depth]
        summary_list = []
        with tf.name_scope(hidden_layer.name):
            with tf.name_scope("x_values"):
                x_original = tf.placeholder(tf.float32, shape=[None, self.input_dim])
                x_latent = self.get_encoded_input(x_original, depth)
                x_corrupt = corrupt(x_latent, corruption_level=self.noise)

            with tf.name_scope("encoded_and_decoded"):
                encoded = hidden_layer.encode(x_corrupt)
                encoded = tf.nn.dropout(encoded, keep_prob=0.5)
                decoded = hidden_layer.decode(encoded)
                attach_variable_summaries(encoded, "encoded", summ_list=summary_list)
                attach_variable_summaries(decoded, "decoded", summ_list=summary_list)

            # Reconstruction loss
            with tf.name_scope("reconstruction_loss"):
                # loss = self.get_loss(x_latent, decoded)
                loss = self.get_l2_loss(x_latent, decoded)
                attach_scalar_summary(loss, "%s_loss" % 'l2_loss', summ_list=summary_list)

            trainable_vars = [hidden_layer.weights, hidden_layer.biases, hidden_layer.decode_biases]
            # Only optimize variables for this layer ("greedy")
            with tf.name_scope("train_step"):
                train_op = tf.train.AdamOptimizer(learning_rate=self.pretrain_lr).minimize(
                    loss, var_list=trainable_vars)
            sess.run(tf.initialize_all_variables())

            # Merge summaries and get a summary writer
            merged = tf.merge_summary(summary_list)
            pretrain_writer = tf.train.SummaryWriter("model/" + hidden_layer.name, sess.graph)

            step = 0
            for i in range(epoch):
                np.random.shuffle(data)
                batches = [_ for _ in utilities.gen_batches(data, FLAGS.batch_size)]
                for batch_x_original in batches:
                    sess.run(train_op, feed_dict={x_original: batch_x_original})

                    if step % self.print_step == 0:
                        loss_value = sess.run(loss, feed_dict={x_original: batch_x_original})
                        endoce_mean = sess.run(tf.reduce_mean(encoded), feed_dict={x_original: batch_x_original})
                        print("Step %s, batch %s loss = %s, weights_mean=%s" % (step, 'l2_loss', loss_value, endoce_mean))

                    if step % FLAGS.log_step == 0:
                        summary = sess.run(merged, feed_dict={x_original: batch_x_original})
                        pretrain_writer.add_summary(summary, global_step=step)

                    # Break for debugging purposes
                    if FLAGS.debug and step > 5:
                        break
                    step += 1
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
        # if batch_method == "random":
        x_train = data.train.images
        y_label = data.train.labels
        shuff = zip(x_train, y_label)
            # np.random.shuffle(shuff)
            # xy_train = [_ for _ in utilities.gen_batches(shuff, FLAGS.batch_size)]
        # else:
        #     x_train = data.train.images
        #     y_label = data.train.labels
        #     shuff = zip(x_train, y_label)
        #     xy_train = [_ for _ in utilities.gen_batches(shuff, FLAGS.batch_size)]
        return self.finetune_parameters_gen(xy_train_gen=shuff, output_dim=output_dim, epochs=epochs)

    def finetune_parameters_gen(self, xy_train_gen, output_dim, epochs):
        """An implementation of finetuning to support data feeding from generators."""
        sess = self.sess
        summary_list = []

        print("Starting to fine tune parameters of network.")
        with tf.name_scope("finetuning"):
            with tf.name_scope("inputs"):
                x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="raw_input")
                with tf.name_scope("fully_encoded"):
                    x_encoded = self.get_encoded_input(x, depth=-1)  # Full depth encoding

            """Note on W below: The difference between self.output_dim and output_dim is that the former
            is the output dimension of the autoencoder stack, which is the dimension of the new feature
            space. The latter is the dimension of the y value space for classification. Ex: If the output
            should be binary, then the output_dim = 2."""
            with tf.name_scope("softmax_variables"):
                self.W = weight_variable(self.output_dim, output_dim, name="weights")
                self.b = bias_variable(output_dim, initial_value=0, name="biases")
                attach_variable_summaries(self.W, self.W.name, summ_list=summary_list)
                attach_variable_summaries(self.b, self.b.name, summ_list=summary_list)
            with tf.name_scope("outputs"):
                y_logits = tf.matmul(x_encoded, self.W) + self.b
                with tf.name_scope("predicted"):
                    y_pred = tf.nn.softmax(y_logits, name="y_pred")
                    attach_variable_summaries(y_pred, y_pred.name, summ_list=summary_list)
                with tf.name_scope("actual"):
                    y_actual = tf.placeholder(tf.float32, shape=[None, output_dim], name="y_actual")
                    attach_variable_summaries(y_actual, y_actual.name, summ_list=summary_list)

            with tf.name_scope("cross_entropy"):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_logits, y_actual))
                attach_scalar_summary(cross_entropy, "cross_entropy", summ_list=summary_list)

            trainable_vars = self.get_all_variables(additional_layer=[self.W, self.b])

            with tf.name_scope("train_step"):
                train_step = tf.train.AdamOptimizer(learning_rate=self.finetune_lr).minimize(
                    cross_entropy, var_list=trainable_vars)

            with tf.name_scope("evaluation"):
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                attach_scalar_summary(accuracy, "finetune_accuracy", summ_list=summary_list)

            sess.run(tf.initialize_all_variables())

            # Merge summaries and get a summary writer
            merged = tf.merge_summary(summary_list)
            train_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/train/finetune", sess.graph)

            step = 0
            for i in range(epochs):
                np.random.shuffle(xy_train_gen)
                train_data_batchs = [_ for _ in utilities.gen_batches(xy_train_gen, FLAGS.batch_size)]
                for batch in train_data_batchs:
                    batch_xs, batch_ys = zip(*batch)
                    # print 'get xs batch size===', len(batch_xs), type(batch_xs[0]), batch_xs[0].shape
                    # print 'get ys batch size===', len(batch_ys), type(batch_ys[0]), batch_ys[0].shape
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
            print("Completed fine-tuning of parameters.")
            tuned_params = {"layer1_weights": sess.run(self.hidden_layers[0].get_weight_variable()), "layer2_weights": sess.run(self.hidden_layers[1].get_weight_variable()),
                            "layer3_weights": sess.run(self.hidden_layers[2].get_weight_variable()), "weights": sess.run(self.W), "biases": sess.run(self.b)}
            return tuned_params

    def evaluation(self, output_dim):
        data = input_data.read_data_sets("data/MNIST", one_hot=True)
        # if batch_method == "random":
        x_train = data.test.images
        y_label = data.test.labels
        x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="raw_test_input")
        y_actual = tf.placeholder(tf.float32, shape=[None, output_dim], name="y_test_actual")
        x_encoded = self.get_encoded_input(x, depth=-1)  # Full depth encoding
        y_logits = tf.matmul(x_encoded, self.W) + self.b
        y_pred = tf.nn.softmax(y_logits, name="y_pred")
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("test, batch accuracy: " ,
              sess.run(accuracy, feed_dict={x: x_train, y_actual: y_label}))


def draw_weights(weights, name, N_COL, N_ROW, sub_factor, use_rondam):
    weights = (weights - weights.min())/(weights.max() - weights.min())
    weights = weights.T
    size = np.sqrt(weights.shape[1])
    total_weight = weights.shape[0]
    print weights.shape
    weights = weights.reshape(weights.shape[0], size, size)
    print weights.shape

    # N_COL = 10
    # N_ROW = 2
    plt.figure(figsize=(N_COL, N_ROW * 2.5))
    for row in range(N_ROW):
        for col in range(N_COL):
            if use_rondam:
                i = row * N_COL + col
                j = np.random.randint(total_weight-sub_factor, size=1)
                data = weights[i]
                val = i+j
                data2 = weights[val]
                # Draw Input Data(x)
                plt.subplot(2 * N_ROW, N_COL, 2 * row * N_COL + col + 1)
                plt.title('IN:%02d' % i)
                plt.imshow(data, cmap="gray", clim=(0, 1.0), origin='upper')
                # plt.imshow(data.reshape((28, 28)))
                plt.tick_params(labelbottom="off")
                plt.tick_params(labelleft="off")
            else:
                i = row * N_COL + col
                data = weights[i]
                # Draw Input Data(x)
                plt.subplot(N_ROW, N_COL, row * N_COL + col + 1)
                plt.title('IN:%02d' % i)
                plt.imshow(data, cmap="gray", clim=(0, 1.0), origin='upper')
                # plt.imshow(data.reshape((28, 28)))
                plt.tick_params(labelbottom="off")
                plt.tick_params(labelleft="off")


    plt.savefig("result_weights_layer_"+name+".png")
    plt.show()


if __name__ == '__main__':
    # Start a TensorFlow session
    sess = tf.Session()

    # Initialize an unconfigured autoencoder with specified dimensions, etc.
    sda = SDAutoencoder(dims=[784, 576, 400, 225],
                        activations=["relu", "relu", "relu"],
                        sess=sess,
                        noise=0.3)

    # Pretrain weights and biases of each layer in the network.
    sda.pre_train_network(100)
    # Read in test y-values to softmax classifier.
    tuned_params = sda.finetune_parameters(epochs=100, output_dim=10)
    sda.evaluation(10)

    draw_weights(tuned_params['layer1_weights'], 'layer1', 10, 2, 30, True)
    draw_weights(tuned_params['layer2_weights'], 'layer2', 10, 2, 30, True)
    draw_weights(tuned_params['layer3_weights'], 'layer3', 10, 2, 30, True)
    draw_weights(tuned_params['weights'], 'output', 5, 2, 0, False)
    # Write to file the newly represented features.
    # sda.write_encoded_input(filepath="data/transformed.csv", x_test_path=FLAGS.x_train_path)

    # data = input_data.read_data_sets("data/MNIST", one_hot=True)
    # temp_train = data.train.images
    # sda.pre_train_layer(0, temp_train, 400)

    # # Draw Encode/Decode Result
    # print 'begin draw encode and decode'
    # N_COL = 10
    # N_ROW = 2
    # plt.figure(figsize=(N_COL, N_ROW * 2.5))
    # np.random.shuffle(temp_train)
    # batches = [_ for _ in utilities.gen_batches(temp_train, FLAGS.batch_size)]
    # batch_xs = batches[0]
    # for row in range(N_ROW):
    #     for col in range(N_COL):
    #         i = row * N_COL + col
    #         data = batch_xs[i:i + 1]
    #         data = np.array(data)
    #         # Draw Input Data(x)
    #         plt.subplot(2 * N_ROW, N_COL, 2 * row * N_COL + col + 1)
    #         plt.title('IN:%02d' % i)
    #         plt.imshow(data.reshape((28, 28)), cmap="gray", clim=(0, 1.0), origin='upper')
    #         # plt.imshow(data.reshape((28, 28)))
    #         plt.tick_params(labelbottom="off")
    #         plt.tick_params(labelleft="off")
    #
    #         # Draw Output Data(y)
    #         plt.subplot(2 * N_ROW, N_COL, 2 * row * N_COL + N_COL + col + 1)
    #         plt.title('OUT:%02d' % i)
    #         output = sda.hidden_layers[0].encode(data)
    #         output = sda.hidden_layers[0].decode(output)
    #         y_value = output.eval(session=sess)
    #         # print y_value.shape
    #         # plt.imshow(y_value.reshape((28, 28)))
    #         plt.imshow(y_value.reshape((28, 28)), cmap="gray", clim=(0, 1.0), origin='upper')
    #         plt.tick_params(labelbottom="off")
    #         plt.tick_params(labelleft="off")
    #
    # plt.savefig("result.png")
    # plt.show()






