import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utilities
from util import *
from tensorflow.examples.tutorials.mnist import input_data
from mstar_patch_batch_generator import get_train_dataset, get_test_dataset, get_test_target_dataset, get_test_back_dataset, get_test_bg_dataset
import os
import time
import cv2
from sklearn.metrics import confusion_matrix
from theano_rbm.data_process import getFiles

__version__ = '0.1'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', 'data', 'data dir')
flags.DEFINE_integer('drop_out_rate', 0.0, 'the drop out rate')
flags.DEFINE_string('activations', 'relu', 'Type of activations. ["sigmoid", "tanh", "relu"]')
flags.DEFINE_string('x_train_path', 'data/MNIST', 'Train file path')
flags.DEFINE_string('batch_method', 'random', 'How to generate data')
flags.DEFINE_integer('epochs', 100, 'Train file path')
flags.DEFINE_integer('log_step', 100, 'tensorflow writer log data')
flags.DEFINE_integer('batch_size', 256, 'training batch size')
flags.DEFINE_boolean('debug', False, 'debug the system')
flags.DEFINE_string('log_dir', 'model', 'the log dir')
flags.DEFINE_float('deacy_factor', 0.3, 'neural networks weight deacy factors')
flags.DEFINE_integer('train_set_size', 358101, 'neural networks weight deacy factors')
flags.DEFINE_string('is_train', 'test', 'used for stage, option value is [pre_train, fine_tune, test]')
flags.DEFINE_float('deacy', 0.95, 'batch_normalization weights mean average')
flags.DEFINE_float('epsilon', 1e-3, 'batch_normalization weights mean average')

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

('test, batch accuracy: ', 0.80240768)

def print_confusion_matrix(cls_true, cls_pred, num_classes):
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

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

        # batch normalization
        self.scale = tf.Variable(tf.ones([output_dim]), name='bn_scale')
        self.beta = tf.Variable(tf.zeros([output_dim]), name='bn_beta')
        self.pop_mean = tf.Variable(tf.zeros([output_dim]), trainable=False, name='bn_pop_mean')
        self.pop_var = tf.Variable(tf.ones([output_dim]), trainable=False, name='bn_pop_var')

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
        if FLAGS.is_train == 'pre_train':
            batch_mean, batch_var = tf.nn.moments(tf.matmul(input_tensor, self.weights), [0])
            return self.activate(
                tf.nn.batch_normalization(tf.matmul(input_tensor, self.weights), batch_mean, batch_var, self.beta,
                                          self.scale, FLAGS.epsilon))
            # return self.activate(tf.matmul(input_tensor, self.weights) + self.biases)
        elif FLAGS.is_train == 'fine_tune':
            batch_mean, batch_var = tf.nn.moments(tf.matmul(input_tensor, self.weights), [0])
            train_mean = tf.assign(self.pop_mean, self.pop_mean * FLAGS.deacy + batch_mean * (1 - FLAGS.deacy))
            train_var = tf.assign(self.pop_var, self.pop_var * FLAGS.deacy + batch_var * (1 - FLAGS.deacy))
            with tf.control_dependencies([train_mean, train_var]):
                return self.activate(tf.nn.batch_normalization(tf.matmul(input_tensor, self.weights), batch_mean, batch_var, self.beta, self.scale, FLAGS.epsilon))
                # return tf.nn.batch_normalization(tf.matmul(input_tensor, self.weights), batch_mean, batch_var, self.beta, self.scale, FLAGS.epsilon)
            # return self.activate(tf.matmul(input_tensor, self.weights) + self.biases)
        elif FLAGS.is_train == 'test':
            # return tf.nn.batch_normalization(tf.matmul(input_tensor, self.weights), self.pop_mean, self.pop_var, self.beta, self.scale, FLAGS.epsilon)
            return self.activate(tf.nn.batch_normalization(tf.matmul(input_tensor, self.weights), self.pop_mean, self.pop_var, self.beta, self.scale, FLAGS.epsilon))
            # return self.activate(tf.matmul(input_tensor, self.weights) + self.biases)

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

    def __init__(self, output_dim, dims, activations, sess, noise=0.0, pretrain_lr=0.001, finetune_lr=0.001,
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

        with tf.name_scope("softmax_variables"):
            self.W = weight_variable(self.output_dim, output_dim, name="weights")
            self.b = bias_variable(output_dim, initial_value=0, name="biases")
            # attach_variable_summaries(self.W, self.W.name)
            # attach_variable_summaries(self.b, self.b.name, summ_list=summary_list)
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

    def get_all_variables_weights(self, additional_layer=None):
        all_variables = []
        for layers in self.hidden_layers:
            all_variables.extend([layers.get_weight_variable()])
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
            # print 'get encode input hidden layers ', i
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
    def pre_train_network(self, epochs, data):
        print 'Starting to pretrain autoencoder network.'
        x_train = data
        for i in range(len(self.hidden_layers)):
            self.pre_train_layer(i, x_train, epochs)
        print 'Finished pretraining of autoencoder network.'

    # reference weight deacy in tensorflow
    # # Create your variables
    # weights = tf.get_variable('weights', collections=['variables'])
    #
    # with tf.variable_scope('weights_norm') as scope:
    #     weights_norm = tf.reduce_sum(
    #         input_tensor=WEIGHT_DECAY_FACTOR * tf.pack(
    #             [tf.nn.l2_loss(i) for i in tf.get_collection('weights')]
    #         ),
    #         name='weights_norm'
    #     )
    #
    # # Add the weight decay loss to another collection called losses
    # tf.add_to_collection('losses', weights_norm)
    #
    # # Add the other loss components to the collection losses
    # # ...
    #
    # # To calculate your total loss
    # tf.add_n(tf.get_collection('losses'), name='total_loss')

    def pre_train_layer(self, depth, data, epoch):
        self.pretrain_lr = 0.01
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
                # attach_variable_summaries(encoded, "encoded", summ_list=summary_list)
                # attach_variable_summaries(decoded, "decoded", summ_list=summary_list)
                attach_variable_summaries(hidden_layer.get_weight_variable(), "weights", summ_list=summary_list)

            with tf.name_scope('weights_norm'):
                weights_norm = tf.reduce_sum(input_tensor=FLAGS.deacy_factor * tf.nn.l2_loss(hidden_layer.get_weight_variable()),
                                         name='weights_norm')
            # Reconstruction loss
            with tf.name_scope("reconstruction_loss"):
                # loss = self.get_loss(x_latent, decoded)
                val_loss = self.get_l2_loss(x_latent, decoded)
                loss = val_loss + weights_norm
                attach_scalar_summary(loss, "%s_loss" % 'l2_loss', summ_list=summary_list)
                # attach_scalar_summary(self.pretrain_lr, 'pretrain_lr', summ_list=summary_list)
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
                # if epoch % 5 == 0:
                #     if self.pretrain_lr >= 0.00001:
                #         self.pretrain_lr /= 2.0
            print("Finished pretraining of layer %d. Updated layer weights and biases." % depth)

    ##############
    # FINETUNING #
    ##############
    def finetune_parameters(self, output_dim, data, label, epochs=1, batch_method="random"):
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
        # if batch_method == "random":
        x_train = data
        y_label = label
        shuff = zip(x_train, y_label)
        return self.finetune_parameters_gen(xy_train_gen=shuff, output_dim=output_dim, epochs=epochs)

    def finetune_parameters_gen(self, xy_train_gen, output_dim, epochs):
        """An implementation of finetuning to support data feeding from generators."""
        sess = self.sess
        summary_list = []
        batch_s = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            0.0001,  # Base learning rate.
            batch_s,  # Current index into the dataset.
            30000,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)

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
            with tf.name_scope("outputs"):
                y_logits = tf.matmul(x_encoded, self.W) + self.b
                with tf.name_scope("predicted"):
                    y_pred = tf.nn.softmax(y_logits, name="y_pred")
                    # attach_variable_summaries(y_pred, y_pred.name, summ_list=summary_list)
                with tf.name_scope("actual"):
                    y_actual = tf.placeholder(tf.float32, shape=[None, output_dim], name="y_actual")
                    # attach_variable_summaries(y_actual, y_actual.name, summ_list=summary_list)

            trainable_vars = self.get_all_variables(additional_layer=[self.W, self.b])
            trainable_weights = self.get_all_variables_weights(additional_layer=[self.W])

            with tf.name_scope('weights_norm'):
                weights_norm = tf.reduce_sum(
                    input_tensor=tf.reduce_mean(tf.pack([FLAGS.deacy_factor * tf.nn.l2_loss(weight) for weight in trainable_weights])),
                    name='weights_norm')

            with tf.name_scope("cross_entropy"):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_logits, y_actual))
                # loss = cross_entropy + weights_norm
                loss = cross_entropy
                attach_scalar_summary(cross_entropy, "cross_entropy", summ_list=summary_list)
                attach_scalar_summary(learning_rate, 'finetune_lr', summ_list=summary_list)
                # attach_scalar_summary(loss, 'loss_val', summ_list=summary_list)

            with tf.name_scope("train_step"):
                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                    loss, var_list=trainable_vars, global_step=batch_s)
                # # Use simple momentum for the optimization.
                # train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, var_list=trainable_vars, global_step=batch_s)

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
                # print len(xy_train_gen)
                train_data_batchs = [_ for _ in utilities.gen_batches(xy_train_gen, FLAGS.batch_size)]
                for batch in train_data_batchs:
                    batch_xs, batch_ys = zip(*batch)
                    # print 'get xs batch size===', len(batch_xs), type(batch_xs[0]), batch_xs[0].shape
                    # print 'get ys batch size===', len(batch_ys), type(batch_ys[0]), batch_ys[0].shape
                    if step % self.print_step == 0:
                        print("Step %s, batch accuracy: " % step,
                              sess.run(accuracy, feed_dict={x: batch_xs, y_actual: batch_ys}))
                        # print('pop_mean in hidden layer 1 is :', sess.run(self.hidden_layers[1].pop_mean))

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
                            "layer3_weights": sess.run(self.hidden_layers[2].get_weight_variable()), "weights": sess.run(self.W), "biases": sess.run(self.b),
                            "layer1_pop_mean": sess.run(self.hidden_layers[1].pop_mean), "layer1_pop_var": sess.run(self.hidden_layers[1].pop_var)}
            return tuned_params

    def evaluation(self, output_dim, test_data, test_label):
        # if batch_method == "random":
        x_train = test_data
        y_label = test_label
        x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="raw_test_input")
        y_actual = tf.placeholder(tf.float32, shape=[None, output_dim], name="y_test_actual")
        x_encoded = self.get_encoded_input(x, depth=-1)  # Full depth encoding
        y_logits = tf.matmul(x_encoded, self.W) + self.b
        y_pred = tf.nn.softmax(y_logits, name="y_pred")
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pred_val, acc = sess.run([y_pred, accuracy], feed_dict={x: x_train, y_actual: y_label})
        print("test, batch accuracy: ", acc)
        # print('pop_mean values : ',  sess.run(self.hidden_layers[1].pop_mean))
        return pred_val


    def get_label(self, test_data):
        # if batch_method == "random":
        x_train = test_data
        x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="raw_test_input")
        x_encoded = self.get_encoded_input(x, depth=-1)  # Full depth encoding
        y_logits = tf.matmul(x_encoded, self.W) + self.b
        y_pred = tf.nn.softmax(y_logits, name="y_pred")
        pred_val = tf.argmax(y_pred, 1)
        pred, value = sess.run([y_pred, pred_val], feed_dict={x: x_train})
        # print("test, batch accuracy: " ,
        #       sess.run([y_pred, pred_val], feed_dict={x: x_train}))
        return pred, value


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
    pre_train_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/25_25_new/pre_train'
    target_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/25_25_new/target_patch'
    back_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/25_25_new/shadow_patch'
    bg_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/25_25_new/bg_patch'

    target_test_set_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/25_25_new/target_test'
    back_test_set_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/25_25_new/shadow_test'
    bg_test_set_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/25_25_new/bg_test'
    pre_train_data, fine_tune_data, fine_tune_label = get_train_dataset(pre_train_path, target_path, back_path, bg_path)
    test_data, test_label = get_test_dataset(target_test_set_path, back_test_set_path, bg_test_set_path)
    test_target_data, test_target_label = get_test_target_dataset(target_test_set_path, back_test_set_path, bg_test_set_path)
    test_back_data, test_back_label = get_test_back_dataset(target_test_set_path, back_test_set_path, bg_test_set_path)
    test_bg_data, test_bg_label = get_test_bg_dataset(target_test_set_path, back_test_set_path, bg_test_set_path)

    sess = tf.Session()
    # # Initialize an unconfigured autoencoder with specified dimensions, etc.
    sda = SDAutoencoder(3, dims=[625, 300, 200, 100],
                        activations=["relu", "relu", "relu"],
                        sess=sess,
                        noise=0.3)
    saver = tf.train.Saver()

    if FLAGS.is_train == 'pre_train':
        start_time = time.time()
        sda.pre_train_network(5, pre_train_data)
        duration = time.time() - start_time
        print('#########per_train cost (%.3f sec)' % (duration))
        # Read in test y-values to softmax classifier.
        sda.save_variables('/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/model_sar_patch/pre_train/model_pre_train')
    elif FLAGS.is_train == 'fine_tune':
        PATCH_SIZE = 25
        # load saved model
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/model_sar_patch/pre_train/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        tuned_params = sda.finetune_parameters(epochs=300, output_dim=3, data=fine_tune_data, label=fine_tune_label)
        sda.evaluation(3, test_data=test_data, test_label=test_label)
        sda.save_variables('/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/model_sar_patch/fine_tune/model_fintune')
    elif FLAGS.is_train == 'test':
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/model_sar_patch/fine_tune/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        PATCH_SIZE = 25
        # img_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei/HB03848.015.jpg'
        img_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei/HB03333.015.jpg'
        img_test = cv2.imread(img_url, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        total_data = np.zeros((104*104, PATCH_SIZE*PATCH_SIZE), dtype=np.float32)
        count = 0
        img_test2 = img_test.astype(np.float32)
        img_test2 /= 255
        img_copy = img_test2.copy()

        for i in range(img_copy.shape[0]):
            for j in range(img_copy.shape[1]):
                img_copy[i, j] = 1.0
        print 'after initialization'
        for i in range(12, img_test2.shape[0]-13):
            for j in range(12, img_test2.shape[1]-13):
                patch = img_test2[i-12:i+13, j-12:j+13]
                patch = patch.reshape(1, PATCH_SIZE*PATCH_SIZE)
                total_data[count, :] = patch
                count += 1
        print 'total_data size :', len(total_data)
        pred, value = sda.get_label(total_data)

        count = 0
        for i in range(12, img_test2.shape[0]-13):
            for j in range(12, img_test2.shape[1]-13):
                img_copy[i, j] = value[count]
                count += 1
        img_copy /= 2
        cv2.imshow('segment', img_copy)
        cv2.waitKey(0)