import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in+fan_out))
    high = constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=np.float32, name='weights_init')


class VariationalAutoEncoder(object):
    def __init__(self, input_size, transfer_fct=tf.nn.softplus, learning_rate=0.001,
                 batch_size=100):
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_x = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name='input_x')
        self._create_network()
        self._create_loss_optimizer()
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.hidden_encoder = []
        self.hidden_decoder = []

    def _create_network(self):
        pass

    def _create_loss_optimizer(self):
        pass

    def _initialize_weights(self, n_hidden_encoder, n_hidden_decoder):
        assert n_hidden_encoder[-1] == n_hidden_decoder[0] and n_hidden_encoder[0] == n_hidden_decoder[-1], \
            "Invalid number of layers given by encoder and decoder."
        n_hidden_encoder_temp = n_hidden_encoder[0:-1]
        n_hidden_decoder_temp = n_hidden_decoder[0:-1]
        self.hidden_encoder = [net_layer(val, n_hidden_encoder[index+1], index) for index, val in n_hidden_encoder_temp]
        self.hidden_encoder.append(net_layer(n_hidden_encoder[-2], n_hidden_encoder[-1]))
        self.hidden_decoder = [net_layer(val, n_hidden_decoder[index+1], index) for index, val in n_hidden_decoder_temp]
        return self.hidden_encoder, self.hidden_decoder

    def _encoder_process(self):
        input_x = self.input_x
        hidden_encoder_temp = self.hidden_encoder[0:-2]
        for layer in hidden_encoder_temp:
            input_x = layer.encode(input_x, self.transfer_fct, 'encoder')
        z = self.hidden_encoder[-2].encode_without_activate(input_x, 'encoder')
        sigma = self.hidden_encoder[-1].encode_without_activate(input_x, 'encoder')
        return z, sigma

    def _decode_process(self, input_x):
        for layer in self.hidden_decoder:
            input_x = layer.encode(input_x, self.transfer_fct, 'decoder')
        return input_x

    def create_all_net(self):
        z, sigma = self._encoder_process()
        eps = tf.random_normal()


class net_layer(object):
    def __init__(self, input, output, index):
        self.input = input
        self.output = output
        self.index = index
        self.weights = tf.Variable(xavier_init(input, output), name='weights_layer_'+str(index))
        self.bias = tf.Variable(tf.zeros([output], dtype=tf.float32, name='bias_layer_'+str(index)))

    def encode(self, input_x, activate, type):
        return activate(tf.add(tf.matmul(input_x, self.w), self.bias), name=type+'_layer_'+self.index)

    def encode_without_activate(self, input_x, type):
        return tf.add(tf.matmul(input_x, self.w), self.bias, name=type+'_layer_'+self.index)


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.train.num_examples
    print n_samples
