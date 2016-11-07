import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in+fan_out))
    high = constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=np.float32, name='weights_init')


class VariationalAutoEncoder(object):
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001,
                 batch_size=100):
        self.network_architecture =network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, shape=[None, network_architecture])


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)