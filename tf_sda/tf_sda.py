import tensorflow as tf
import numpy as np
from tf_da_w import tf_daw, _corrupt_input
import utilities
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('pre_batch_size', 100, 'pre training batch size')
flags.DEFINE_float('pre_learning_rate', 0.001, 'pre_training learning rate')
flags.DEFINE_integer('input_layer_size', 784, 'input layer size')
flags.DEFINE_integer('hidden_layer1_size', 640, 'hidden layer 1 size')
flags.DEFINE_integer('hidden_layer2_size', 400, 'hidden layer 2 size')
flags.DEFINE_integer('hidden_layer3_size', 250,  'fully connected layer size')
flags.DEFINE_integer('output_class', 10, 'total category')

flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('epochs', 400, 'max epoch')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('prob', 0.5, 'drop out probability')


class SdA(object):
    def __init__(self):
        with tf.name_scope('input'):
            self.input_img_noise = tf.placeholder(tf.float32, shape=[None, FLAGS.input_layer_size], name='input_image')
            self.input_img_correct = tf.placeholder(tf.float32, shape=[None, FLAGS.input_layer_size], name='input_image')
            self.hidden_layer1_input = tf.placeholder(tf.float32, shape=[None, FLAGS.hidden_layer1_size], name='hidden_layer1_input')
            self.hidden_layer2_input = tf.placeholder(tf.float32, shape=[None, FLAGS.hidden_layer2_size], name='hidden_layer2_input')
            self.hidden_layer3_input = tf.placeholder(tf.float32, shape=[None, FLAGS.hidden_layer3_size], name='hidden_layer3_input')

        with tf.name_scope('layer1'):
            self.w = tf.Variable(tf.random_uniform(shape=[FLAGS.input_layer_size, FLAGS.hidden_layer1_size],
                            minval=-4*np.sqrt(6./(FLAGS.input_layer_size+FLAGS.hidden_layer1_size)),
                            maxval=4*np.sqrt(6./(FLAGS.hidden_layer1_size+FLAGS.input_layer_size))),  name='input_weight')
            self.hb = tf.Variable(tf.zeros(FLAGS.hidden_layer1_size), name='hidden_layer1_bias_size')
            self.vb = tf.Variable(tf.zeros(FLAGS.input_layer_size), name='hidden_layer1_visual_size')


        with tf.name_scope('layer2'):
            self.w1 = tf.Variable(tf.random_uniform(shape=[FLAGS.hidden_layer1_size, FLAGS.hidden_layer2_size],
                            minval=-4*np.sqrt(6./(FLAGS.hidden_layer2_size+FLAGS.hidden_layer1_size)),
                            maxval=4*np.sqrt(6./(FLAGS.hidden_layer1_size+FLAGS.hidden_layer2_size))),  name='hidden_layer1_weight')
            self.hb1 = tf.Variable(tf.zeros(FLAGS.hidden_layer2_size), name='hidden_layer2_bias')
            self.vb1 = tf.Variable(tf.zeros(FLAGS.hidden_layer1_size), name='hidden_layer2_visual_size')

        with tf.name_scope('layer3'):
            self.w2 = tf.Variable(tf.random_uniform(shape=[FLAGS.hidden_layer2_size, FLAGS.hidden_layer3_size],
                            minval=-4*np.sqrt(6./(FLAGS.hidden_layer2_size+FLAGS.hidden_layer3_size)),
                            maxval=4*np.sqrt(6./(FLAGS.hidden_layer3_size+FLAGS.hidden_layer2_size))),  name='hidden_layer2_weight')
            self.b2 = tf.Variable(tf.zeros(FLAGS.hidden_layer3_size), name='hidden_layer3_bias')
            self.vb2 = tf.Variable(tf.zeros(FLAGS.hidden_layer2_size), name='hidden_layer3_visual_size')

        with tf.name_scope('output_layer'):
            self.w3 = tf.Variable(tf.random_uniform(shape=[FLAGS.hidden_layer3_size, FLAGS.output_class],
                            minval=-4*np.sqrt(6./(FLAGS.output_class+FLAGS.hidden_layer3_size)),
                            maxval=4*np.sqrt(6./(FLAGS.hidden_layer3_size+FLAGS.output_class))),  name='output_layer_weight')
            self.b3 = tf.Variable(tf.zeros(FLAGS.output_class), name='output_layer_bias')

    def pre_train(self, o_train_set_x, sess, writer, summary):
        layer1 = tf_daw(self.w, self.hb, self.vb, self.input_img_noise, self.input_img_correct, FLAGS.prob, 'sda1')
        layer1.run_train(o_train_set_x, sess, writer, summary)


if __name__ == '__main__':
    data_url = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/theano_rbm/data/origin_target_train_28.npy'
    o_train_set_x = np.load(data_url)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    writer = tf.train.SummaryWriter('model', sess.graph)
    summary = tf.merge_all_summaries()


    stack_ae = SdA()
    stack_ae.pre_train(o_train_set_x, sess, writer, summary)