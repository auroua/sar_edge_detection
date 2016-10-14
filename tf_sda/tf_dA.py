import tensorflow as tf
import numpy as np
import utilities
import time
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('hidden_layer', 500, 'hidden layer size')
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('epochs', 3000, 'max epoch')
flags.DEFINE_integer('input_size', 784, 'max epoch')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_float('prob', 0.5, 'drop out probability')
flags.DEFINE_string('corr_type', 'salt_and_pepper', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_float('corr_frac', 0.3, 'Fraction of the input to corrupt.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
flags.DEFINE_string('train_dir', 'data', 'data dir')


class dA(object):
    def __init__(self, input_size, hidden_layer, batch_size, epochs, learning_rate, keep_prob, name_scope):
        self.input_size = input_size
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.name_scope = name_scope
        self.keep_prob = keep_prob

        with tf.name_scope(name_scope+'variable'):
            self.input = tf.placeholder(tf.float32, shape=[None, input_size], name='input_image')
            self.input_with_out_noise = tf.placeholder(tf.float32, shape=[None, input_size], name='input_image')
            # self.w = tf.Variable(tf.truncated_normal([input_size, hidden_layer], stddev=1.0/FLAGS.input_size), name='weight')
            self.w = tf.Variable(tf.random_uniform(shape=[input_size, hidden_layer], minval=-4*np.sqrt(6./(self.hidden_layer+self.input_size)),
                                            maxval=4*np.sqrt(6./(self.hidden_layer+self.input_size))), name='weight')
            self.vbias = tf.Variable(tf.zeros([1, input_size]), name='v_bias', dtype=tf.float32)
            self.hbias = tf.Variable(tf.zeros([1, hidden_layer]), name='v_bias', dtype=tf.float32)
            self.global_step = tf.Variable(0.0, trainable=False, name='global_stpe')

    def inference(self):
        with tf.name_scope(self.name_scope+'inference'):
            # input_hidden = tf.nn.relu(tf.matmul(self.input, self.w, name='input_weight') + self.hbias, name='activition')
            input_hidden = tf.nn.softsign(tf.matmul(self.input, self.w, name='input_weight') + self.hbias, name='activition')
            # input_hidden = tf.nn.sigmoid(tf.matmul(self.input, self.w, name='input_weight') + self.hbias, name='activition')
            drop_out = tf.nn.dropout(input_hidden, keep_prob=self.keep_prob, name='drop_out_layer')
            hidden_output = tf.nn.relu(tf.matmul(drop_out, tf.transpose(self.w), name='hidden_weight') + self.vbias,
                                       name='activition')
        return hidden_output

    def loss(self, inference_output):
        input_size = tf.shape(self.input)[0]
        loss_val = tf.nn.l2_loss(self.input_with_out_noise - inference_output)/tf.cast(input_size, tf.float32)
        return loss_val

    def loss_corss_entropy(self, inference_output):
        # input_size = tf.shape(self.input)[0]
        corss_entropy = -tf.reduce_sum(self.input_with_out_noise*tf.log(inference_output), reduction_indices=[1], name='cross_entropy')
        loss_val = tf.reduce_mean(corss_entropy, name='loss')
        return loss_val

    def train(self, cost):
        # train = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train = tf.train.AdamOptimizer()
        optim = train.minimize(cost, global_step=self.global_step)
        return optim

    def summary_parameter(self, summary_loss):
        with tf.name_scope(self.name_scope + 'summary'):
            tf.scalar_summary('loss value', summary_loss, name='summary_loss')

    def evaluation(self, recon):
        input_size = tf.shape(auto_encoder.input)[0]
        error = tf.nn.l2_loss(self.input_with_out_noise - recon)/tf.cast(input_size, tf.float32)
        return error


def _corrupt_input(data):
    """Corrupt a fraction of data according to the chosen noise method.

    :return: corrupted data
    """
    corruption_ratio = np.round(
        FLAGS.corr_frac * data.shape[1]).astype(np.int)

    if FLAGS.corr_type == 'none':
        return np.copy(data)

    if FLAGS.corr_frac > 0.0:
        if FLAGS.corr_type == 'masking':
            return utilities.masking_noise(
                data, FLAGS.tf_session, FLAGS.corr_frac)

        elif FLAGS.corr_type == 'salt_and_pepper':
            return utilities.salt_and_pepper_noise(data, corruption_ratio)
    else:
        return np.copy(data)


def vis_weights(weights):
    weights = (weights - weights.min())/(weights.max() - weights.min())
    n = int(np.ceil(np.sqrt(weights.shape[0])))
    padding = (((0, n*2 - weights.shape[0]), (0, 1), (0, 1)) + ((0, 0),)*(weights.ndim-3))
    weights = np.pad(weights, padding, mode='constant', constant_values=1)
    weights = weights.reshape((n, n) + weights.shape[1:]).transpose((0, 2, 1, 3)+tuple(range(4, weights.ndim +1)))
    weights = weights.reshape((n*weights.shape[1], n*weights.shape[3])+weights.shape[4:])
    # plt.imshow(weights, cmap=cm.gray)
    plt.imshow(weights)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    data_url = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/theano_rbm/data/origin_target_train_28.npy'
    o_train_set_x = np.load(data_url)
    print o_train_set_x.shape

    auto_encoder = dA(FLAGS.input_size, FLAGS.hidden_layer, FLAGS.batch_size, FLAGS.epochs, FLAGS.learning_rate,
                     FLAGS.prob, 'sda1')
    output = auto_encoder.inference()
    loss = auto_encoder.loss(output)
    # loss = auto_encoder.loss_corss_entropy(output)
    optimize = auto_encoder.train(loss)
    auto_encoder.summary_parameter(loss)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    writer = tf.train.SummaryWriter('model', sess.graph)
    summary = tf.merge_all_summaries()

    x_corrupted = _corrupt_input(o_train_set_x)
    shuff = zip(o_train_set_x, x_corrupted)
    for step in range(FLAGS.epochs):
        np.random.shuffle(shuff)
        batches = [_ for _ in utilities.gen_batches(shuff, FLAGS.batch_size)]
        start_time = time.time()
        for batch in batches:
            x_batch, x_corr_batch = zip(*batch)
            _, loss_value, summary_val, output_val = sess.run([optimize, loss, summary, output],
                            feed_dict={auto_encoder.input: x_corr_batch, auto_encoder.input_with_out_noise: x_batch})
            writer.add_summary(summary_val)
        duration = time.time() - start_time
        # Write the summaries and print an overview fairly often.
        if step % 100 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)   %.2f ' % (step, loss_value, duration, np.mean(output_val)))
        writer.flush()

    # Draw Encode/Decode Result
    print 'begin draw encode and decode'
    N_COL = 10
    N_ROW = 2
    plt.figure(figsize=(N_COL, N_ROW * 2.5))
    np.random.shuffle(shuff)
    batches = [_ for _ in utilities.gen_batches(shuff, FLAGS.batch_size)]
    batch_xs, _ = x_batch, x_corr_batch = zip(*batches[0])
    for row in range(N_ROW):
        for col in range(N_COL):
                i = row * N_COL + col
                data = batch_xs[i:i + 1]
                data = np.array(data)
                # Draw Input Data(x)
                plt.subplot(2 * N_ROW, N_COL, 2 * row * N_COL + col + 1)
                plt.title('IN:%02d' % i)
                plt.imshow(data.reshape((28, 28)), cmap="gray", clim=(0, 1.0), origin='upper')
                # plt.imshow(data.reshape((28, 28)))
                plt.tick_params(labelbottom="off")
                plt.tick_params(labelleft="off")

                # Draw Output Data(y)
                plt.subplot(2 * N_ROW, N_COL, 2 * row * N_COL + N_COL + col + 1)
                plt.title('OUT:%02d' % i)
                y_value = output.eval(session=sess,
                                      feed_dict={auto_encoder.input: data})
                # plt.imshow(y_value.reshape((28, 28)))
                plt.imshow(y_value.reshape((28, 28)), cmap="gray", clim=(0, 1.0), origin='upper')
                plt.tick_params(labelbottom="off")
                plt.tick_params(labelleft="off")

    plt.savefig("result.png")
    plt.show()

    # Draw Weights Result
    # print 'begin draw weights'
    # weights = sess.run(auto_encoder.w, feed_dict={auto_encoder.input: x_batch})
    # weights = (weights - weights.min())/(weights.max() - weights.min())
    # weights = weights.T
    # weights = weights.reshape(weights.shape[0], 28, 28)
    #
    # N_COL = 10
    # N_ROW = 2
    # plt.figure(figsize=(N_COL, N_ROW * 2.5))
    # np.random.shuffle(shuff)
    # for row in range(N_ROW):
    #     for col in range(N_COL):
    #             i = row * N_COL + col
    #             j = np.random.randint(470, size=1)
    #             data = weights[i]
    #             data2 = weights[i+j]
    #             # Draw Input Data(x)
    #             plt.subplot(2 * N_ROW, N_COL, 2 * row * N_COL + col + 1)
    #             plt.title('IN:%02d' % i)
    #             plt.imshow(data, cmap="gray", clim=(0, 1.0), origin='upper')
    #             # plt.imshow(data.reshape((28, 28)))
    #             plt.tick_params(labelbottom="off")
    #             plt.tick_params(labelleft="off")
    #
    #             # Draw Output Data(y)
    #             plt.subplot(2 * N_ROW, N_COL, 2 * row * N_COL + N_COL + col + 1)
    #             plt.title('IN:%02d' % i+j)
    #             plt.imshow(data2, cmap="gray", clim=(0, 1.0), origin='upper')
    #             plt.tick_params(labelbottom="off")
    #             plt.tick_params(labelleft="off")
    #
    # plt.savefig("result.png")
    # plt.show()



