import tensorflow as tf
import numpy as np
import time
import utilities
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS

# flags.DEFINE_integer('input_layer_size', 784, 'input layer size')
# flags.DEFINE_integer('hidden_layer1_size', 640, 'hidden layer 1 size')
# flags.DEFINE_integer('batch_size', 100, 'batch size')
# flags.DEFINE_integer('epochs', 100, 'max epoch')
# flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
# flags.DEFINE_float('prob', 0.5, 'drop out probability')
flags.DEFINE_string('corr_type', 'salt_and_pepper', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_float('corr_frac', 0.4, 'Fraction of the input to corrupt.')


class tf_daw(object):
    def __init__(self, weights, hbias, vbias, input1, input2, keep_prob, name_scope):
        self.w = weights
        self.hb = hbias
        self.vb = vbias
        self.input_noise = input1
        self.input_correct = input2
        self.name_scope = name_scope
        self.keep_prob = keep_prob
        self.global_step = tf.Variable(0.0, trainable=False, name='global_stpe')

    def inference(self):
        with tf.name_scope(self.name_scope+'inference'):
            input_hidden = tf.nn.softsign(tf.matmul(self.input_noise, self.w, name='input_weight') + self.hb, name='activition')
            drop_out = tf.nn.dropout(input_hidden, keep_prob=self.keep_prob, name='drop_out_layer')
            hidden_output = tf.nn.relu(tf.matmul(drop_out, tf.transpose(self.w), name='hidden_weight') + self.vb,
                                       name='activition')
        return hidden_output

    def loss(self, inference_output):
        input_size = tf.shape(self.input_noise)[0]
        loss_val = tf.nn.l2_loss(self.input_correct - inference_output)/tf.cast(input_size, tf.float32)
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
        input_size = tf.shape(self.input_noise)[0]
        error = tf.nn.l2_loss(self.input_correct - recon)/tf.cast(input_size, tf.float32)
        return error

    def run_train(self, o_train_set_x, sess, writer, summary):
        output_train = self.inference()
        loss_train = self.loss(output_train)
        optimize_train = self.train(loss_train)
        self.summary_parameter(loss_train)
        x_corrupted_train = _corrupt_input(o_train_set_x)
        shuff_train = zip(o_train_set_x, x_corrupted_train)
        for step in range(FLAGS.epochs):
            np.random.shuffle(shuff_train)
            batches = [_ for _ in utilities.gen_batches(shuff_train, FLAGS.batch_size)]
            start_time = time.time()
            for batch in batches:
                x_batch, x_corr_batch = zip(*batch)
                _, loss_value, summary_val, output_val = sess.run([optimize_train, loss_train, summary, output_train],
                                                                  feed_dict={self.input_noise: x_corr_batch,
                                                                             self.input_correct: x_batch})
                writer.add_summary(summary_val)
            duration = time.time() - start_time
            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)   %.2f ' % (step, loss_value, duration, np.mean(output_val)))
            writer.flush()


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

if __name__=='__main__':
    data_url = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/theano_rbm/data/origin_target_train_28.npy'
    o_train_set_x = np.load(data_url)
    print o_train_set_x.shape
    w = tf.Variable(tf.random_uniform(shape=[FLAGS.input_layer_size, FLAGS.hidden_layer1_size],
                        minval=-4 * np.sqrt(6. / (FLAGS.input_layer_size + FLAGS.hidden_layer1_size)),
                        maxval=4 * np.sqrt(6. / (FLAGS.hidden_layer1_size + FLAGS.input_layer_size))), name='input_weight')
    hb = tf.Variable(tf.zeros(FLAGS.hidden_layer1_size), name='hidden_layer1_bias_size')
    vb = tf.Variable(tf.zeros(FLAGS.input_layer_size), name='hidden_layer1_visual_size')
    input_img_noise = tf.placeholder(tf.float32, shape=[None, FLAGS.input_layer_size], name='input_image')
    input_img_correct = tf.placeholder(tf.float32, shape=[None, FLAGS.input_layer_size], name='input_image')

    auto_encoder = tf_daw(w, hb, vb, input_img_noise, input_img_correct, FLAGS.prob, 'sda1')

    output = auto_encoder.inference()
    loss = auto_encoder.loss(output)
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
                                                              feed_dict={auto_encoder.input_noise: x_corr_batch,
                                                                         auto_encoder.input_correct: x_batch})
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
                                  feed_dict={auto_encoder.input_noise: data})
            # plt.imshow(y_value.reshape((28, 28)))
            plt.imshow(y_value.reshape((28, 28)), cmap="gray", clim=(0, 1.0), origin='upper')
            plt.tick_params(labelbottom="off")
            plt.tick_params(labelleft="off")

    plt.savefig("result.png")
    plt.show()


    # Draw Weights Result
    # print 'begin draw weights'
    # weights = sess.run(auto_encoder.w, feed_dict={auto_encoder.input_noise: x_batch})
    # weights = (weights - weights.min())/(weights.max() - weights.min())
    # weights = weights.T
    # print weights.shape
    # weights = weights.reshape(weights.shape[0], 28, 28)
    # print weights.shape
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
    #             val = i+j
    #             data2 = weights[val[0]]
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
    #             plt.title('IN:%02d' % val)
    #             plt.imshow(data2, cmap="gray", clim=(0, 1.0), origin='upper')
    #             plt.tick_params(labelbottom="off")
    #             plt.tick_params(labelleft="off")
    #
    # plt.savefig("result.png")
    # plt.show()