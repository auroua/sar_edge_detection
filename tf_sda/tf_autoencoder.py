import tensorflow as tf
import input_data
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import utilities

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('hidden_size', 256, 'hidden_layer_size')
flags.DEFINE_integer('max_iter', 40000, 'max iteration times')
flags.DEFINE_float('learning_rate', 0.01, 'learning rage')
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('image_size', 784, 'image pixel size')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
flags.DEFINE_string('train_dir', 'data', 'data dir')
flags.DEFINE_integer('drop_out_rate', 0.5, 'the drop out rate')
flags.DEFINE_string('corr_type', 'salt_and_pepper', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_float('corr_frac', 0.3, 'Fraction of the input to corrupt.')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def place_holder_inputs():
    x_ = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    return x_, y_


def noise_place_holder_inputs():
    train_img = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size])
    corr_img = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size])
    return train_img, corr_img


def get_dict(image, label, data_set):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {image: images_feed, label:labels_feed}
    return feed_dict


# def gen_dict2(train_img, corr_img):
#     feed_dict = {image_placeholder: train_img, corr_img_placeholder: corr_img}
#     return feed_dict


def inference(images):
    #hidden layer
    with tf.name_scope('hidden_layer'):
        w1 = tf.Variable(tf.truncated_normal([FLAGS.image_size, FLAGS.hidden_size], stddev=1.0/FLAGS.image_size), name='weights')
        b1 = tf.Variable(tf.zeros([FLAGS.hidden_size]), name='bias')
        # h_layer1 = tf.nn.relu(tf.matmul(images, w1)+b1)
        h_layer1 = tf.nn.softsign(tf.matmul(images, w1)+b1)
        h_drop_out = tf.nn.dropout(h_layer1, FLAGS.drop_out_rate)

    with tf.name_scope('output_layer'):
        b2 = tf.Variable(tf.zeros([FLAGS.image_size]), name='bias')
        output = tf.nn.relu(tf.matmul(h_drop_out, tf.transpose(w1)) + b2)
    return output

def loss_cross_entropy(inference_output, images):
    cross_entropy = -tf.mul(images, tf.log(inference_output), name='xentropy')
    cost = tf.reduce_mean(cross_entropy, name='cost_xentropy')
    return cross_entropy, cost

def loss_l2(inference_output, images):
    loss = tf.nn.l2_loss(inference_output - images)/FLAGS.batch_size
    tf.scalar_summary('l2_loss', loss)
    return loss

def train(cost):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # optimize = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    optimize = tf.train.AdamOptimizer()
    train_op = optimize.minimize(cost, global_step=global_step)
    return train_op

def evaluation(images, recon):
    cross_entropy = -tf.mul(images, tf.log(recon))
    cost = tf.recude_mean(cross_entropy)
    return cost


def _corrupt_input(self, data):
    """Corrupt a fraction of data according to the chosen noise method.

    :return: corrupted data
    """
    corruption_ratio = np.round(
        self.corr_frac * data.shape[1]).astype(np.int)

    if self.corr_type == 'none':
        return np.copy(data)

    if self.corr_frac > 0.0:
        if self.corr_type == 'masking':
            return utilities.masking_noise(
                data, self.tf_session, self.corr_frac)

        elif self.corr_type == 'salt_and_pepper':
            return utilities.salt_and_pepper_noise(data, corruption_ratio)
    else:
        return np.copy(data)


if __name__=='__main__':
    data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data, one_hot=True)
    with tf.Graph().as_default():
        # original ad
        image_placeholder, bias_placeholder = place_holder_inputs()
        # noise autoencoder
        # image_placeholder, corr_img_placeholder = noise_place_holder_inputs()
        logit = inference(image_placeholder)
        cost = loss_l2(logit, image_placeholder)
        train_op = train(cost)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.merge_all_summaries()
        # Add the variable initializer Op.
        init = tf.initialize_all_variables()
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        sess = tf.Session()
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
        sess.run(init)
        batchs_pre_epoch = data_sets.train.num_examples//FLAGS.batch_size
        # add noise
        # x_corrupted = _corrupt_input(data_sets.train.images)
        # shuff = zip(data_sets.train.images, x_corrupted)
        # print data_sets.train.num_examples, batchs_pre_epoch, FLAGS.batch_size
        for step in range(FLAGS.max_iter):
            # for i in range(int(batchs_pre_epoch)):
            #     np.random.shuffle(shuff)
            #     batches = [_ for _ in utilities.gen_batches(shuff, FLAGS.batch_size)]
                start_time = time.time()
                # for batch in batches:
                #     x_batch, x_corr_batch = zip(*batch)
                feed_dict = get_dict(image_placeholder, bias_placeholder, data_sets.train)
                _, loss_value = sess.run([train_op, cost], feed_dict=feed_dict)

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % 100 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    # Update the events file.
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

        checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)

        # Draw Encode/Decode Result
        print 'begin draw encode and decode'
        N_COL = 10
        N_ROW = 2
        plt.figure(figsize=(N_COL, N_ROW * 2.5))
        batch_xs, _ = data_sets.train.next_batch(N_COL * N_ROW)
        for row in range(N_ROW):
            for col in range(N_COL):
                i = row * N_COL + col
                data = batch_xs[i:i + 1]

                # Draw Input Data(x)
                plt.subplot(2 * N_ROW, N_COL, 2 * row * N_COL + col + 1)
                plt.title('IN:%02d' % i)
                plt.imshow(data.reshape((28, 28)), cmap="coolwarm", clim=(0, 1.0), origin='upper')
                plt.tick_params(labelbottom="off")
                plt.tick_params(labelleft="off")

                # Draw Output Data(y)
                plt.subplot(2 * N_ROW, N_COL, 2 * row * N_COL + N_COL + col + 1)
                plt.title('OUT:%02d' % i)
                y_value = logit.eval(session=sess, feed_dict={image_placeholder: data})
                plt.imshow(y_value.reshape((28, 28)), cmap="coolwarm", clim=(0, 1.0), origin='upper')
                plt.tick_params(labelbottom="off")
                plt.tick_params(labelleft="off")

        plt.savefig("result.png")
        plt.show()