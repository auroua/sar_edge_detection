from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='input_x')
    y_ = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='input_y')

    # # softmax version dnn
    # w = tf.Variable(tf.zeros([784, 10]), name='w')
    # b = tf.Variable(tf.zeros([10]), name='b')
    #
    # y = tf.matmul(x, w) + b
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    #
    # sess = tf.InteractiveSession()
    # sess.run(tf.initialize_all_variables())
    # for i in range(1000):
    #     batch = mnist.train.next_batch(100)
    #     train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    #
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    # print sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})

    # cnn
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # conv-1
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # conv-2
    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # fc1
    w_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fc2
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_predict = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run([train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % sess.run(accuracy, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

