import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# http://r2rt.com/implementing-batch-normalization-in-tensorflow.html


def batch_norm_wrapper(inputs, is_training, decay = 0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)


def build_graph(is_training):
    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # Layer 1
    w1 = tf.Variable(w1_initial)
    z1 = tf.matmul(x,w1)
    bn1 = batch_norm_wrapper(z1, is_training)
    l1 = tf.nn.sigmoid(bn1)

    #Layer 2
    w2 = tf.Variable(w2_initial)
    z2 = tf.matmul(l1,w2)
    bn2 = batch_norm_wrapper(z2, is_training)
    l2 = tf.nn.sigmoid(bn2)

    # Softmax
    w3 = tf.Variable(w3_initial)
    b3 = tf.Variable(tf.zeros([10]))
    y  = tf.nn.softmax(tf.matmul(l2, w3))

    # Loss, Optimizer and Predictions
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    return (x, y_), train_step, accuracy, y, tf.train.Saver()


if __name__ == '__main__':
    mnist = input_data.read_data_sets('data/MNIST', one_hot=True)


    w1_initial = np.random.normal(size=(784, 100)).astype(np.float32)
    w2_initial = np.random.normal(size=(100, 100)).astype(np.float32)
    w3_initial = np.random.normal(size=(100, 10)).astype(np.float32)
    epsilon = 1e-3

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # Layer 1 without BN
    w1 = tf.Variable(w1_initial)
    b1 = tf.Variable(tf.zeros([100]))
    z1 = tf.matmul(x, w1)+b1
    l1 = tf.nn.sigmoid(z1)

    # Layer 1 with BN
    w1_BN = tf.Variable(w1_initial)
    # Note that pre-batch normalization bias is ommitted. The effect of this bias would be
    # eliminated when subtracting the batch mean. Instead, the role of the bias is performed
    # by the new beta variable. See Section 3.2 of the BN2015 paper.
    z1_BN = tf.matmul(x, w1_BN)
    # Calculate batch mean and variance
    batch_mean, batch_var1 = tf.nn.moments(z1_BN, [0])
    # Apply the initial batch normalizing transform
    z1_hat = (z1_BN - batch_mean) / tf.sqrt(batch_var1 + epsilon)
    # Create two new parameters, scale and beta
    scale1 = tf.Variable(tf.ones([100]))
    beta1 = tf.Variable(tf.zeros([100]))

    # Scale and shift to obtain the final output of the batch normalization
    # this value is fed into the activation function (here a sigmoid)
    BN1 = scale1 * z1_hat + beta1
    l1_BN = tf.nn.sigmoid(BN1)

    # Layer 2 without BN
    w2 = tf.Variable(w2_initial)
    b2 = tf.Variable(tf.zeros([100]))
    z2 = tf.matmul(l1, w2) + b2
    l2 = tf.nn.sigmoid(z2)

    # Layer 2 with BN, using Tensorflows built-in BN function
    w2_BN = tf.Variable(w2_initial)
    z2_BN = tf.matmul(l1_BN, w2_BN)
    batch_mean2, batch_var2 = tf.nn.moments(z2_BN, [0])
    scale2 = tf.Variable(tf.ones([100]))
    beta2 = tf.Variable(tf.zeros([100]))
    BN2 = tf.nn.batch_normalization(z2_BN, batch_mean2, batch_var2, beta2, scale2, epsilon)
    l2_BN = tf.nn.sigmoid(BN2)

    # Softmax
    w3 = tf.Variable(w3_initial)
    b3 = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(l2, w3) + b3)

    # Softmax layer3 with BN
    w3_BN = tf.Variable(w3_initial)
    b3_BN = tf.Variable(tf.zeros([10]))
    y_BN = tf.nn.softmax(tf.matmul(l2_BN, w3_BN) + b3_BN)

    # Loss, optimizer and predictions
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    cross_entropy_BN = -tf.reduce_sum(y_ * tf.log(y_BN))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    train_step_BN = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_BN)

    # Loss, optimizer and predictions
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    correct_prediction_BN = tf.equal(tf.arg_max(y_BN, 1), tf.arg_max(y_, 1))
    accuracy_BN = tf.reduce_mean(tf.cast(correct_prediction_BN, tf.float32))

    # Training the network
    zs, BNs, acc, acc_BN = [], [], [], []

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for i in range(40000):
        batch = mnist.train.next_batch(60)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        train_step_BN.run(feed_dict={x: batch[0], y_: batch[1]})
        if i % 50 is 0:
            res = sess.run([accuracy, accuracy_BN, z2, BN2],
                           feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            acc.append(res[0])
            acc_BN.append(res[1])
            zs.append(np.mean(res[2], axis=0))  # record the mean value of z2 over the entire test set
            BNs.append(np.mean(res[3], axis=0))  # record the mean value of BN2 over the entire test set
            print 'accuracy: ', res[0], 'BN accuracy: ', res[1]

    zs, BNs, acc, acc_BN = np.array(zs), np.array(BNs), np.array(acc), np.array(acc_BN)

    fig, ax = plt.subplots()

    ax.plot(range(0, len(acc) * 50, 50), acc, label='Without BN')
    ax.plot(range(0, len(acc) * 50, 50), acc_BN, label='With BN')
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0.8, 1])
    ax.set_title('Batch Normalization Accuracy')
    ax.legend(loc=4)
    plt.show()

    fig, axes = plt.subplots(5, 2, figsize=(6, 12))
    fig.tight_layout()

    for i, ax in enumerate(axes):
        ax[0].set_title("Without BN")
        ax[1].set_title("With BN")
        ax[0].plot(zs[:, i])
        ax[1].plot(BNs[:, i])
    plt.show()

    predictions = []
    correct = 0
    for i in range(100):
        pred, corr = sess.run([tf.arg_max(y_BN, 1), accuracy_BN],
                              feed_dict={x: [mnist.test.images[i]], y_: [mnist.test.labels[i]]})
        correct += corr
        predictions.append(pred[0])
    print("PREDICTIONS:", predictions)
    print("ACCURACY:", correct / 100)

    ##########################
    # fixed version for test #
    ##########################
    sess.close()
    tf.reset_default_graph()
    (x, y_), train_step, accuracy, _, saver = build_graph(is_training=True)

    acc = []
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(10000):
            batch = mnist.train.next_batch(60)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            if i % 50 is 0:
                res = sess.run([accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                acc.append(res[0])

                saved_model = saver.save(sess, 'temp-bn-save')

    print("Final accuracy:", acc[-1])


    tf.reset_default_graph()
    (x, y_), _, accuracy, y, saver = build_graph(is_training=False)

    predictions = []
    correct = 0
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, saved_model)
        for i in range(100):
            pred, corr = sess.run([tf.arg_max(y, 1), accuracy],
                                  feed_dict={x: [mnist.test.images[i]], y_: [mnist.test.labels[i]]})
            correct += corr
            predictions.append(pred[0])
    print("PREDICTIONS:", predictions)
    print("ACCURACY:", correct / 100)