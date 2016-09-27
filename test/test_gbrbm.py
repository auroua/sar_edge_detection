from tf_unsupervised.tf_grbm import GBRBM
import numpy as np
import tensorflow as tf
import unittest


if __name__ == '__main__':
    gb = GBRBM(3, 2, batch_size=2)
    # mean, samples = gb.sample_hidden_given_visible(gb.input_img)
    v1 = np.array([[1, 0], [1, 0]])
    samples = tf.constant(v1, dtype=tf.float32)
    mean_gaussian = gb.propdown_mean(samples)
    mean_visible, samples_gaussian = gb.sample_visible_from_gaussian(mean_gaussian)
    inputs = np.array([[1, 0, 1], [0, 1, 1]])
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    # for i in range(10):

    print sess.run([samples, mean_gaussian, samples_gaussian], feed_dict={gb.input_img: inputs})