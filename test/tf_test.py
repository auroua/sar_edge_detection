import numpy as np
import tensorflow as tf

if __name__=='__main__':
    a = np.array([[1., 100.0, 300.0], [10, 50, 250]])
    sigma = np.array([[0.3, 0.4, 10]])
    sigma2 = np.array([[0.3, 0.4, 10]])
    tf_mean = tf.constant(a, dtype=tf.float32)
    tf_reduce_mean = tf.reduce_mean(tf_mean, reduction_indices=[0])
    tf_mean_reshape = tf.reshape(tf_mean, shape=[1, -1])

    tf_mean_reshape_origin = tf.reshape(tf_mean_reshape, shape=[tf_mean.get_shape().as_list()[0], tf_mean.get_shape().as_list()[1]])
    tf_sigma = tf.constant(sigma, dtype=tf.float32)
    tf_contact_sigma = tf.concat(1, [tf_sigma, tf_sigma])
    # sample = tf.random_normal([1], mean=tf_mean, stddev=tf_sigma)
    sess = tf.Session()
    print sess.run([tf_reduce_mean])
    # print sess.run([tf_mean, tf_mean_reshape, tf_mean_reshape_origin, tf_contact_sigma])
    # print tf_mean.get_shape().as_list()
    # print tf_mean_reshape.get_shape().as_list()
    # print tf_mean_reshape_origin.get_shape().as_list()
    # print tf_contact_sigma.get_shape().as_list()
