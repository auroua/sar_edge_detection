import tensorflow as tf
import numpy as np

if __name__=='__main__':
    a = tf.Variable(tf.random_normal(shape=[2, 3]))
    b = tf.Variable(tf.random_normal(shape=[2, 3]))
    w = tf.Variable(tf.random_normal(shape=[3, 1]))

    d = tf.reduce_mean(tf.matmul(a, w))
    e = tf.reduce_mean(tf.matmul(b, w))

    f = tf.greater(d, e)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # print sess.run(f)
    # g = tf.cond(d > e, lambda: a+1, b+1, lambda: w+1)
    c = tf.reshape(d, [])
    total_val = []
    # print c
    for i in range(3):
        d = d + 1
        total_val.append(d)
    max_index = tf.argmax(total_val, dimension=0)

    # print tf_max_val
    print sess.run(max_index)
    # max_val = total_val[max_index]


    total_test = tf.gather(total_val, max_index)

    print sess.run(total_test)
    # for i, val in enumerate(total_val):
    #     print i, val
    #     indexs = tf.constant(i)
    # # print sess.run(max_val)
