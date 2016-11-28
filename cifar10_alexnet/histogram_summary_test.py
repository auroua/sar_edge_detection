import tensorflow as tf

cwd = "logs"

W1 = tf.Variable(tf.random_normal([200, 10], stddev=1.0))
W2 = tf.Variable(tf.random_normal([200, 10], stddev=0.13))

w1_hist = tf.histogram_summary("weights-stdev 1.0", W1)
w2_hist = tf.histogram_summary("weights-stdev 0.13", W2)

summary_op = tf.merge_all_summaries()

init = tf.initialize_all_variables()
sess = tf.Session()

writer = tf.train.SummaryWriter(cwd)

sess.run(init)

for i in xrange(2):
    writer.add_summary(sess.run(summary_op), i)

writer.flush()
writer.close()
sess.close()