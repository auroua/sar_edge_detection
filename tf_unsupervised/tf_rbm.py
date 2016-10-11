import tensorflow as tf
import numpy as np
import cv2
import utilities

try:
    import PIL.Image as Image
except ImportError:
    import Image

class RBM(object):
    """Restricted Boltzmann Machine implementation using TensorFlow.

    The interface of the class is sklearn-like.
    """
    def __init__(self, num_visual, num_hidden, num_visible_unit_type='bin', learning_rate=1e-3,
                 gradient_lr= 1e-2, max_iter=100, epoch=10000, batch_size=32.0, regtype='l2'):
        self.input_img = tf.placeholder(tf.float32, shape=[batch_size, num_visual])
        self.W = tf.Variable(tf.truncated_normal(shape=[num_visual, num_hidden], stddev=0.1), name='weights', dtype=tf.float32)
        self.v = tf.Variable(tf.constant(0.1, shape=[num_visual]), name='visible-bias', dtype=tf.float32)
        self.h = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name='hidden-bias', dtype=tf.float32)
        self.num_hidden = num_hidden
        self.num_visual = num_visual
        self.lr = learning_rate
        self.max_iter = max_iter
        self.epoch = epoch
        self.batch_size = batch_size
        self.visible_type = num_visible_unit_type
        self.regtype = regtype
        self.gradient_lr = gradient_lr


    def free_energy_function(self, visible_input):
        v_reshape = tf.reshape(self.v, [self.num_visual, 1])
        h_reshape = tf.reshape(self.h, [self.num_hidden, 1])
        v_input = tf.reduce_sum(tf.matmul(visible_input, v_reshape))
        hidden_output = self.propup()
        v_hidden = tf.reduce_sum(tf.matmul(hidden_output, h_reshape))
        v_w = tf.matmul(tf.matmul(visible_input, self.W), tf.transpose(hidden_output))
        ones = np.ones(v_w.get_shape()[0])
        ones_diag = tf.diag(ones)
        ones_diag = tf.to_float(ones_diag)
        v_w_val = tf.reduce_sum(tf.mul(v_w, ones_diag))
        return v_input+v_hidden+v_w_val


    def propup(self, visible_layer):
        # output matrix size = 5*300 batch_size*num_hidden
        # return 1.0/(1.0+tf.exp(tf.neg(tf.add(tf.matmul(self.input_img, self.W), self.h))))
        return tf.nn.sigmoid(tf.add(tf.matmul(visible_layer, self.W), self.h))


    def propdown(self, hidden_layer):
        visible_reshape = tf.reshape(self.v, shape=[1, self.num_visual])
        # hidden_layer: 5*500   500*784
        return tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer, tf.transpose(self.W)), visible_reshape))


    def sample_prob(self, prob):
        return tf.nn.relu(tf.sign(prob - tf.random_uniform(shape=prob.get_shape())))

    def sample_hidden_given_visible(self, visible_layer):
        output = self.propup(visible_layer)
        return output, self.sample_prob(output)


    def sample_visible_given_hidden(self, hidden_laryer):
        h_v_input = self.propdown(hidden_laryer)
        return h_v_input, self.sample_prob(h_v_input)


    def gibbs_vhv(self):
        h0_out_prob, h0_out_sample = self.sample_hidden_given_visible(self.input_img)
        v1_out_prob, v1_out_sample = self.sample_visible_given_hidden(h0_out_sample)
        return [h0_out_prob, h0_out_sample, v1_out_prob, v1_out_sample]


    def gibbs_hvh(self, h0_sample):
        v0_out_prob, v0_out_sample = self.sample_visible_given_hidden(h0_sample)
        h1_out_prob, h1_out_sample = self.sample_hidden_given_visible(v0_out_sample)
        return [v0_out_prob, v0_out_sample, h1_out_prob, h1_out_sample]


    def update_parameter(self):
        h1_out_prob, h1_out_sample, v1_out_prob, v1_out_sample = self.gibbs_vhv()
        h2_out_prob, h2_out_sample = self.sample_hidden_given_visible(v1_out_sample)
        positive = tf.matmul(tf.transpose(self.input_img), h1_out_prob)
        negative = tf.matmul(tf.transpose(v1_out_sample), h2_out_prob)
        w_update = self.W.assign_add(self.lr*(positive - negative)/self.batch_size)
        v_update = self.v.assign_add(self.lr*(tf.reduce_mean(self.input_img - v1_out_sample, reduction_indices=0)))
        h_update = self.h.assign_add(self.lr*(tf.reduce_mean(h1_out_prob - h2_out_prob, reduction_indices=0)))
        return [w_update, v_update, h_update, v1_out_sample]

    # def compute_regularization(self, vars):
    #     if self.regtype != 'none':
    #         regularizers = tf.constant(0.0)
    #         for v in vars:
    #             if self.regtype == 'l2':
    #                 regularizers = tf.add(regularizers, tf.nn.l2_loss(v))
    #             elif self.regtype == 'l1':
    #                 regularizers = tf.add(regularizers, tf.reduce_sum(tf.abs(v)))
    #         return tf.mul(self.l2reg, regularizers)
    #     else:
    #         return None

    def cost_function(self, visible_input, recon_visible):
        clip_inf = tf.clip_by_value(recon_visible, 1e-10, float('inf'))
        clip_sup = tf.clip_by_value(1-recon_visible, 1e-10, float('inf'))
        return tf.reduce_mean(visible_input*tf.log(clip_inf)+(1-visible_input)*tf.log(clip_sup))


    def train_step_node(self, cost):
        train = tf.train.GradientDescentOptimizer(self.gradient_lr).minimize(cost)
        return train


    # def _run_train_step(self, train_set):
    #     """Run a training step.
    #
    #     A training step is made by randomly shuffling the training set,
    #     divide into batches and run the variable update nodes for each batch.
    #     :param train_set: training set
    #     :return: self
    #     """


if __name__ == '__main__':
    rbm = RBM(784, 600)
    o_train_set_x = np.load('../theano_rbm/data/mnist_train_data_28.npy')
    # o_train_set_x = o_train_set_x[0:32]
    # print type(batches[0])
    # print len(batches[0])
    # print batches[0].shape, batches[0].dtype
    # input = tf.constant(train_data)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    # energy_val = rbm.free_energy_function()
    # dim = energy_val.get_shape().as_list()
    # sample_prob, sample_output = rbm.sample_hidden_given_visible()
    # print sample_output.get_shape().as_list()
    # sample_v_prob, sample_v_output = rbm.sample_visible_given_hidden(sample_output)
    # print input.get_shape().as_list()
    # h0_p, h0_sample, v1_p, v1_sample = rbm.gibbs_vhv(input)
    # v0_p, v0_sample, h1_p, h1_sample = rbm.gibbs_hvh(h0_sample)
    # params = [h0_p, h0_sample, v1_p, v1_sample, v0_p, v0_sample, h1_p, h1_sample]
    # h00, h01, v10, v11, v0p, v0s, h1p, h1s = sess.run(params)
    # print h00.shape, h01.shape, v10.shape, v11.shape, v0p.shape, v0s.shape, h1p.shape, h1s.shape
    # print sample_img.shape
    # cv2.imshow('sample_value', sample_img[0, :].reshape([28, 28]))
    # cv2.waitKey(0)
    # print sess.run(input)
    # print sess.run(w)

    updates = rbm.update_parameter()
    for i in range(rbm.epoch):
        np.random.shuffle(o_train_set_x)
        batches = [_ for _ in utilities.gen_batches(o_train_set_x, int(rbm.batch_size))]
        for batch in batches:
            w_, v_, h_, v_out = sess.run(updates, feed_dict={rbm.input_img: batch})
    # utilities.get_weights_as_images(w_, 28, 28, )
    # for i in range(32):
    #     cv2.imshow('sample_value', v_out[i].reshape([28, 28]))
    #     cv2.waitKey(0)

    # Construct image from the weight matrix
    image = Image.fromarray(
        utilities.tile_raster_images(
            X=w_.T,
            img_shape=(28, 28),
            tile_shape=(10, 10),
            tile_spacing=(1, 1)
        )
    )
    image.save('filters_at_epoch_%i.png' % 0)
    samples = o_train_set_x[:32]
    image = Image.fromarray(
        utilities.tile_raster_images(
            X=samples,
            img_shape=(28, 28),
            tile_shape=(6, 6),
            tile_spacing=(1, 1)
        )
    )
    image.save('original_image_%i.png' % 1)

    h0_out_prob, h0_out_sample, v1_out_prob, v1_out_sample = rbm.gibbs_vhv()
    # for i in range(10):
    #     if i == 0:
    #         v_sample = sess.run(v1_out_sample, feed_dict={rbm.input_img: samples})
    #     else:
    #         v_sample = sess.run(v1_out_sample, feed_dict={rbm.input_img: v_sample})

    v_sample = sess.run(v1_out_sample, feed_dict={rbm.input_img: samples})
    image = Image.fromarray(
        utilities.tile_raster_images(
            X=v_sample,
            img_shape=(28, 28),
            tile_shape=(6, 6),
            tile_spacing=(1, 1)
        )
    )
    image.save('sample_image_at_epoch_%i.png' % 1)

