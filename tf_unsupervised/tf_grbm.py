import tensorflow as tf
import numpy as np
from tf_rbm import RBM
import utilities
try:
    import PIL.Image as Image
except ImportError:
    import Image


class GBRBM(RBM):
    def __init__(self, num_visual, num_hidden, num_visible_unit_type='bin', learning_rate=0.1,
                 gradient_lr=0.0001, max_iter=100, epoch=100, batch_size=32.0, regtype='l2', cdk = 1):
        RBM.__init__(self, num_visual, num_hidden, num_visible_unit_type=num_visible_unit_type, learning_rate=learning_rate, gradient_lr=gradient_lr, max_iter=max_iter, epoch=epoch, batch_size=batch_size, regtype=regtype)
        self.sigma = tf.Variable(tf.constant(1.0, shape=[1, num_visual]), dtype=tf.float32)
        # test purposes
        self.W = tf.Variable(tf.random_uniform(shape=[num_visual, num_hidden], minval=-1.0/(num_visual + num_hidden),
                                               maxval=1.0/(num_visual+num_hidden)), name='weights', dtype=tf.float32)
        self.v = tf.Variable(tf.constant(0., shape=[1, num_visual]), name='visible-bias', dtype=tf.float32)
        self.h = tf.Variable(tf.constant(0., shape=[1, num_hidden]), name='hidden-bias', dtype=tf.float32)
        self.cdk = cdk

    def free_energy_function(self, visible_input):
        temp = tf.square(visible_input - self.v)
        part1 = tf.reduce_sum(tf.div(temp, 2*tf.square(self.sigma)), reduction_indices=[1])
        hidden_state = self.propup_mean(visible_input)
        part2 = tf.reduce_sum(tf.mul(self.h, hidden_state), reduction_indices=[1])
        vb_div_sigma = tf.div(self.v, tf.square(self.sigma))
        part3 = tf.matmul(tf.matmul(vb_div_sigma, self.W), tf.transpose(self.h))
        return part1 - part2 - part3

    def propup_mean(self, visible_layer):
        step1 = tf.div(visible_layer, tf.square(self.sigma))
        step2 = tf.matmul(step1, self.W)
        step3 = tf.add(step2, self.h)
        output = tf.nn.sigmoid(step3)
        # output = tf.nn.sigmoid(tf.add(tf.matmul(tf.div(visible_layer, tf.square(self.sigma)), self.W), self.h))
        return output

    def propdown_mean(self, hidden_layer):
        reshape_W = tf.transpose(self.W)
        step1 = tf.matmul(hidden_layer, reshape_W)
        step2 = tf.add(step1, self.v)
        return step2

    def sample_hidden_given_visible(self, visible_layer):
        mean = self.propup_mean(visible_layer)
        return mean, self.sample_prob(mean)

    # smaples from gaussian
    def sample_visible_from_gaussian(self, mean):
        mean_1_dim = tf.reshape(mean, shape=[1, -1])
        sigma = []
        for i in range(mean.get_shape().as_list()[0]):
            sigma.append(self.sigma)
        sigmas = tf.concat(1, sigma)
        samples_1_dim = tf.random_normal([1], mean=mean_1_dim, stddev=sigmas, dtype=tf.float32)
        samples = tf.reshape(samples_1_dim, [mean.get_shape().as_list()[0], mean.get_shape().as_list()[1]])
        return mean, samples


    def gibbs_vhv(self):
        h0_out_prob, h0_out_sample = self.sample_hidden_given_visible(self.input_img)
        v1_out_prob  = tf.add(tf.matmul(h0_out_sample, tf.transpose(self.W)), self.v)
        return [h0_out_prob, h0_out_sample, v1_out_prob, v1_out_prob]


    def update_parameter(self):
        logsigmas = tf.log(tf.square(self.sigma))
        logsigmas_ub = np.log(np.inf)
        epsilon_sigma = 1e-8
        epsilon_logsigma = np.log(epsilon_sigma**2)

        # positive phase  update paramenter v_bias h_bias W sigma
        v_bias0 = tf.reduce_mean(self.input_img, reduction_indices=[0])
        hidden_mean = self.propup_mean(self.input_img)
        h_bias0 = tf.reduce_mean(hidden_mean, reduction_indices=[0], keep_dims=True)
        W0 = tf.matmul(tf.transpose(self.input_img), hidden_mean)/self.input_img.get_shape().as_list()[0]
        v_bias0 = tf.div(v_bias0, tf.square(self.sigma))
        W0 = tf.div(W0, tf.transpose(tf.square(self.sigma)))
        sigma0 = tf.square(self.input_img - self.v) - self.input_img*tf.matmul(hidden_mean, tf.transpose(self.W))
        sigma0 = tf.reduce_mean(sigma0, reduction_indices=[0])
        sigma0 = tf.div(sigma0, tf.square(self.sigma))

        # negative phase
        h1 = hidden_mean
        for _ in range(self.cdk):
            hidden_val = self.sample_prob(h1)
            v1_mean = tf.add(tf.matmul(hidden_val, tf.transpose(self.W)), self.v)
            h1 = self.propup_mean(v1_mean)

        v_bias1 = tf.reduce_mean(v1_mean, reduction_indices=[0], keep_dims=True)
        h_bias1 = tf.reduce_mean(h1, reduction_indices=[0], keep_dims=True)
        W1 = tf.matmul(tf.transpose(v1_mean), h1) / self.input_img.get_shape().as_list()[0]
        v_bias1 = tf.div(v_bias1, tf.square(self.sigma))
        W1 = tf.div(W1, tf.transpose(tf.square(self.sigma)))
        sigma1 = tf.square(v1_mean - self.v) - v1_mean*tf.matmul(h1, tf.transpose(self.W))
        sigma1 = tf.reduce_mean(sigma1, reduction_indices=[0])
        sigma1 = tf.div(sigma1, tf.square(self.sigma))

        vbiase_grad = v_bias0 - v_bias1
        w_grad = W0 - W1
        sigma_grad = sigma0 - sigma1
        hbias_grad = h_bias0 - h_bias1
        v_update = self.v.assign_add(self.gradient_lr*vbiase_grad)
        h_update = self.h.assign_add(self.gradient_lr*hbias_grad)
        W_update = self.W.assign_add(self.gradient_lr*w_grad)
        logsigmas = logsigmas + self.gradient_lr*sigma_grad
        logsigmas = tf.maximum(epsilon_logsigma, tf.minimum(logsigmas_ub, tf.cast(logsigmas, tf.float64)))
        sigma_update = self.sigma.assign(tf.cast(tf.sqrt(tf.exp(logsigmas)), tf.float32))
        return [v_update, h_update, W_update, sigma_update]


if __name__ == '__main__':
    IMG_SIZE = 19
    gbrbm = GBRBM(IMG_SIZE*IMG_SIZE, 256, cdk=15, epoch=1000)
    # o_train_set_x = np.load('../theano_rbm/data/origin_target_train_28.npy')
    o_train_set_x = np.load('../theano_rbm/data/face_train_dataset_19.npy')

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    np.random.shuffle(o_train_set_x)
    batches = [_ for _ in utilities.gen_batches(o_train_set_x, int(gbrbm.batch_size))]
    updates = gbrbm.update_parameter()
    for i in range(gbrbm.epoch):
        np.random.shuffle(o_train_set_x)
        batches = [_ for _ in utilities.gen_batches(o_train_set_x, int(gbrbm.batch_size))]
        for batch in batches:
            if batch.shape[0] != 32:
                continue
            v_, h_, w_, sigma_ = sess.run(updates, feed_dict={gbrbm.input_img: batch})

    # Construct image from the weight matrix
    image = Image.fromarray(
        utilities.tile_raster_images(
            X=w_.T,
            img_shape=(IMG_SIZE, IMG_SIZE),
            tile_shape=(6, 6),
            tile_spacing=(1, 1)
        )
    )
    image.save('filters_at_epoch_%i.png' % 0)
    samples = o_train_set_x[:32]
    image = Image.fromarray(
        utilities.tile_raster_images(
            X=samples,
            img_shape=(IMG_SIZE, IMG_SIZE),
            tile_shape=(6, 6),
            tile_spacing=(1, 1)
        )
    )
    image.save('original_image_%i.png' % 1)

    h0_out_prob, h0_out_sample, v1_out_prob, v1_out_sample = gbrbm.gibbs_vhv()
    # for i in range(10):
    #     if i == 0:
    #         v_sample = sess.run(v1_out_sample, feed_dict={rbm.input_img: samples})
    #     else:
    #         v_sample = sess.run(v1_out_sample, feed_dict={rbm.input_img: v_sample})

    v_sample = sess.run(v1_out_sample, feed_dict={gbrbm.input_img: samples})
    image = Image.fromarray(
        utilities.tile_raster_images(
            X=v_sample,
            img_shape=(IMG_SIZE, IMG_SIZE),
            tile_shape=(6, 6),
            tile_spacing=(1, 1)
        )
    )
    image.save('sample_image_at_epoch_%i.png' % 1)


