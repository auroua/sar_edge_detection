import tensorflow as tf
import numpy as np
from tf_rbm import RBM
import utilities
try:
    import PIL.Image as Image
except ImportError:
    import Image
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', 'data', 'data dir')


class GBRBM(RBM):
    def __init__(self, num_visual, num_hidden, num_visible_unit_type='bin', learning_rate=0.1,
                 gradient_lr=0.01, max_iter=100, epoch=100, batch_size=32.0, regtype='l2', cdk=1):
        RBM.__init__(self, num_visual, num_hidden, num_visible_unit_type=num_visible_unit_type, learning_rate=learning_rate, gradient_lr=gradient_lr, max_iter=max_iter, epoch=epoch, batch_size=batch_size, regtype=regtype)
        self.sigma = tf.Variable(tf.constant(1.0, shape=[1, num_visual]), dtype=tf.float32)
        # test purposes
        self.W = tf.Variable(tf.random_uniform(shape=[num_visual, num_hidden], minval=-1.0/(num_visual + num_hidden),
                                               maxval=1.0/(num_visual+num_hidden)), name='weights', dtype=tf.float32)
        self.v = tf.Variable(tf.constant(0., shape=[1, num_visual]), name='visible-bias', dtype=tf.float32)
        self.h = tf.Variable(tf.constant(0., shape=[1, num_hidden]), name='hidden-bias', dtype=tf.float32)
        self.cdk = cdk

        # adaptive_learning_rate
        self.exp_up = 1.01
        self.exp_down = 0.99
        self.max_iter_up = 1
        self.max_iter_down = 1
        self.lrate_anneal = 0.9
        self.lrate_lb = -1*np.inf
        self.lrate_ub = np.inf
        self.pt_n_chains = 11
        self.temperatures = np.linspace(0, 1, self.pt_n_chains)
        self.swap_interval = 1

    def free_energy_function(self, visible_input):
        temp = tf.square(visible_input - self.v)
        part1 = tf.reduce_sum(tf.div(temp, 2*tf.square(self.sigma)), reduction_indices=[1])
        hidden_state = self.propup_mean(visible_input)
        part2 = tf.reduce_sum(tf.mul(self.h, hidden_state), reduction_indices=[1])
        vb_div_sigma = tf.div(self.v, tf.square(self.sigma))
        part3 = tf.matmul(tf.matmul(vb_div_sigma, self.W), tf.transpose(self.h))
        return part1 - part2 - part3

    def energy_function(self, input, w_test, v_test, h_test, sigma_test):
        wh = tf.matmul(tf.div(input, sigma_test**2), w_test) + h_test
        step1 = tf.reduce_sum(tf.div(tf.square(input - v_test), 2*tf.square(sigma_test)), reduction_indices=[1])
        wh_plus = tf.maximum(wh, 0)
        step2 = step1 - tf.reduce_sum(tf.log(tf.exp(tf.neg(wh_plus))+tf.exp(wh - wh_plus)) + wh_plus,
                                      reduction_indices=[1])
        step3 = tf.transpose(step2)
        e = tf.reduce_mean(step3)
        e_sum = tf.reduce_sum(tf.neg(step3))
        tf.scalar_summary('energy_val', e_sum)
        return step3, e, e_sum

    def propup_mean(self, visible_layer):
        step1 = tf.div(visible_layer, tf.square(self.sigma))
        step2 = tf.matmul(step1, self.W)
        step3 = tf.add(step2, self.h)
        output = tf.nn.sigmoid(step3)
        # output = tf.nn.sigmoid(tf.add(tf.matmul(tf.div(visible_layer, tf.square(self.sigma)), self.W), self.h))
        return output

    def propup_mean_param(self, visible_layer, w, h, sigma):
        step1 = tf.div(visible_layer, tf.square(sigma))
        step2 = tf.matmul(step1, w)
        step3 = tf.add(step2, h)
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
        samples_1_dim = tf.random_normal([1], mean=mean_1_dim, stddev=sigmas, dtype=tf.float64)
        samples = tf.reshape(samples_1_dim, [mean.get_shape().as_list()[0], mean.get_shape().as_list()[1]])
        return mean, samples


    def gibbs_vhv(self):
        h0_out_prob, h0_out_sample = self.sample_hidden_given_visible(self.input_img)
        v1_out_prob  = tf.add(tf.matmul(h0_out_sample, tf.transpose(self.W)), self.v)
        return [h0_out_prob, h0_out_sample, v1_out_prob, v1_out_prob]


    def get_adaptive_lrage(self):
        cand_lrates = []
        cand_lrate = self.gradient_lr
        for i in range(self.max_iter_up+1):
            cand_lrates.append(cand_lrate)
            cand_lrate = cand_lrate*self.exp_up
        cand_lrate = self.gradient_lr*self.exp_down
        for i in range(self.max_iter_down):
            cand_lrates.append(cand_lrate)
            cand_lrate = cand_lrate*self.exp_down
        return cand_lrates

    def update_parameter(self, mu, std):
        logsigmas = tf.log(tf.square(self.sigma))
        logsigmas_ub = np.log(np.inf, dtype=np.float32)
        epsilon_sigma = 1e-8
        epsilon_logsigma = np.log(epsilon_sigma**2, dtype=np.float32)

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
        # h1 = hidden_mean
        h1_list = []
        v1_list = []
        v1_mean_list = []

        init_v = tf.Variable(tf.zeros(shape=[1, self.num_visual]))
        # initialize tp list
        for k in range(self.pt_n_chains):
            h1_list.append(hidden_mean)
            v1_list.append(init_v)
            v1_mean_list.append(v1_mean_list)

        for ck_k in range(self.cdk):
            for ti in range(self.pt_n_chains):
                t = self.temperatures[ti]
                w_t = t*self.W
                vbias_t = t*self.v + (1-t)*mu
                hbias_t = t*self.h
                sigma2s_t = tf.sqrt(t*tf.square(self.sigma)+(1-t)*std**2)

                hidden_val = self.sample_prob(tf.gather(h1_list, ti))
                v1_mean = tf.add(tf.matmul(hidden_val, tf.transpose(w_t)), vbias_t)
                v1 = v1_mean
                h1 = self.propup_mean_param(v1_mean, w_t, hbias_t, sigma2s_t)
                h1_list[ti] = h1
                v1_list[ti] = v1
                v1_mean_list[ti] = v1_mean

        for ti in range(self.pt_n_chains-1):
            t1 = self.temperatures[ti]
            t2 = self.temperatures[ti+1]

            w1_t = t1 * self.W
            vbias1_t = t1 * self.v + (1 - t1) * mu
            hbias1_t = t1 * self.h
            sigma2s1_t = tf.sqrt(t1 * tf.square(self.sigma) + (1 - t1) * std ** 2)

            w2_t = t2 * self.W
            vbias2_t = t2 * self.v + (1 - t2) * mu
            hbias2_t = t2 * self.h
            sigma2s2_t = tf.sqrt(t2 * tf.square(self.sigma) + (1 - t2) * std ** 2)

            energyv11, energy0_mean_v11, energy0_sum_v11 = self.energy_function(v1_list[ti], w1_t, vbias1_t,
                                                                                     hbias1_t, sigma2s1_t)
            energyv12, energy0_mean_v12, energy0_sum_v12 = self.energy_function(v1_list[ti], w2_t, vbias2_t,
                                                                                           hbias2_t, sigma2s2_t)
            energyv21, energy0_mean_v21, energy0_sum_v21 = self.energy_function(v1_list[ti+1], w1_t, vbias1_t,
                                                                            hbias1_t, sigma2s1_t)
            energyv22, energy0_mean_v22, energy0_sum_v22 = self.energy_function(v1_list[ti+1], w2_t, vbias2_t,
                                                                            hbias2_t, sigma2s2_t)
            costs = tf.exp(energyv11 - energyv12 + energyv22 - energyv21)
            std_one = tf.ones_like(energyv11)
            swap_prop = tf.minimum(std_one, costs)
            swap_prob = self.sample_prob(swap_prop)
            staying_particles = 1 - swap_prob

            swap_prob_reshape = tf.reshape(swap_prob, shape=[32, 1])
            staying_particles_reshape = tf.reshape(staying_particles, shape=[32, 1])
            v1t1 = v1_list[ti] * swap_prob_reshape
            v1mt1 = v1_mean_list[ti] * swap_prob_reshape
            h1t1 = h1_list[ti] * swap_prob_reshape
            v1t2 = v1_list[ti+1] * swap_prob_reshape
            v1mt2 = v1_mean_list[ti+1] * swap_prob_reshape
            h1t2 = h1_list[ti+1] * swap_prob_reshape

            v1_list[ti] = v1_list[ti] * staying_particles_reshape
            v1_mean_list[ti] = v1_mean_list[ti] * staying_particles_reshape
            h1_list[ti] = h1_list[ti] * staying_particles_reshape
            v1_list[ti+1] = v1_list[ti+1] * staying_particles_reshape
            v1_mean_list[ti+1] = v1_mean_list[ti+1] * staying_particles_reshape
            h1_list[ti+1] = h1_list[ti+1] * staying_particles_reshape

            v1_list[ti] = v1_list[ti] + v1t2
            v1_mean_list[ti] = v1_mean_list[ti] + v1mt2
            h1_list[ti] = h1_list[ti] + h1t2
            v1_list[ti + 1] = v1_list[ti + 1] + v1t1
            v1_mean_list[ti + 1] = v1_mean_list[ti + 1] + v1mt1
            h1_list[ti + 1] = h1_list[ti + 1] + h1t1

        v_bias1 = tf.reduce_mean(v1_list[self.pt_n_chains-1], reduction_indices=[0], keep_dims=True)
        h_bias1 = tf.reduce_mean(h1_list[self.pt_n_chains-1], reduction_indices=[0], keep_dims=True)
        W1 = tf.matmul(tf.transpose(v1_list[self.pt_n_chains-1]), h1_list[self.pt_n_chains-1]) / self.input_img.get_shape().as_list()[0]
        v_bias1 = tf.div(v_bias1, tf.square(self.sigma))
        W1 = tf.div(W1, tf.transpose(tf.square(self.sigma)))
        sigma1 = tf.square(v1_list[self.pt_n_chains-1] - self.v) - v1_list[self.pt_n_chains-1]*tf.matmul(h1_list[self.pt_n_chains-1], tf.transpose(self.W))
        sigma1 = tf.reduce_mean(sigma1, reduction_indices=[0])
        sigma1 = tf.div(sigma1, tf.square(self.sigma))

        vbiase_grad = v_bias0 - v_bias1
        w_grad = W0 - W1
        sigma_grad = sigma0 - sigma1
        hbias_grad = h_bias0 - h_bias1

        # # for adaptive learning rate
        # v1_mean = v1_list[self.pt_n_chains-1]
        # energy0_rcon, energy0_mean_rcon, energy0_sum_rcon = self.energy_function(v1_mean, self.W, self.v, self.h, self.sigma)
        # # energy0_origin, energy0_mean_origin, energy0_sum_origin = self.energy_function(self.input_img, self.W, self.v, self.h, self.sigma)
        # lrates = []
        # cost_total = []
        # max_cost = tf.constant([0.0])
        # # adaptive learning rate
        # learning_rate = self.get_adaptive_lrage()
        # for index, lrate in enumerate(learning_rate):
        #     v_update_test = self.v + lrate*vbiase_grad
        #     h_update_test = self.h + lrate*hbias_grad
        #     W_update_test = self.W + lrate*w_grad
        #     logsigmas = logsigmas + lrate * sigma_grad
        #     logsigmas = tf.maximum(epsilon_logsigma, tf.minimum(logsigmas_ub, tf.cast(logsigmas, tf.float32)))
        #     sigma_update_test = tf.sqrt(tf.exp(logsigmas))
        #
        #     energy, energy_mean, _ = self.energy_function(self.input_img, W_update_test, v_update_test, h_update_test, sigma_update_test)
        #     energy2, energy_mean2, _ = self.energy_function(v1_mean, W_update_test, v_update_test, h_update_test, sigma_update_test)
        #
        #     now_cost = tf.reduce_sum(tf.neg(energy) - tf.log(tf.reduce_sum(tf.exp((tf.neg(energy2) + energy0_rcon))))
        #                                + tf.log(self.batch_size))
        #     lrates.append(lrate)
        #     cost_total.append(now_cost)
        #     max_cost = tf.maximum(now_cost, max_cost)
        # max_cost_index = tf.argmax(cost_total, dimension=0)
        # self.gradient_lr = tf.gather(lrates, max_cost_index)

        v_update = self.v.assign_add(self.gradient_lr*vbiase_grad)
        h_update = self.h.assign_add(self.gradient_lr*hbias_grad)
        W_update = self.W.assign_add(self.gradient_lr*w_grad)
        logsigmas = logsigmas + self.gradient_lr*sigma_grad
        logsigmas = tf.maximum(epsilon_logsigma, tf.minimum(logsigmas_ub, logsigmas))
        sigma_update = self.sigma.assign(tf.sqrt(tf.exp(logsigmas)))
        return [v_update, h_update, W_update, sigma_update]
        # print swap_prob
        # print v1t1
        # return [costs, std_one, swap_prob, v1t1]

    # test purpose
    #def update_parameter(self, sess, batch):
    #     logsigmas = tf.log(tf.square(self.sigma))
    #     logsigmas_ub = np.log(np.inf, dtype=np.float32)
    #     epsilon_sigma = 1e-8
    #     epsilon_logsigma = np.log(epsilon_sigma**2, dtype=np.float32)
    #
    #     # positive phase  update paramenter v_bias h_bias W sigma
    #     v_bias0 = tf.reduce_mean(self.input_img, reduction_indices=[0])
    #     hidden_mean = self.propup_mean(self.input_img)
    #     h_bias0 = tf.reduce_mean(hidden_mean, reduction_indices=[0], keep_dims=True)
    #     W0 = tf.matmul(tf.transpose(self.input_img), hidden_mean)/self.input_img.get_shape().as_list()[0]
    #     v_bias0 = tf.div(v_bias0, tf.square(self.sigma))
    #     W0 = tf.div(W0, tf.transpose(tf.square(self.sigma)))
    #     sigma0 = tf.square(self.input_img - self.v) - self.input_img*tf.matmul(hidden_mean, tf.transpose(self.W))
    #     sigma0 = tf.reduce_mean(sigma0, reduction_indices=[0])
    #     sigma0 = tf.div(sigma0, tf.square(self.sigma))
    #
    #     # negative phase
    #     h1 = hidden_mean
    #     for _ in range(self.cdk):
    #         hidden_val = self.sample_prob(h1)
    #         v1_mean = tf.add(tf.matmul(hidden_val, tf.transpose(self.W)), self.v)
    #         h1 = self.propup_mean(v1_mean)
    #
    #     v_bias1 = tf.reduce_mean(v1_mean, reduction_indices=[0], keep_dims=True)
    #     h_bias1 = tf.reduce_mean(h1, reduction_indices=[0], keep_dims=True)
    #     W1 = tf.matmul(tf.transpose(v1_mean), h1) / self.input_img.get_shape().as_list()[0]
    #     v_bias1 = tf.div(v_bias1, tf.square(self.sigma))
    #     W1 = tf.div(W1, tf.transpose(tf.square(self.sigma)))
    #     sigma1 = tf.square(v1_mean - self.v) - v1_mean*tf.matmul(h1, tf.transpose(self.W))
    #     sigma1 = tf.reduce_mean(sigma1, reduction_indices=[0])
    #     sigma1 = tf.div(sigma1, tf.square(self.sigma))
    #
    #     vbiase_grad = v_bias0 - v_bias1
    #     w_grad = W0 - W1
    #     sigma_grad = sigma0 - sigma1
    #     hbias_grad = h_bias0 - h_bias1
    #
    #     energy0_rcon, energy0_mean_rcon, energy0_sum_rcon = self.energy_function(v1_mean, self.W, self.v, self.h, self.sigma)
    #     energy0_origin, energy0_mean_origin, energy0_sum_origin = self.energy_function(self.input_img, self.W, self.v, self.h, self.sigma)
    #     curr_cost = energy0_sum_origin
    #     lrates = []
    #     cost_total = []
    #     # lrates.append(self.gradient_lr)
    #     # cost_total.append(curr_cost)
    #     max_cost_index = 0
    #     max_cost = tf.constant([0.0])
    #     # adaptive learning rate
    #     learning_rate = self.get_adaptive_lrage()
    #     for index, lrate in enumerate(learning_rate):
    #         v_update_test = self.v - lrate*vbiase_grad
    #         h_update_test = self.h - lrate*hbias_grad
    #         W_update_test = self.W - lrate*w_grad
    #         logsigmas = logsigmas + lrate * sigma_grad
    #         logsigmas = tf.maximum(epsilon_logsigma, tf.minimum(logsigmas_ub, tf.cast(logsigmas, tf.float32)))
    #         sigma_update_test = tf.sqrt(tf.exp(logsigmas))
    #
    #         energy, energy_mean, _ = self.energy_function(self.input_img, W_update_test, v_update_test, h_update_test, sigma_update_test)
    #         energy2, energy_mean2, _ = self.energy_function(v1_mean, W_update_test, v_update_test, h_update_test, sigma_update_test)
    #
    #         now_cost = tf.reduce_sum(tf.neg(energy) - tf.log(tf.reduce_sum(tf.exp((tf.neg(energy2) + energy0_rcon))))
    #                                    + tf.log(self.batch_size))
    #
    #         print 'max_cost value is ',  max_cost.eval(session=sess, feed_dict={self.input_img: batch[0]})
    #         print 'now_cost value is ',  now_cost.eval(session=sess, feed_dict={self.input_img: batch[0]})
    #         print 'max_cost and now_cost compare is', compare.eval(session=sess, feed_dict={self.input_img: batch[0]})
    #         lrates.append(lrate)
    #         cost_total.append(now_cost)
    #         max_cost = tf.maximum(now_cost, max_cost)
    #     max_cost_index = tf.argmax(cost_total, dimension=0)
    #     print 'max_cost index is ',  max_cost_index.eval(session=sess, feed_dict={self.input_img: batch[0]})
    #     self.gradient_lr = lrates[max_cost_index]
    #     self.gradient_lr = tf.gather(lrates, max_cost_index)
    #     print 'gradient_lr value is ',  self.gradient_lr.eval(session=sess, feed_dict={self.input_img: batch[0]})
    #
    #     print '------------------'
    #     for learn_rates in lrates:
    #         print learn_rates
    #
    #     for cost_value in cost_total:
    #         print cost_value.eval(session=sess, feed_dict={self.input_img: batch[0]})
    #     v_update = self.v.assign_add(self.gradient_lr*vbiase_grad)
    #     h_update = self.h.assign_add(self.gradient_lr*hbias_grad)
    #     W_update = self.W.assign_add(self.gradient_lr*w_grad)
    #     print '-------------------stage 2-------------------------------'
    #     print logsigmas
    #     logsigmas = logsigmas + self.gradient_lr*sigma_grad
    #     print logsigmas
    #     logsigmas = tf.maximum(epsilon_logsigma, tf.minimum(logsigmas_ub, logsigmas))
    #     print logsigmas
    #     sigma_update = self.sigma.assign(tf.sqrt(tf.exp(logsigmas)))
    #     print sigma_update
    #     return [v_update, h_update, W_update, sigma_update]

    def no_adaptive_update_parameter(self):
        logsigmas = tf.log(tf.square(self.sigma))
        logsigmas_ub = np.log(np.inf, dtype=np.float32)
        epsilon_sigma = 1e-8
        epsilon_logsigma = np.log(epsilon_sigma**2, dtype=np.float32)

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
        logsigmas = tf.maximum(epsilon_logsigma, tf.minimum(logsigmas_ub, logsigmas))
        sigma_update = self.sigma.assign(tf.sqrt(tf.exp(logsigmas)))
        return [v_update, h_update, W_update, sigma_update]

if __name__ == '__main__':
    IMG_SIZE = 28
    gbrbm = GBRBM(IMG_SIZE*IMG_SIZE, 256, cdk=1000, epoch=3)
    o_train_set_x = np.load('../theano_rbm/data/origin_target_train_28.npy')
    # o_train_set_x = np.load('../theano_rbm/data/face_train_dataset_19.npy')
    total_mu = np.mean(o_train_set_x, axis=0, dtype=np.float32)
    total_std = np.std(o_train_set_x, axis=0, dtype=np.float32)


    sess = tf.Session()
    summary = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    sess.run(tf.initialize_all_variables())

    np.random.shuffle(o_train_set_x)
    batches = [_ for _ in utilities.gen_batches(o_train_set_x, int(gbrbm.batch_size))]
    updates = gbrbm.update_parameter(total_mu, total_std)

    # x, y, z, k = sess.run(updates, feed_dict={gbrbm.input_img:batches[0]})
    # updates = gbrbm.no_adaptive_update_parameter()


    anneal_counter = 0
    base_lrate = gbrbm.gradient_lr

    _, _, energy_sum = gbrbm.energy_function(gbrbm.input_img, gbrbm.W, gbrbm.v, gbrbm.h, gbrbm.sigma)

    # print updates
    # energy0_sum_rcon, energy0_sum_origin, energy_sum, energy_sum2, now_cost = sess.run(updates, feed_dict={gbrbm.input_img: batches[0]})
    # print energy0_sum_rcon, energy0_sum_origin, energy_sum, energy_sum2
    # print now_cost, type(now_cost)

    for i in range(gbrbm.epoch):
        np.random.shuffle(o_train_set_x)
        batches = [_ for _ in utilities.gen_batches(o_train_set_x, int(gbrbm.batch_size))]
        base_samples = o_train_set_x[:32]
        # if i >= gbrbm.epoch * gbrbm.lrate_anneal:
        #     anneal_counter += 1
        #     gbrbm.gradient_lr = base_lrate / anneal_counter
        start_time = time.time()
        for j, batch in enumerate(batches):
            if batch.shape[0] != 32:
                continue
            v_, h_, w_, sigma_ = sess.run(updates, feed_dict={gbrbm.input_img: batch})
        duration = time.time() - start_time
        energy_sum_val = sess.run(energy_sum, feed_dict={gbrbm.input_img: batches[0]})
        print 'energy val is =====', energy_sum_val
        _, _, _, v1_out_sample = gbrbm.gibbs_vhv()
        recon_val = sess.run(v1_out_sample, feed_dict={gbrbm.input_img: base_samples})
        recon_error = np.mean(np.sqrt(np.square(base_samples-recon_val)))
        print 'reconstruct error after each epoch: ', recon_error

        if i % 30 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (i, energy_sum_val, duration))
            # Update the events file.
            # summary_str = sess.run(summary, feed_dict={gbrbm.input_img:batches[0]})
            # summary_writer.add_summary(summary_str, i)
            # summary_writer.flush()

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

    print 'recon error: ',  np.mean(np.sqrt(np.square(samples-v_sample)))
    # print samples.shape
    # print v_sample.shape
    image.save('sample_image_at_epoch_%i.png' % 1)