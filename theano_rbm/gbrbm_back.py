
from __future__ import print_function

import timeit
from rbm import RBM
try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

from util import tile_raster_images
from logistic_sgd import load_data
from rbm2 import RBM


# --------------------------------------------------------------------------
class GBRBM(RBM):
    # --------------------------------------------------------------------------
    # initialize class
    # def __init__(self, input, n_in=784, n_hidden=500, \
    # W=None, hbias=None, vbias=None, numpy_rng=None, transpose=False, activation=T.nnet.sigmoid,
    # theano_rng=None, name='grbm', W_r=None, dropout=0, dropconnect=0):
    def __init__(self, input, n_visible=784, n_hidden=500, W=None, hbias=None, vbias=None, numpy_rng=None,
                 theano_rng=None):
        # initialize parent class (RBM)
        # RBM.__init__(self, input=input, n_visible=n_in, n_hidden=n_hidden, activation=activation,
        # W=W, hbias=hbias, vbias=vbias, transpose=transpose, numpy_rng=numpy_rng,
        # theano_rng=theano_rng, name=name, dropout=dropout, dropconnect=dropconnect)
        RBM.__init__(self, input=input, n_visible=n_visible, n_hidden=n_hidden, W=W, hbias=hbias, vbias=vbias,
                     numpy_rng=numpy_rng, theano_rng=theano_rng)

    # --------------------------------------------------------------------------
    def type(self):
        return 'gauss-bernoulli'

    # --------------------------------------------------------------------------
    # overwrite free energy function (here only vbias term is different)
    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = 0.5 * T.dot((v_sample - self.vbias), (v_sample - self.vbias).T)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    # --------------------------------------------------------------------------
    # overwrite sampling function (here you sample from normal distribution)
    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)

        '''
            Since the input data is normalized to unit variance and zero mean, we do not have to sample
            from a normal distribution and pass the pre_sigmoid instead. If this is not the case, we have to sample the
            distribution.
        '''
        # in fact, you don't need to sample from normal distribution here and just use pre_sigmoid activation instead
        # v1_sample = self.theano_rng.normal(size=v1_mean.shape, avg=v1_mean, std=1.0, dtype=theano.config.floatX) + pre_sigmoid_v1
        v1_sample = pre_sigmoid_v1
        return [pre_sigmoid_v1, v1_mean, v1_sample]




def test_rbm(learning_rate=0.1, training_epochs=15,
             dataset='data/mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_folder='data/rbm_plots',
             n_hidden=10000):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    """
    datasets = load_data(dataset)

    # train_set_x, train_set_y = datasets[0]
    # test_set_x, test_set_y = datasets[2]

    img_size = 28
    img_size_1 = img_size+1

    o_train_set_x = numpy.load('data/origin_target_train_28.npy')
    # o_train_set_x = numpy.load('f_total_data_original.npy')
    o_test_set_x = o_train_set_x[0:3000, :]
    train_set_x = theano.shared(numpy.asarray(o_train_set_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    test_set_x = theano.shared(numpy.asarray(o_test_set_x,
                                           dtype=theano.config.floatX),
                             borrow=True)


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # construct the GBRBM class
    rbm = GBRBM(input=x, n_visible=img_size * img_size,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=15)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in range(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(img_size, img_size),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))
    # end-snippet-5 start-snippet-6
    #################################
    #     Sampling from the RBM     #
    #################################
    # find out the number of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )
    # end-snippet-6 start-snippet-7
    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every,
        name="gibbs_vhv"
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = numpy.zeros(
        (img_size_1 * (n_samples+1) + 1, img_size_1 * n_chains - 1),
        dtype='uint8'
    )
    image_data[0:img_size, :] = tile_raster_images(
        X=test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
        img_shape=(img_size, img_size),
        tile_shape=(1, n_chains),
        tile_spacing=(1, 1)
    )

    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print(' ... plotting sample %d' % idx)
        idx += 1
        image_data[img_size_1 * idx:img_size_1 * idx + img_size, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(img_size, img_size),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save('samples.png')
    # end-snippet-7
    os.chdir('../')


if __name__ == '__main__':
    test_rbm()