"""Utitilies module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import misc
import tensorflow as tf

# ################### #
#   Network helpers   #
# ################### #


def sample_prob(probs, rand):
    """Get samples from a tensor of probabilities.

    :param probs: tensor of probabilities
    :param rand: tensor (of the same shape as probs) of random values
    :return: binary sample of probabilities
    """
    return tf.nn.relu(tf.sign(probs - rand))


def xavier_init(fan_in, fan_out, const=1):
    """Xavier initialization of network weights.

    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow

    :param fan_in: fan in of the network (n_features)
    :param fan_out: fan out of the network (n_components)
    :param const: multiplicative constant
    """
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)


def seq_data_iterator(raw_data, batch_size, num_steps):
    """Sequence data iterator.

    Taken from tensorflow/models/rnn/ptb/reader.py
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps: (i+1) * num_steps]
        y = data[:, i * num_steps + 1: (i+1) * num_steps + 1]
    yield (x, y)


# ################ #
#   Data helpers   #
# ################ #


def gen_batches(data, batch_size):
    """Divide input data into batches.

    :param data: input data
    :param batch_size: size of each batch
    :return: data divided into batches
    """
    data_set = []
    # data = np.array(data)
    for i in range(0, data.shape[0], batch_size):
        yield data[i:i + batch_size]
        # data_set.append(data[i:i + batch_size])
    # return data_set

def to_one_hot(dataY):
    """Convert the vector of labels dataY into one-hot encoding.

    :param dataY: vector of labels
    :return: one-hot encoded labels
    """
    nc = 1 + np.max(dataY)
    onehot = [np.zeros(nc, dtype=np.int8) for _ in dataY]
    for i, j in enumerate(dataY):
        onehot[i][j] = 1
    return onehot


def conv2bin(data):
    """Convert a matrix of probabilities into binary values.

    If the matrix has values <= 0 or >= 1, the values are
    normalized to be in [0, 1].

    :type data: numpy array
    :param data: input matrix
    :return: converted binary matrix
    """
    if data.min() < 0 or data.max() > 1:
        data = normalize(data)

    out_data = data.copy()

    for i, sample in enumerate(out_data):

        for j, val in enumerate(sample):

            if np.random.random() <= val:
                out_data[i][j] = 1
            else:
                out_data[i][j] = 0

    return out_data


def normalize(data):
    """Normalize the data to be in the [0, 1] range.

    :param data:
    :return: normalized data
    """
    out_data = data.copy()

    for i, sample in enumerate(out_data):
        out_data[i] /= sum(out_data[i])

    return out_data


def masking_noise(data, sess, v):
    """Apply masking noise to data in X.

    In other words a fraction v of elements of X
    (chosen at random) is forced to zero.
    :param data: array_like, Input data
    :param sess: TensorFlow session
    :param v: fraction of elements to distort, float
    :return: transformed data
    """
    data_noise = data.copy()
    rand = tf.random_uniform(data.shape)
    data_noise[sess.run(tf.nn.relu(tf.sign(v - rand))).astype(np.bool)] = 0

    return data_noise


def salt_and_pepper_noise(X, v):
    """Apply salt and pepper noise to data in X.

    In other words a fraction v of elements of X
    (chosen at random) is set to its maximum or minimum value according to a
    fair coin flip.
    If minimum or maximum are not given, the min (max) value in X is taken.
    :param X: array_like, Input data
    :param v: int, fraction of elements to distort
    :return: transformed data
    """
    X_noise = X.copy()
    n_features = X.shape[1]

    mn = X.min()
    mx = X.max()

    for i, sample in enumerate(X):
        mask = np.random.randint(0, n_features, v)

        for m in mask:

            if np.random.random() < 0.5:
                X_noise[i][m] = mn
            else:
                X_noise[i][m] = mx

    return X_noise

# ############# #
#   Utilities   #
# ############# #


def expand_args(**args_to_expand):
    """Expand the given lists into the length of the layers.

    This is used as a convenience so that the user does not need to specify the
    complete list of parameters for model initialization.
    IE the user can just specify one parameter and this function will expand it
    """
    layers = args_to_expand['layers']
    for key, val in args_to_expand.iteritems():
        if isinstance(val, list) and len(val) != len(layers):
            args_to_expand[key] = [val[0] for _ in layers]

    return args_to_expand


def flag_to_list(flagval, flagtype):
    """Convert a string of comma-separated tf flags to a list of values."""
    if flagtype == 'int':
        return [int(_) for _ in flagval.split(',') if _]

    elif flagtype == 'float':
        return [float(_) for _ in flagval.split(',') if _]

    elif flagtype == 'str':
        return [_ for _ in flagval.split(',') if _]

    else:
        raise Exception("incorrect type")


def str2actfunc(act_func):
    """Convert activation function name to tf function."""
    if act_func == 'sigmoid':
        return tf.nn.sigmoid

    elif act_func == 'tanh':
        return tf.nn.tanh

    elif act_func == 'relu':
        return tf.nn.relu


def random_seed_np_tf(seed):
    """Seed numpy and tensorflow random number generators.

    :param seed: seed parameter
    """
    if seed >= 0:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        return True
    else:
        return False


def gen_image(img, width, height, outfile, img_type='grey'):
    """Save an image with the given parameters."""
    assert len(img) == width * height or len(img) == width * height * 3

    if img_type == 'grey':
        misc.imsave(outfile, img.reshape(width, height))

    elif img_type == 'color':
        misc.imsave(outfile, img.reshape(3, width, height))


def get_weights_as_images(self, weights, width, height, outdir='img/',
                          n_images=30, img_type='grey'):
    """Create and save the weights of the hidden units as images.

    :param weights:
    :param width:
    :param height:
    :param outdir:
    :param n_images:
    :param img_type:
    :return: self
    """
    perm = np.random.permutation(weights.shape[1])[:n_images]

    for p in perm:
        w = np.array([i[p] for i in weights])
        image_path = outdir + 'w_{}.png'.format(p)
        gen_image(w, width, height, image_path, img_type)


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array
