import os
import gzip
import pickle
import cv2
import numpy as np
if __name__ == '__main__':
    dataset = 'data/mnist.pkl.gz'
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib

        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    a = train_set[0]
    print a.shape, a.dtype
    b = a[0].reshape((28, 28))
    print b.shape
    print a[0]
    cv2.imshow('image', b)
    cv2.waitKey(0)

    o_train_set_x = np.load('data/mnist_train_data_128.npy')
    print o_train_set_x.shape
    print o_train_set_x[0].dtype
    c = o_train_set_x[0].reshape((128, 128))
    print o_train_set_x[0]
    cv2.imshow('image', c)
    cv2.waitKey(0)