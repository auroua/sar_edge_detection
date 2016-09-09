from sklearn import preprocessing
import numpy as np
from theano_rbm.data_process import getFiles, getFiles_jpg
import cv2
from sklearn import svm, neighbors
from sklearn.decomposition import PCA, KernelPCA
from sklearn import mixture


def gen_label_file(dir, dist, type='target'):
    if type == 'target':
        files = getFiles(dir)
        val = 1
    else:
        files = getFiles_jpg(dir)
        val = 0

    with open(dist, 'w') as label_file:
        for file_path in files:
            label_file.write(file_path + ',' + str(val))
            label_file.write('\n')


def resize_data():
    files = getFiles(test_dir)
    for index, file in enumerate(files):
        filename = file.split('/')[-1]
        dist_file_path = test_dir+'2'+filename
        img = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        img2 = np.resize(img, (128, 128))
        cv2.imwrite(dist_file_path, img2)


def crop_data():
    # files = getFiles(target_dir)
    files = getFiles_jpg(test_dir)
    for index, file in enumerate(files):
        filename = file.split('/')[-1]
        # dist_file_path = target_resize_dir+filename
        dist_file_path = test_dir+'2'+filename
        img = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if img.shape[0] != 128 or img.shape[1] != 128:
            if img.shape[0] != img.shape[1]:
                length = min(img.shape[0], img.shape[1])
                width = int(round((length - 128)/2))
                right = width + 128
                roi = img[width:right, width:right]
                cv2.imwrite(dist_file_path, roi)
                # cv2.imshow('sar_target', roi)
                # cv2.waitKey(0)
            elif img.shape[0] == img.shape[1]:
                length = img.shape[0]
                width = int(round((length - 128)/2))
                right = width + 128
                roi = img[width:right, width:right]
                cv2.imwrite(dist_file_path, roi)
                # cv2.imshow('sar_target', roi)
                # cv2.waitKey(0)
        elif img.shape[0] == 128:
            cv2.imwrite(dist_file_path, img)
        else:
            print file


def filter_img():
    # files = getFiles(clutter_dir)
    files = getFiles_jpg(clutter_dir)
    for index, file in enumerate(files):
        filename = file.split('/')[-1]
        img = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if img.shape[0] != img.shape[1]:
            print file, img.shape
        elif img.shape[0] != 128 or img.shape[1] != 128:
            print file, img.shape


def generate_train_data(sample_count):
    files_target = getFiles(target_resize_dir)
    files_clutter = getFiles_jpg(clutter_dir)
    # files = getFiles_jpg(clutter_dir)
    target_data = np.zeros([13411, 128*128])
    clutter_data = np.zeros([1159, 128*128])
    for index, file in enumerate(files_target):
        img = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        target_data[index, :] = img.flatten()
        # cv2.imshow('sar_target', img)
        # cv2.waitKey(0)
    # print target_data.shape

    for index, file in enumerate(files_clutter):
        img = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        clutter_data[index, :] = img.flatten()
        # cv2.imshow('sar_target', img)
        # cv2.waitKey(0)
    # print clutter_data.shape
    target_data_index = np.random.permutation(13411)
    clutter_data_index = np.random.permutation(1159)

    train_data_set = np.zeros([2*sample_count, 128*128])
    train_data_set[:sample_count, :] = target_data[target_data_index[0:sample_count]]
    train_data_set[sample_count:, :] = clutter_data[clutter_data_index[0:sample_count]]

    test_data_count = 13411 + 1159 - 2*sample_count
    test_data_set = np.zeros([test_data_count, 128*128])
    test_target_count = 13411 - sample_count
    test_data_set[:test_target_count, :] = target_data[target_data_index[sample_count:]]
    test_data_set[test_target_count:, :] = clutter_data[clutter_data_index[sample_count:]]

    train_labels = np.zeros([2*sample_count])
    train_labels[0:sample_count] = 1

    test_labels = np.zeros([test_data_count])
    test_labels[:test_target_count] = 1

    # print train_data_set.shape, train_labels.shape, test_data_set.shape, test_labels.shape
    return train_data_set, train_labels, test_data_set, test_labels


# svm-rbm: without data preprocessing, feature selection, and directly to fit a svm-rbm model
# accuracy: train: 100%, test: 96.5
def classification_svm_rbm(sample=700):
    train_data_set, train_labels, test_data_set, test_labels = generate_train_data(sample)

    C = 2.0  # SVM regularization parameter
    # rbf_svc = svm.SVC(kernel='rbf', gamma=0.4, C=C).fit(train_data_set, train_labels)
    rbf_svc = svm.SVC(kernel='poly', degree=9, C=C).fit(train_data_set, train_labels)

    Z = rbf_svc.predict(train_data_set)
    # print 'train accuracy'
    # print np.mean(Z == train_labels)

    # print 'test accuracy'
    Z = rbf_svc.predict(test_data_set)
    print 'sample count: ', sample, 'test accuracy: ', np.mean(Z == test_labels)


# svm-rbm-mean_substarct-normalization:
def classification_svm_rbm_normalization(sample=700):
    train_data_set, train_labels, test_data_set, test_labels = generate_train_data(sample)

    # preprocessing
    mean_vector = np.mean(train_data_set, axis=0)  # zero-center the data (important)
    train_data_set -= mean_vector
    train_std = np.std(train_data_set, axis=0)
    train_data_set /= train_std
    # cov = np.dot(mean_vector.T, mean_vector) / mean_vector.shape[0]  # get the data covariance matrix
    # U, S, V = np.linalg.svd(cov)
    # Xrot = np.dot(train_data_set, U)  # decorrelate the data

    C = 2.0  # SVM regularization parameter
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(train_data_set, train_labels)

    # Z = rbf_svc.predict(train_data_set)
    # print 'train accuracy'
    # print np.mean(Z == train_labels)

    test_data_set -= mean_vector
    test_data_set /= train_std
    # print 'test accuracy'
    Z = rbf_svc.predict(test_data_set)
    print 'sample count: ', sample, 'test accuracy: ', np.mean(Z == test_labels)



# kpca_svm-rbm
# sample=700, dim=10000  test accuracy=0.965148063781
def classification_kpca_svm_rbm(dim=10000, sample = 700):
    train_data_set, train_labels, test_data_set, test_labels = generate_train_data(sample)
    total_data_set = np.vstack((train_data_set, test_data_set))

    mean_vector = np.mean(train_data_set, axis=0)  # zero-center the data (important)
    train_data_set -= mean_vector
    cov = np.dot(train_data_set.T, train_data_set) / mean_vector.shape[0]  # get the data covariance matrix
    U, S, V = np.linalg.svd(cov)
    Xrot = np.dot(train_data_set, U)  # decorrelate the data
    x_kpca = np.dot(train_data_set, U[:, :dim])  # Xrot_reduced becomes [N x 100]
    # whiten the data:
    # divide by the eigenvalues (which are square roots of the singular values)
    x_kpca = x_kpca / np.sqrt(S[:dim] + 1e-5)

    # kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    # x_kpca = kpca.fit_transform(train_data_set)
    # print x_kpca.shape

    # train_data_set_kpca = x_kpca[:2*sample, :]
    # test_data_set_kpca = x_kpca[2*sample:, :]

    # print train_data_set_kpca.shape, test_data_set_kpca.shape

    C = 2.0  # SVM regularization parameter
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.4, C=C).fit(x_kpca, train_labels)

    Z = rbf_svc.predict(x_kpca)
    print 'train accuracy'
    print np.mean(Z == train_labels)

    # test_data preprocessing
    test_data_set -= mean_vector
    x_test_pca = np.dot(test_data_set, U[:, :dim])
    x_test_pca = x_test_pca / np.sqrt(S[:dim] + 1e-5)
    print 'test accuracy'
    Z = rbf_svc.predict(x_test_pca)
    print np.mean(Z == test_labels)


def classification_dp_gmm(sample=700):
    train_data_set, train_labels, test_data_set, test_labels = generate_train_data(sample)
    # Fit a Dirichlet process mixture of Gaussians using five components
    # dpgmm = mixture.DPGMM(n_components=5, covariance_type='full', n_iter=20)
    dpgmm = mixture.DPGMM(n_components=2, covariance_type='diag', n_iter=10)
    # dpgmm = mixture.VBGMM(n_components=2, covariance_type='diag', n_iter=30)
    dpgmm.fit(train_data_set)
    # print 'train accuracy'
    y_train_pred = dpgmm.predict(train_data_set)
    train_accuracy = np.mean(y_train_pred.ravel() == train_labels.ravel())
    print train_accuracy,
    print ',',
    # print 'test accuracy'
    y_test_pred = dpgmm.predict(test_data_set)
    test_accuracy = np.mean(y_test_pred.ravel() == test_labels.ravel())
    print test_accuracy

    # print 'compoent means:'
    # print dpgmm.means_


def classification_knn(sample=700, n_neighbors=5):
    train_data_set, train_labels, test_data_set, test_labels = generate_train_data(sample)
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(train_data_set, train_labels)
    Z = clf.predict(test_data_set)
    print 'knn test accuracy'
    print np.mean(Z == test_labels)


if __name__ == '__main__':
    # sar_clutter : 1159   sar_target: 13411
    # training_target_data : 700   training_test_data : 700
    target_dir = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized/'
    target_resize_dir = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei/'
    clutter_dir = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/clutter_chips_128x128_normalized/'
    label_dir = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/'

    test_dir = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/test_dir/'
    # generate label file
    # gen_label_file(clutter_dir, label_dir+'sar_label1', 'clutter')
    # data preprocessing
    # crop_data()
    # filter_img()


    # classification
    # print 'svm_rbm'
    # for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    #     classification_svm_rbm(i)
    # print 'svm_rbm_normalization'
    # for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    #      classification_svm_rbm_normalization(i)
    # classification_svm_rbm_normalization(600)
    # print 'svm_rbm_pca'
    # classification_kpca_svm_rbm()
    # classification_knn()

    # for i in [100, 200, 300, 400, 500, 600]:
    #     print 'sample count: ', i
    #     classification_knn(i)
    # classification_knn(4)

    for i in [30, 40, 50, 60, 70, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]:
        # print 'sample count ', i
        classification_dp_gmm(i)
