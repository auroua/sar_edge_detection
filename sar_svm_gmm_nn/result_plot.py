import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # index = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    # svm_rbm = np.array([0.926, 0.932, 0.938, 0.9448, 0.95148, 0.958, 0.965, 0.972, 0.979, 0.987])
    # svm_rbm_norm = np.array([0.926, 0.932, 0.9385, 0.9448, 0.9514, 0.04397, 0.965, 0.03107, 0.02568, 0.01559])
    # svm_poly = np.array([0.99, 0.99, 1, 1, 1, 1, 1, 1, 1, 1])
    # knn = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # # color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    # #                   '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    # #                   '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    # #                   '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    # color_sequence = ['#1f77b4', '#d62728', '#c49c94', '#e377c2']
    #
    # fig, ax = plt.subplots()
    # ax.plot(index, svm_rbm, 'o-', color=color_sequence[0], lw=2.5, label='svm_rbm')
    # ax.plot(index, svm_rbm_norm, '*-', color=color_sequence[1], lw=2.5, label='svm_rbm_norm')
    # ax.plot(index, svm_poly, 's-', color=color_sequence[2], lw=2.5, label='svm_poly')
    # ax.plot(index, knn, 'p-', color=color_sequence[3], lw=2.5, label='knn-5')
    # ax.legend(prop={'size': 10})
    # plt.show()






    index = np.array([30, 40, 50, 60, 70, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
    gmm_train_acc = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0.9925, 0.006, 0.993333, 0.997, 0.00376, 0.0022, 0.007, 0.0045, 0.995, 0.99538, 0.9957])
    gmm_test_acc = np.array([0.998, 0.992, 0.01368, 0.9948, 0.00776, 0.9917, 0.0093, 0.0074, 0.993, 0.0072, 0.988, 0.992, 0.008, 0.00599, 0.0081, 0.0080, 0.9915, 0.9914, 0.991])
    vbgmm_train_acc = np.array([0.98, 1.0, 0.98, 0.99, 0.014, 0.983, 0.005, 0.033, 0.015, 0.984, 0.0166, 0.0228, 0.9775,
                                0.0177, 0.026, 0.019, 0.02, 0.984, 0.9757])
    vbgmm_test_acc = np.array([0.970, 0.942, 0.9466, 0.94989, 0.04158, 0.9591, 0.05379, 0.0603, 0.043, 0.952, 0.04395,
                               0.04434, 0.95896, 0.04228, 0.04679, 0.0414, 0.0397, 0.9569, 0.959])
    dpgmm_train_acc = np.array([1, 0.0125,  0.99,  0.0166,  0.0142, 0.977,  0.035,  0.01,  0.025,  0.02, 0.025, 0.02,
                                0.978,  0.9788,  0.977, 0.97, 0.0225,  0.02,  0.97])
    dpgmm_test_acc = np.array([0.936,  0.0595,  0.969,  0.0498,  0.035, 0.948,  0.046,  0.0535,  0.04869,  0.04079,
                               0.045,  0.0439,  0.9598,  0.9535,  0.95534, 0.960,  0.047,  0.04129,  0.960])
    color_sequence = ['#1f77b4', '#d62728',  '#e377c2']
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(index, gmm_train_acc, 'o-', color=color_sequence[0], lw=2.5, label='gmm_train_acc')
    ax[0].plot(index, vbgmm_train_acc, '*-', color=color_sequence[1], lw=2.5, label='vbgmm_train_acc')
    ax[0].plot(index, dpgmm_train_acc, 's-', color=color_sequence[2], lw=2.5, label='dpgmm_train_acc')
    ax[0].legend(prop={'size': 10})
    ax[1].plot(index, gmm_test_acc, 'o-', color=color_sequence[0], lw=2.5, label='gmm_test_acc')
    ax[1].plot(index, vbgmm_test_acc, '*-', color=color_sequence[1], lw=2.5, label='vbgmm_test_acc')
    ax[1].plot(index, dpgmm_test_acc, 's-', color=color_sequence[2], lw=2.5, label='dpgmm_test_acc')
    ax[1].legend(prop={'size': 10})
    plt.show()

