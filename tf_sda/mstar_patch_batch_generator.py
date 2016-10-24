import numpy as np
from theano_rbm.data_process import getFiles
import cv2
import pickle
import random

def process_train_data(pic_url, img_size, save_url):
    files = getFiles(pic_url)
    total_data = np.zeros((len(files), img_size*img_size), dtype=np.float32)
    for index, path in enumerate(files):
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        total_data[index, :] = img.flatten()
    total_data = total_data/255
    np.save(save_url, total_data)
    return total_data


def generate_train_label_data(pic_url, img_size, save_url):
    files = getFiles(pic_url)
    total_data = np.zeros((len(files), img_size*img_size), dtype=np.float32)
    for index, path in enumerate(files):
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            print path
            continue
        total_data[index, :] = img.flatten()
    total_data /= 255
    label_data = np.zeros((len(files), 3), dtype=np.float32)
    label_data[:, 0] = 1
    save_file = open(save_url, "wb")
    pickle.dump([total_data, label_data], save_file)
    print total_data.shape
    print label_data.shape


def generate_train_label_back_data(pic_url, img_size, save_url):
    files = getFiles(pic_url)
    total_data = np.zeros((len(files), img_size*img_size), dtype=np.float32)
    for index, path in enumerate(files):
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            print path
            continue
        total_data[index, :] = img.flatten()
    total_data /= 255
    label_data = np.zeros((len(files), 3), dtype=np.float32)
    label_data[:, 1] = 1
    save_file = open(save_url, "wb")
    pickle.dump([total_data, label_data], save_file)
    print total_data.shape
    print label_data.shape


def generate_train_label_back_ground_data(pic_url, img_size, save_url):
    files = getFiles(pic_url)
    # files = random.sample(files, 100000)
    # print len(files)
    total_data = np.zeros((len(files), img_size*img_size), dtype=np.float32)
    for index, path in enumerate(files):
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            print path
            continue
        total_data[index, :] = img.flatten()
    total_data /= 255
    label_data = np.zeros((len(files), 3), dtype=np.float32)
    label_data[:, 2] = 1
    save_file = open(save_url, "wb")
    pickle.dump([total_data, label_data], save_file)
    print total_data.shape
    print label_data.shape


if __name__ == '__main__':
    target_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/target_patch/'
    back_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/back_patch/'
    back_ground_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/back_ground_patch/'
    pre_train_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/pre_train_patch/'

    back_ground_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/back_ground_patch_test/'
    target_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/target_patch_test/'
    back_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/back_patch_test/'

    save_url = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/'

    # save pre_train file
    # pre_train_file = getFiles(pre_train_patch)
    # data = process_train_data(pre_train_patch[0:100000], 5, save_url+'pre_train_set')
    # print data
    # print data.shape

    # save target_patch file
    # generate_train_label_data(target_patch, 5, save_url+'target_set')
    # generate_train_label_back_data(back_patch, 5, save_url+'back_set')
    # generate_train_label_back_ground_data(back_ground_patch, 5, save_url+'back_ground')


    # generate_train_label_data(target_patch_test, 5, save_url+'target_set_test')
    # generate_train_label_back_data(back_patch_test, 5, save_url+'back_set_test')
    generate_train_label_back_ground_data(back_ground_patch_test, 5, save_url+'back_ground_test')




