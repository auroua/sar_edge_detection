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
    # print total_data
    label_data = np.zeros((len(files), 3), dtype=np.float32)
    label_data[:, 2] = 1
    save_file = open(save_url, "wb")
    pickle.dump([total_data, label_data], save_file)
    print total_data.shape
    print label_data.shape


def get_train_dataset(pre_train_url, target_url, back_url, bg_url):
    file = open(pre_train_url)
    pre_train_data, pre_train_2 = pickle.load(file)
    # print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2'
    # print pre_train_data.shape, pre_train_2.shape
    file = open(target_url)
    target_data, target_label = pickle.load(file)
    # print target_data.shape, target_label.shape
    file = open(back_url)
    back_data, back_label = pickle.load(file)
    # print back_data.shape, back_label.shape
    file = open(bg_url)
    bg_data, bg_label = pickle.load(file)
    # print bg_data.shape, bg_label.shape
    train_set = np.vstack((pre_train_data, target_data, back_data, bg_data))
    fine_tune_set = np.vstack((target_data, back_data, bg_data))
    fine_tune_label = np.vstack((target_label, back_label, bg_label))
    # print train_set.shape, fine_tune_set.shape, fine_tune_label.shape
    return train_set, fine_tune_set, fine_tune_label


def get_test_dataset(target_test_url, back_test_url, bg_test_url):
    file = open(target_test_url)
    target_test_data, target_test_label = pickle.load(file)
    # print target_test_data.shape, target_test_label.shape
    file = open(back_test_url)
    back_test_data, back_test_label = pickle.load(file)
    # print back_test_data.shape, back_test_label.shape
    file = open(bg_test_url)
    bg_test_data, bg_test_label = pickle.load(file)
    # print bg_test_data.shape, bg_test_label.shape
    test_set = np.vstack((target_test_data, back_test_data, bg_test_data))
    test_label = np.vstack((target_test_label, back_test_label, bg_test_label))
    # print test_set.shape, test_label.shape
    return test_set, test_label


def get_test_target_dataset(target_test_url, back_test_url, bg_test_url):
    file = open(target_test_url)
    target_test_data, target_test_label = pickle.load(file)
    # print test_set.shape, test_label.shape
    return target_test_data, target_test_label


def get_test_back_dataset(target_test_url, back_test_url, bg_test_url):
    file = open(back_test_url)
    back_test_data, back_test_label = pickle.load(file)
    return back_test_data, back_test_label


def get_test_bg_dataset(target_test_url, back_test_url, bg_test_url):
    file = open(bg_test_url)
    bg_test_data, bg_test_label = pickle.load(file)
    # print test_set.shape, test_label.shape
    return bg_test_data, bg_test_label


if __name__ == '__main__':
    # target_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/target_patch/'
    # back_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/back_patch/'
    # back_ground_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/back_ground_patch/'
    # back_ground_patch_dialit = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/dialit_background_patch/'
    # pre_train_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/pre_train_patch/'
    #
    # back_ground_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/back_ground_patch_test/'
    # target_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/target_patch_test/'
    # back_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/back_patch_test/'
    #
    # save_url = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/'
    # dialit_background_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/dialit_background_patch_test/'

    # save pre_train file
    # pre_train_file = getFiles(pre_train_patch)
    # data = process_train_data(pre_train_patch[0:100000], 5, save_url+'pre_train_set')
    # print data
    # print data.shape

    # save target_patch file
    # generate_train_label_data(target_patch, 5, save_url+'target_set')
    # generate_train_label_back_data(back_patch, 5, save_url+'back_set')



    # generate_train_label_data(target_patch_test, 5, save_url+'target_set_test')
    # generate_train_label_back_data(back_patch_test, 5, save_url+'back_set_test')
    # generate_train_label_back_ground_data(back_ground_patch_test, 5, save_url+'back_ground_test')


    # pre_train_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/pre_train_set.npy'
    # target_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/target_set'
    # back_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/back_set'
    # bg_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/back_ground'
    #
    # target_test_set_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/target_set_test'
    # back_test_set_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/back_set_test'
    # bg_test_set_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/back_ground_test'
    #
    # get_train_dataset(pre_train_path, target_path, back_path, bg_path)
    # get_test_dataset(target_test_set_path, back_test_set_path, bg_test_set_path)


    # target_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/target_patch/'
    # back_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/back_patch/'
    # back_ground_patch_dialit = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/dialit_background_patch/'
    # pre_train_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/pre_train_patch/'
    #
    # back_ground_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/dialit_background_patch_test/'
    # target_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/target_patch_test/'
    # back_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/back_patch_test/'
    #
    # save_url = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/25_25/'

    # generate_train_label_data(target_patch_test, 25, save_url + 'target_test')
    # generate_train_label_back_data(back_patch, 25, save_url + 'back_patch')
    # generate_train_label_back_data(back_patch_test, 25, save_url + 'back_test')
    # generate_train_label_back_ground_data(back_ground_patch_dialit, 25, save_url+'bg_patch')
    # generate_train_label_back_ground_data(back_ground_patch_test, 25, save_url+'bg_test')


    PATCH_SIZE = 25
    counter_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/'
    img_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei/'
    target_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/target_patch_3/'
    shadow_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/shadow_patch_3/'
    bg_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_patch/'
    bg_patch_3 = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_patch_3/'
    bg_patch_5 = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_patch_5/'
    pre_train_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/pre_train_3/'
    test_counter_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/test_counter_3/'
    target_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/target_test_3/'
    shadow_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/shadow_test_3/'
    bg_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_test/'
    bg_test_3 = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_test_3/'
    bg_test_5 = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_test_5/'
    save_url = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/25_25_new/'

    bg_without_diliate = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_patch_without_diliate/'
    bg_test_without_diliate = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_test_without_diliate/'
    # generate_train_label_data(target_test, 25, save_url + 'target_test')
    # generate_train_label_back_data(shadow_test, 25, save_url + 'shadow_test')
    generate_train_label_back_ground_data(bg_test_without_diliate, 25, save_url+'bg_test_without_diliate')
    # generate_train_label_back_ground_data(pre_train_patch, 25, save_url+'pre_train')
