import numpy as np
import tensorflow as tf
from theano_rbm.data_process import getFiles
import os
import cv2

if __name__ == '__main__':
    PATCH_SIZE = 25
    target_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/target_test/'
    shadow_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/shadow_test/'
    bg_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_test/'

    files = getFiles(target_test)
    filelist = []
    FileNames = os.listdir(target_test)
    if len(FileNames) > 0:
        for fn in FileNames:
            if 'HB03848' in fn:
                fullfilename = os.path.join(target_test, fn)
                filelist.append(fullfilename)
    total_data = np.zeros((len(filelist), PATCH_SIZE * PATCH_SIZE), dtype=np.float32)
    for index, path in enumerate(filelist):
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            print path
            continue
        total_data[index, :] = img.flatten()
        print index, path
    total_data /= 255
    # img_back = total_data[-1].reshape(25, 25)
    for index, fn in enumerate(filelist):
        filename = fn.split('/')[-1]
        print filename
        print 'index is: ', filename[12: 14], ': ', filename[15: 17]

        # target_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/target_test/'
        # shadow_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/shadow_test/'
        # bg_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_test/'
        #
        # files = getFiles(target_test)
        # filelist = []
        # FileNames = os.listdir(target_test)
        # if len(FileNames) > 0:
        #     for fn in FileNames:
        #         if 'HB03848' in fn:
        #             fullfilename = os.path.join(target_test, fn)
        #             filelist.append(fullfilename)
        # total_data = np.zeros((len(filelist), PATCH_SIZE * PATCH_SIZE), dtype=np.float32)
        # for index, path in enumerate(filelist):
        #     img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        #     if img is None:
        #         print path
        #         continue
        #     total_data[index, :] = img.flatten()
        #     print index, path
        # print img.dtype
        # total_data /= 255
        # pred, value = sda.get_label(total_data)
        # print value
        #
        # for index, fn in enumerate(filelist):
        #     filename = fn.split('/')[-1]
        #     img_copy[filename[12: 14], filename[15: 17]] = value[index]
        # img_test = img_test*img_copy
        # cv2.imshow('test', img_test)
        # cv2.waitKey(0)
        #
        #
        # total_data2 = np.zeros((len(filelist), PATCH_SIZE * PATCH_SIZE), dtype=np.float32)
        # for index, fn in enumerate(filelist):
        #     filename = fn.split('/')[-1]
        #     print filename
        #     print 'index is: ', filename[12: 14], ': ', filename[15: 17]
        #     img_copy[filename[12: 14], filename[15: 17]] = value[index]
        #     patch = img_test2[filename[12: 14]-12:filename[12: 14]+13, filename[15: 17]-12:filename[15: 17]+13]
        #     patch = patch.reshape(1, PATCH_SIZE*PATCH_SIZE)
        #     total_data2[count, :] = patch
        #     count += 1
        # print 'total_data size :', len(total_data)
        # pred, value = sda.get_label(total_data2)
        # print value
        # print value == 0
        #
        # for index, fn in enumerate(filelist):
        #     filename = fn.split('/')[-1]
        #     img_copy[filename[12: 14], filename[15: 17]] = value[index]
        # cv2.imshow('test', img_copy)
        # cv2.waitKey(0)

        # value_reshape = value.reshape(104, 104)
        # img_copy[12:116, 12:116] = value_reshape
        # img_copy /= 2
        # cv2.imshow('segment', img_copy)
        # cv2.waitKey(0)