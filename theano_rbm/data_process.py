# encoding:UTF-8
import cv2
import numpy as np
from PIL import Image
import cPickle
import os

def getFiles(path):
    '''获取目录下的文件的绝对路径,带文件名'''
    filelist = []
    FileNames = os.listdir(path)
    if len(FileNames) > 0:
       for fn in FileNames:
            fullfilename = os.path.join(path, fn)
            filelist.append(fullfilename)

    return filelist


def getFiles_jpg(path):
    '''filter files that daesn't end with jpg'''
    filelist = []
    FileNames = os.listdir(path)
    if len(FileNames) > 0:
       for fn in FileNames:
            if not fn.endswith('jpg'):
                continue
            fullfilename = os.path.join(path, fn)
            filelist.append(fullfilename)

    return filelist


def  process_train_data():
    # pic_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_28*28/'
    pic_url = '/home/aurora/hdd/workspace/data/CBCL/MIT-CBCL-Face-Database/train/face/'
    img_size = 19
    files = getFiles(pic_url)
    total_data = np.zeros((len(files), img_size*img_size), dtype=np.float32)
    for index, path in enumerate(files):
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        total_data[index, :] = img.flatten()
    # f_total_data = total_data.astype(np.float32)
    # # mean substraction
    # f_total_data -= np.mean(f_total_data, axis=0)
    # normalization
    # f_total_data /= np.std(total_data, axis=0)

    total_data = total_data/255
    # for index, data in enumerate(f_total_data):
    #    img2 = f_total_data[index, :]
    #    img2 = np.resize(img2, (64, 64))
    #    cv2.imshow('after mean', img2)
    #    cv2.waitKey(0)
    np.save('face_train_dataset_19', total_data)

def mnist_resize():
    mnist_origin = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei/'
    mnist_target = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_28*28/'
    files = getFiles(mnist_origin)
    for index, file in enumerate(files):
        filename = file.split('/')[-1]
        dist_file_path = mnist_target+filename
        img = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        img2 = cv2.resize(img, (28, 28))
        # cv2.imshow('mnist', img2)
        # cv2.waitKey(0)
        cv2.imwrite(dist_file_path, img2)


def  process_train_label_data():
    pic_url = '/home/aurora/hdd/workspace/data/sar_data_test/rbm_data_set/'
    label_url = '/home/aurora/hdd/workspace/data/sar_data_test/rbm_data_set/label'

    total_data = np.zeros((2747, 64 * 64), dtype=np.uint8)
    label_data = np.zeros((2747, 1), dtype=np.uint8)
    f = open(label_url, 'r')
    for index, line in enumerate(f):
        file_name, file_label = line.split('  ')
        img = cv2.imread(pic_url+file_name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        total_data[index, :] = img.flatten()
        label_data[index, :] = file_label[:-1]
        print file_name, file_label[:-1]
    f_total_data = total_data.astype(np.float32)
    # mean substraction
    f_total_data -= np.mean(f_total_data, axis=0)
    # normalization
    f_total_data /= np.std(total_data, axis=0)

    np.save('f_total_label_data_data', f_total_data)
    np.save('f_total_label_data_label', label_data)

if __name__ == '__main__':
    # process_train_label_data()
    # # load data
    process_train_data()
    # data = np.load('f_total_data.npy')
    # print data.shape, data.dtype
    # mnist_resize()
