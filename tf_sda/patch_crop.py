import numpy as np
from theano_rbm.data_process import getFiles_flags, getFiles_flags_back, getFiles_flags_all, getFiles_jpg
import cv2
import os
import random

def get_file_name(url):
    filename = url.split('/')[-1][:-11]
    return filename


def get_file_name_image(url):
    filename = url.split('/')[-1][:-4]
    return filename


def get_file_name_back(url):
    filename = url.split('/')[-1][:-9]
    return filename


def get_file_name_all(url):
    filename = url.split('/')[-1][:-8]
    return filename


def get_file_name_diliate(url):
    filename = url.split('/')[-1][:-17]
    return filename

def get_file_name_diliate_3(url):
    filename = url.split('/')[-1][:-15]
    return filename

def get_file_name_diliate_5(url):
    filename = url.split('/')[-1][:-15]
    return filename


def generate_patch(img_url, counter_url, patch_url, patch_size, counter_category, counter_exclued, flags):
    """generate img patches from original image
    :param img_url: A string, the path to image.
    :param counter_url: A string, the path to the counter file.
    :param patch_url: A String, the the patch to save patches.
    :param patch_size: An int, the patch size
    :param counter_category: A String, the patch category, ['target', 'back', 'all']
    :param counter_exclued: A string, the file not contained in the counter files. eg: 'target_patch', this is a folder is not a file
    :param flags: A String, the patch category. eg: @1 target   @2 back    @3  background
    :return: null.
    """
    target_counter_list = getFiles_flags(counter_url, counter_category, counter_exclued)
    patches = np.zeros([patch_size, patch_size])
    for target_counter in target_counter_list:
        file_name = get_file_name(target_counter)
        img = cv2.imread(img_url+file_name+'.jpg', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        img_counter_target = cv2.imread(target_counter, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img_counter_target[i, j] == 255:
                    starty = i - (patch_size // 2)
                    startx = j - (patch_size // 2)
                    patches = img[starty:starty + patch_size, startx:startx + patch_size]
                    cv2.imwrite(patch_url+file_name+'_'+str(i)+'_'+str(j)+flags+'.jpg', patches)
                    print str(i)+'_'+str(j)


def generate_patch_back(img_url, counter_url, patch_url, patch_size, counter_category, counter_exclued, flags):
    """generate img patches from original image
    :param img_url: A string, the path to image.
    :param counter_url: A string, the path to the counter file.
    :param patch_url: A String, the the patch to save patches.
    :param patch_size: An int, the patch size
    :param counter_category: A String, the patch category, ['target', 'back', 'all']
    :param counter_exclued: A string, the file not contained in the counter files. eg: 'target_patch', this is a folder is not a file
    :param flags: A String, the patch category. eg: @1 target   @2 back    @3  background
    :return: null.
    """
    target_counter_list = getFiles_flags_back(counter_url, counter_category, counter_exclued)
    patches = np.zeros([patch_size, patch_size])
    for target_counter in target_counter_list:
        file_name = get_file_name_back(target_counter)
        # print file_name
        img = cv2.imread(img_url+file_name+'.jpg', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        img_counter_target = cv2.imread(target_counter, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img_counter_target[i, j] == 255:
                    starty = i - (patch_size // 2)
                    startx = j - (patch_size // 2)
                    patches = img[starty:starty + patch_size, startx:startx + patch_size]
                    cv2.imwrite(patch_url+file_name+'_'+str(i)+'_'+str(j)+flags+'.jpg', patches)


def generate_patch_all(img_url, counter_url, patch_url, patch_size, counter_category, flags):
    """generate img patches from original image
    :param img_url: A string, the path to image.
    :param counter_url: A string, the path to the counter file.
    :param patch_url: A String, the the patch to save patches.
    :param patch_size: An int, the patch size
    :param counter_category: A String, the patch category, ['target', 'back', 'all']
    :param counter_exclued: A string, the file not contained in the counter files. eg: 'target_patch', this is a folder is not a file
    :param flags: A String, the patch category. eg: @1 target   @2 back    @3  background
    :return: null.
    """
    CROP_PRE_IMAGE = 100
    target_counter_list = getFiles_flags_all(counter_url, counter_category)
    patches = np.zeros([patch_size, patch_size])
    for target_counter in target_counter_list:
        file_name = get_file_name_all(target_counter)
        img = cv2.imread(img_url+file_name+'.jpg', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        img_counter_target = cv2.imread(target_counter, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        img_counter_target = 255 - img_counter_target
        counter = 0
        flags2 = True
        for i in range(img.shape[0]):
            if not flags2:
                break
            for j in range(img.shape[1]):
                if img_counter_target[i, j] == 255 and i >= patch_size and j >= patch_size and i <= img.shape[0]-patch_size and j <= img.shape[1]-patch_size:
                    starty = i - (patch_size // 2)
                    startx = j - (patch_size // 2)
                    patches = img[starty:starty + patch_size, startx:startx + patch_size]
                    # cv2.imwrite(patch_url+file_name+'_'+str(i)+'_'+str(j)+flags+'.jpg', patches)
                    cv2.imwrite(patch_url + file_name + '_' + str(i) + '_' + str(j) + flags + '.jpg', patches)
                    counter+=1
                    print counter
                    if counter==100000:
                        flags2=False


def generate_patch_dialit_back(img_url, counter_url, patch_url, patch_size, counter_category, flags):
    """generate img patches from original image
    :param img_url: A string, the path to image.
    :param counter_url: A string, the path to the counter file.
    :param patch_url: A String, the the patch to save patches.
    :param patch_size: An int, the patch size
    :param counter_category: A String, the patch category, ['target', 'back', 'all']
    :param counter_exclued: A string, the file not contained in the counter files. eg: 'target_patch', this is a folder is not a file
    :param flags: A String, the patch category. eg: @1 target   @2 back    @3  background
    :return: null.
    """
    target_counter_list = getFiles_flags_all(counter_url, counter_category)
    print len(target_counter_list)
    # print target_counter_list
    patches = np.zeros([patch_size, patch_size])
    patch_crop_size = 500
    for target_counter in target_counter_list:
        # file_name = get_file_name_diliate_5(target_counter)
        # file_name = get_file_name_diliate_3(target_counter)
        file_name = get_file_name_all(target_counter)
        print file_name
        img = cv2.imread(img_url+file_name+'.jpg', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        img_counter_target = cv2.imread(target_counter, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        # b, g, r = cv2.split(img_counter_target)
        # img_counter_target = b
        img_counter_target = 255 - img_counter_target
        temp_size = patch_crop_size
        for i in range(100000):
            a = int(random.uniform(patch_size, img.shape[0] - patch_size))
            b = int(random.uniform(patch_size, img.shape[1] - patch_size))
            if img_counter_target[a, b] == 255 and a >= patch_size and b >= patch_size and a <= img.shape[0] - patch_size and b <= img.shape[1] - patch_size:
                temp_size -= 1
                starty = a - (patch_size // 2)
                startx = b - (patch_size // 2)
                patches = img[starty:starty + patch_size, startx:startx + patch_size]
                cv2.imwrite(patch_url + file_name + '_' + str(a) + '_' + str(b) + flags + '.jpg', patches)
                if temp_size == 0:
                    break



def generate_pre_train_patch(img_url, patch_url, patch_size, CROP_PRE_IMAGE):
    """generate img patches from original image
    :param img_url: A string, the path to image.
    :param patch_url: A String, the the patch to save patches.
    :param patch_size: An int, the patch size
    :return: null.
    """
    # CROP_PRE_IMAGE = 100
    target_counter_list = getFiles_jpg(img_url)
    # print target_counter_list
    patches = np.zeros([patch_size, patch_size])
    for target_counter in target_counter_list:
        img = cv2.imread(target_counter, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        file_name = get_file_name_image(target_counter)
        # print file_name
        for i in range(CROP_PRE_IMAGE):
            a = int(random.uniform(patch_size, img.shape[0]-patch_size))
            b = int(random.uniform(patch_size, img.shape[1]-patch_size))
            starty = a - (patch_size // 2)
            startx = b - (patch_size // 2)
            patches = img[starty:starty + patch_size, startx:startx + patch_size]
            cv2.imwrite(patch_url + file_name + '_' + str(a) + '_' + str(b) + '.jpg', patches)



if __name__ == '__main__':
    # PATCH_SIZE = 25
    # counter_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/'
    # img_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei/'
    # target_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/target_patch/'
    # target_patch_5 = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/target_patch/'
    # back_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/back_patch/'
    # back_ground_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/back_ground_patch/'
    # pre_train_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/pre_train_patch/'
    # patch_dialit = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/dialit_background_patch/'
    # patch_dialit_5 = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/dialit_background_patch/'
    #
    #
    # test_counter_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/test_counter/'
    # test_counter_url_5 = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_5/test_counter/'
    # back_ground_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/back_ground_patch_test/'
    # target_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/target_patch_test/'
    # back_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/back_patch_test/'
    # dialit_background_patch_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/dialit_background_patch_test/'
    # generate_patch(img_url, test_counter_url, target_patch_test, PATCH_SIZE, 'target', 'target_patch', '@1')
    # generate_patch_back(img_url, test_counter_url, back_patch_test, PATCH_SIZE, 'back', ['back_patch', 'back_ground_patch'], '@2')
    # generate_patch_all(img_url, counter_url, back_ground_patch, PATCH_SIZE, 'all', '@3')
    # generate_pre_train_patch(img_url, pre_train_patch, PATCH_SIZE)
    # generate_patch_dialit_back(img_url, test_counter_url, dialit_background_patch_test, PATCH_SIZE, '10_10diliate', '@4')
    # FileNames = os.listdir(dialit_background_patch_test)
    # print FileNames
    # print len(FileNames)

    PATCH_SIZE = 25
    counter_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/'
    img_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei/'
    target_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/target_patch_3/'
    shadow_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/shadow_patch_3/'
    bg_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_patch_3/'
    bg_patch_5 = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_patch_5/'
    pre_train_patch = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/pre_train_3/'
    test_counter_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/test_counter/'
    target_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/target_test_3/'
    shadow_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/shadow_test_3/'
    bg_test = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_test_3/'
    bg_test_5 = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_test_5/'
    bg_patch_without_diliate = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_patch_without_diliate/'
    bg_test_without_diliate = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/bg_test_without_diliate/'

    big_image_url = '/home/aurora/hdd/workspace/data/sar_real_data/'
    pre_train_big_image = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25_new/pre_train_big_image/'


    # generate_patch(img_url, counter_url, target_patch, PATCH_SIZE, 'target', 'target_patch', '@1')
    # generate_patch_back(img_url, test_counter_url, shadow_test, PATCH_SIZE, 'back', ['back_patch', 'back_ground_patch'], '@2')
    # generate_patch_dialit_back(img_url, test_counter_url, bg_test_without_diliate, PATCH_SIZE, 'all', '@4')
    # generate_pre_train_patch(img_url, bg_patch_without_diliate, PATCH_SIZE)
    generate_pre_train_patch(big_image_url, pre_train_big_image, PATCH_SIZE, 50000)
    FileNames = os.listdir(pre_train_big_image)
    print len(FileNames)

