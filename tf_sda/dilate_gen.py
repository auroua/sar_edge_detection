import numpy as np
import cv2
from theano_rbm.data_process import getFiles_flags, getFiles_flags_back, getFiles_flags_all, getFiles_jpg


def get_file_name_all(url):
    filename = url.split('/')[-1][:-8]
    return filename


if __name__ == '__main__':
    counter_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/'
    test_counter_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/test_counter/'
    # counter_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/HB03344.003_10_10diliate.jpg'
    # counter_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/patch_size_25/dialit_background_patch/HB19986.018_70_88@4.jpg'
    # img = cv2.imread(counter_url)
    # b,g,r = cv2.split(img)
    # dilated = cv2.dilate(r, np.ones((10, 10)))
    # print b.shape
    # cv2.imshow('test', r)
    # cv2.waitKey(0)

    target_counter_list = getFiles_flags_all(test_counter_url, 'all')
    for target_counter in target_counter_list:
        img_counter = cv2.imread(target_counter, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        file_name = get_file_name_all(target_counter)
        # print file_name
        dilated = cv2.dilate(img_counter, np.ones((10, 10)))
        cv2.imwrite(test_counter_url + file_name + '_' + str(10) + '_' + str(10) + 'diliate' + '.jpg', dilated)