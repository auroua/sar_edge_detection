#!/usr/bin/python
from urllib import *
import cv2
import numpy as np
from sar_svm_gmm_nn.data_preprocessing import getFiles


def img_binary(origin_dir, test_dir):
    files = getFiles(origin_dir)
    for index, file in enumerate(files):
        filename = file.split('/')[-1]
        dist_file_path = test_dir+filename
        img = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        mean = np.mean(img)
        cv2.threshold(img, mean + 25, 255, cv2.THRESH_BINARY, img)
        cv2.imwrite(dist_file_path, img)
        # th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 0)
        # cv2.imwrite(dist_file_path, th2)


if __name__ == '__main__':
    origin_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei/'
    dist_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_binary/'
    # dist_url2 = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_binary2/'
    img_binary(origin_url, dist_url)


# # image = cv2.imread('showimage.jpg')
# image = cv2.imread('/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/test_dir/HB20013.026.jpg')
# image = cv2.cvtColor(image, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
# mean = np.mean(image)
# cv2.threshold(image, mean+25, 255, cv2.THRESH_BINARY, image)
# th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)
# # th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 0)
#
# cv2.namedWindow("Image")
# cv2.imshow("Image", th2)
# cv2.waitKey(0)