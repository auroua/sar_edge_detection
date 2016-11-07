import glob
from itertools import groupby
from collections import defaultdict
import cv2
import numpy as np

def calcAndDrawHist(image, color):
    bin_size = 10
    hist = cv2.calcHist([image], [0], None, [bin_size], [0.0, 255.0])
    print hist, type(hist)
    print np.array(hist)/np.sum(hist)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([bin_size, bin_size, 3], np.uint8)
    hpt = int(0.9 * bin_size)
    for h in range(bin_size):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)
    return histImg


if __name__ == '__main__':
    image_filenames = glob.glob('/home/aurora/hdd/workspace/data/imagenet_dog/Images/n02*/*.jpg')
    training_dataset = defaultdict(list)
    testing_dataset = defaultdict(list)

    # Split up the filename into its breed and corresponding filename. The breed is found by taking the dir
    image_filename_with_breed = map(lambda filename: (filename.split("/")[-2], filename), image_filenames)
    # print image_filename_with_breed

    for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
        for i, breed_image in enumerate(breed_images):
            if i % 5 == 0:
                testing_dataset[dog_breed].append(breed_image[1])
            else:
                training_dataset[dog_breed].append(breed_image[1])

        breed_training_count = len(training_dataset[dog_breed])
        breed_testing_count = len(testing_dataset[dog_breed])
        # print breed_testing_count, breed_training_count

    img = cv2.imread(image_filenames[0])
    b, g, r = cv2.split(img)
    print b
    histImgB = calcAndDrawHist(b, [255, 0, 0])
    cv2.imshow('dog_img', histImgB)
    cv2.waitKey(0)
    # cv2.calcHist([img], [0], None, [256], [0.0, 255.0])


    # pre_train_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/pre_train_set.npy'
    # target_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/target_set'
    # back_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/back_set'
    # bg_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/back_ground'
    #
    # target_test_set_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/target_set_test'
    # back_test_set_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/back_set_test'
    # bg_test_set_path = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/patch_files/back_ground_test'
    #
    # pre_train_data, fine_tune_data, fine_tune_label = get_train_dataset(pre_train_path, target_path, back_path, bg_path)
    # test_data, test_label = get_test_dataset(target_test_set_path, back_test_set_path, bg_test_set_path)
    # test_target_data, test_target_label = get_test_target_dataset(target_test_set_path, back_test_set_path, bg_test_set_path)
    # test_back_data, test_back_label = get_test_back_dataset(target_test_set_path, back_test_set_path, bg_test_set_path)
    # test_bg_data, test_bg_label = get_test_bg_dataset(target_test_set_path, back_test_set_path, bg_test_set_path)
    #
    # print pre_train_data.shape
    # print fine_tune_data.shape
    # print test_data.shape
