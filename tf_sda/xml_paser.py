#!/usr/bin/python
# -*- coding: UTF-8 -*-

import xml.sax
import cv2
import numpy as np
from theano_rbm.data_process import getFiles, getFiles_jpg


class XmlHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ""
        self.filename = ""
        self.name = ''
        self.x = ''
        self.y = ''
        self.target_x = []
        self.target_y = []
        self.back_x = []
        self.back_y = []
        self.list_x = []
        self.list_y = []

    # 元素开始事件处理
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "annotation":
            print "*****filename*****"
        if tag == 'object':
            # print '*****obj-info*****'
            self.target_x = self.list_x
            self.target_y = self.list_y
            self.list_x = []
            self.list_y = []
        if tag == 'imagesize':
            self.back_x = self.list_x
            self.back_y = self.list_y

    # 元素结束事件处理
    def endElement(self, tag):
        if self.CurrentData == "filename":
            print "FileName:", self.filename
        if self.CurrentData == 'x':
            # print 'x', self.x
            self.list_x.append(int(self.x))
        if self.CurrentData == 'y':
            # print 'y', self.y
            self.list_y.append(int(self.y))
        self.CurrentData = ""

    # 内容事件处理
    def characters(self, content):
        if self.CurrentData == "filename":
            self.filename = content
        if self.CurrentData == 'name':
            self.name = content
        if self.CurrentData == 'x':
            self.x = content
        if self.CurrentData == 'y':
            self.y = content


def gen_k(x1, y1, x2, y2):
    if x2 == x1:
        return 0
    return (y2-y1*1.0)/(x2-x1)


def generate_counter(x1, y1, x2, y2, temp_list_x, temp_list_y):
    if x1 < x2:
        t = np.arange(x1, x2, 1)
        t = t.tolist()
    elif x1 > x2:
        t = np.arange(x1, x2, -1)
        t = t.tolist()
    k = gen_k(x1, y1, x2, y2)
    if k == 0:
        if y1 < y2:
            y = np.arange(y1, y2, 1)
            y = y.tolist()
        else:
            y = np.arange(y1, y2, -1)
            y = y.tolist()
        for temp in y:
            if y == y1:
                continue
            elif y == y2:
                continue
            else:
                temp_list_x.append(int(x1))
                temp_list_y.append(int(temp))
    else:
        temp_list_x.extend(t)
        temp = []
        for x in t:
            y = k*(x-x1)+y1
            y = int(round(y))
            temp.append(y)
        temp_list_y.extend(temp)


def gen_coord(coord_x, coord_y):
    lists = []
    for i, x in enumerate(coord_x):
        lists.append([x, coord_y[i]])
    return np.array(lists)


def get_counter(url):
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # 重写 ContextHandler
    handler = XmlHandler()
    parser.setContentHandler(handler)
    parser.parse(url)
    # print len(handler.target_x), len(handler.target_y)
    # print len(handler.back_x), len(handler.back_y)
    return handler


def update_counter_target(Handler):
    target_temp_x = []
    target_temp_y = []
    for i, x in enumerate(Handler.target_x):
        if i == len(Handler.target_x)-1:
            break
        generate_counter(Handler.target_x[i], Handler.target_y[i], Handler.target_x[i+1], Handler.target_y[i+1], target_temp_x, target_temp_y)
    Handler.target_x.extend(target_temp_x)
    Handler.target_y.extend(target_temp_y)
    print len(Handler.target_x), len(Handler.target_y)
    return Handler


def update_counter_back(Handler):
    back_temp_x = []
    back_temp_y = []
    for i, x in enumerate(Handler.back_x):
        if i == len(Handler.back_x)-1:
            break
        generate_counter(Handler.back_x[i], Handler.back_y[i], Handler.back_x[i+1], Handler.back_y[i+1], back_temp_x, back_temp_y)
    Handler.back_x.extend(back_temp_x)
    Handler.back_y.extend(back_temp_y)
    print len(Handler.back_x), len(Handler.back_y)
    return Handler


def draw_counter_target(Handler, img):
    target_coords = gen_coord(Handler.target_x, Handler.target_y)
    for i in range(target_coords.shape[0]-1):
        cv2.line(img, (target_coords[i][0], target_coords[i][1]), (target_coords[i+1][0],target_coords[i+1][1]), (255))

    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i, j] != 255:
    #             img[i, j] = 0
    return img


def draw_counter_back(Handler, img):
    back_coords = gen_coord(Handler.back_x, Handler.back_y)
    for i in range(back_coords.shape[0]-1):
        cv2.line(img, (back_coords[i][0], back_coords[i][1]), (back_coords[i+1][0],back_coords[i+1][1]), (255))

    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i, j] != 255:
    #             img[i, j] = 0
    return img


def fill_counter(img):
    temp_i = []
    temp_j1 = []
    temp_j2 = []
    for i in range(img.shape[0]):
        flags = False
        count = 0
        temp_j = 0
        for j in range(img.shape[1]):
            if img[i, j] == 255 and (count == 0):
                flags = True
                count += 1
                temp_j = j
                temp_j1.append(j)
                temp_i.append(i)
            if flags and img[i, j] != 255:
                img[i, j] = 255
            elif flags and img[i, j] == 255 and np.abs(temp_j-j) >= 2:
                flags = False
                temp_j2.append(j)
            elif np.abs(temp_j-j) == 10:
                flags = False
            else:
                img[i, j] = 0
    print temp_i
    print temp_j1
    print temp_j2
    return img


def after_prosses_img(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 255:
                img[i, j] = 0


def after_prosses_all(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 127:
                img[i, j] = 0
            if img[i, j] == 127:
                # print i, j, img[i, j]
                img[i, j] = 255


def save_target(handler, images, save_url, image_name):
    target_coords = gen_coord(handler.target_x, handler.target_y)
    cv2.fillPoly(images, pts=[target_coords], color=(255))
    after_prosses_img(images)
    cv2.imwrite(save_url+image_name, images)
    return images


def save_back(handler, images, save_url, image_name):
    back_coords = gen_coord(handler.back_x, handler.back_y)
    cv2.fillPoly(images, pts=[back_coords], color=(255))
    after_prosses_img(images)
    cv2.imwrite(save_url+image_name, images)
    return images


def save_all(img_t, img_b, save_url, img_name):
    img_all = (img_t+img_b)/2
    after_prosses_all(img_all)
    cv2.imwrite(save_url+img_name, img_all)
    return img_all


def get_file_name(url):
    filename = url.split('/')[-1][:-4]
    return filename


if __name__ == "__main__":
    save_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei_counter/'
    img_url = '/home/aurora/hdd/workspace/data/MSTAR_data_liang_processed/target_chips_128x128_normalized_wei/'
    xml_url = '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/tf_sda/annoation_xml'
    annoation_xml_list = getFiles(xml_url)
    # file_name_list = [get_file_name(filename) for filename in annoation_xml_list]
    for annoation_name in annoation_xml_list:
        filename = get_file_name(annoation_name)
        img_target = cv2.imread(img_url+filename+'.jpg', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        img_back = img_target.copy()
        handler = get_counter(annoation_name)
        img_target = save_target(handler, img_target, save_url, filename+'_target.jpg')
        img_back = save_back(handler, img_back, save_url, filename+'_back.jpg')
        img_all = save_all(img_target, img_back,  save_url, filename+'_all.jpg')
        # cv2.imshow('test', img_target)
        # cv2.imshow('test', img_back)
        # cv2.imshow('test', img_all)
        # cv2.waitKey(0)