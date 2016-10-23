#!/usr/bin/python
# -*- coding: UTF-8 -*-

import xml.sax
import cv2
import numpy as np


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


if __name__ == "__main__":
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # 重写 ContextHandler
    Handler = XmlHandler()
    parser.setContentHandler(Handler)

    parser.parse("xml_test/HB20001.000.xml")
    print len(Handler.target_x), len(Handler.target_y)
    print len(Handler.back_x), len(Handler.back_y)

    img = cv2.imread('xml_test/HB20001.000.jpg', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    print 'img max, min', np.max(img), np.min(img)
    # print img

    target_temp_x = []
    target_temp_y = []
    for i, x in enumerate(Handler.target_x):
        if i == len(Handler.target_x)-1:
            break
        generate_counter(Handler.target_x[i], Handler.target_y[i], Handler.target_x[i+1], Handler.target_y[i+1], target_temp_x, target_temp_y)
    Handler.target_x.extend(target_temp_x)
    Handler.target_y.extend(target_temp_y)
    print len(Handler.target_x), len(Handler.target_y)
    # for i, x in enumerate(Handler.target_x):
    #     img[Handler.target_y[i], x] = 255
    back_temp_x = []
    back_temp_y = []
    for i, x in enumerate(Handler.back_x):
        if i == len(Handler.back_x)-1:
            break
        generate_counter(Handler.back_x[i], Handler.back_y[i], Handler.back_x[i+1], Handler.back_y[i+1], back_temp_x, back_temp_y)
    Handler.back_x.extend(back_temp_x)
    Handler.back_y.extend(back_temp_y)
    print len(Handler.back_x), len(Handler.back_y)
    # for i, x in enumerate(Handler.back_x):
    #      img[Handler.back_y[i], x] = 255
    target_coords = gen_coord(Handler.target_x, Handler.target_y)
    for i in range(target_coords.shape[0]-1):
        cv2.line(img, (target_coords[i][0], target_coords[i][1]), (target_coords[i+1][0],target_coords[i+1][1]), (255))
    print img.shape

    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i, j] != 255:
    #             img[i, j] = 0

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
            elif np.abs(temp_j-j)==10:
                flags = False
            else:
                img[i, j] = 0
    print temp_i
    print temp_j1
    print temp_j2
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.imwrite('/home/aurora/Desktop/test.jpg', img)