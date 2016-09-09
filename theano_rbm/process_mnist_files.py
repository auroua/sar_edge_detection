# import numpy as np
# import struct
# import matplotlib.pyplot as plt
# import cv2
#
#
# if __name__=='__main__':
#     filename = "/home/aurora/hdd/workspace/data/mnist/train-images.idx3-ubyte"
#     filename2 = "/home/aurora/hdd/workspace/data/mnist/t10k-images.idx3-ubyte"
#     binfile = open(filename2, 'rb')
#     buf = binfile.read()
#
#     index = 0
#     magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
#     print magic, numImages, numRows, numColumns
#     index += struct.calcsize('>IIII')
#
#     im = struct.unpack_from('>784B', buf, index)
#     index += struct.calcsize('>784B')
#
#     im = np.array(im)
#     im = im.reshape(28, 28)
#     print im.dtype
#
#     # cv2.imshow('win name', im)
#     # cv2.waitKey(0)
#
#     fig = plt.figure()
#     plotwindow = fig.add_subplot(111)
#     plt.imshow(im, cmap='gray')
#     plt.imsave()
#     plt.show()


# !/usr/bin/env python
# -*- coding: utf-8 -*-


from PIL import Image
import struct


def read_image(filename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')
    for i in xrange(images):
        # for i in xrange(2000):
        image = Image.new('L', (columns, rows))
        for x in xrange(rows):
            for y in xrange(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')
        print 'save ' + str(i) + 'image'
        image.save('mnist/' + str(i) + '.png')


def read_label(filename, saveFilename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, labels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    labelArr = [0] * labels
    # labelArr = [0] * 2000

    for x in xrange(labels):
        # for x in xrange(2000):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')
    save = open(saveFilename, 'w')
    save.write(','.join(map(lambda x: str(x), labelArr)))
    save.write('\n')
    save.close()
    print 'save labels success'

if __name__ == '__main__':
    filename = "/home/aurora/hdd/workspace/data/mnist/train-images.idx3-ubyte"
    filename2 = "/home/aurora/hdd/workspace/data/mnist/train-labels.idx1-ubyte"
    # read_image(filename)
    read_label(filename2, 'mnist_label/label.txt')