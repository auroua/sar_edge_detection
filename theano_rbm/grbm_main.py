import GRBM
from GRBM import RBMrv_T

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import copy as cp
from copy import deepcopy
import pickle as pkl
import time
import theano
from theano import tensor as T, function, printing
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.compile.nanguardmode import NanGuardMode
import theano.sandbox.cuda as cuda
import os
import cPickle, gzip
from IPython.display import Math
import pylab

pylab.rcParams['figure.figsize'] = (10, 10)


def main():
    # network/training parameters:
    noOfVisibleUnits = 361  # have to be perfect square
    noOfHiddenUnits = 256  # 625  #have to be perfect square
    contrastiveDivergence_n = 15  # no of iterations in truncated MCMC chain, can change this inbetween loadParameters() and saveParameters()
    noOfExamples = 2400  # len(allimages)
    batch_size = noOfExamples
    noOfEpoch = 100
    miniBatchSize = 50

    r_a = 0.001  # training rate for a
    r_b = 0.001  # training rate for b
    r_omg = 0.00005  # trianing rate for omega
    r_sig = 0.00005  # training rate for sigma

    # load CBCL images in matrix
    path = '/home/aurora/hdd/workspace/data/face/train/face'
    allimages = []
    for filename in os.listdir(path):
        with open(path + "/" + filename, 'r') as infile:
            header = infile.readline()
            header = infile.readline()
            header = infile.readline()
            image = np.fromfile(infile, dtype=np.uint8).reshape(1, 361)
            allimages = allimages + [image]
            infile.close()
    ### print some images to have a look:
    egfig, myegAxis = plt.subplots(8, 8)
    print("ploting some training examples, close pop up to continue.")
    for j in range(0, 8):
        for i in range(0, 8):
            myegAxis[j][i].imshow(allimages[j * 10 + i].reshape((19, 19)), cmap=cm.Greys_r, interpolation='nearest')
    plt.show()

    theano.config.exception_verbosity = 'high'
    plt.close("all")
    print("RBM with real visible units")

    myRBMrv = RBMrv_T(noOfVisibleUnits=noOfVisibleUnits, noOfHiddenUnits=noOfHiddenUnits, CD_n=contrastiveDivergence_n,
                      aRate=r_a, bRate=r_b, omegaRate=r_omg, sigmaRate=r_sig, rprop_e=0.05, rprop_en=0.025,
                      sparseTargetp=0.05)

    imgArray = np.asarray(allimages, theano.config.floatX)[0:batch_size].reshape(batch_size, 361)
    miniBatch = theano.shared(np.float32(imgArray / 255.0), borrow=True, allow_downcast=True)

    print("imgArray.shape=" + str(imgArray.shape))

    myRBMrv.loadParameters('./data/face/CBCLTrial_256_2500_A')

    myRBMrv.trainMB(miniBatch.eval(), noOfEpoch, miniBatchSize)

    print("plot network parameters, close pop ups to continue.")
    myRBMrv.plotAllRF(64)
    myRBMrv.plotSD()
    myRBMrv.plot_a()
    myRBMrv.plot_b()
    noOfgenSamples = 64
    sampleSkip = 2102
    fig, myAxis = plt.subplots(int(math.sqrt(noOfgenSamples)), int(math.sqrt(noOfgenSamples)))
    xpt, ypt = myAxis.shape
    fig.tight_layout()
    modelSamples = myRBMrv.genSamples(noOfgenSamples, sampleSkip)
    genSampIndex = 0
    print("plot some generated samples")
    for xind in range(0, xpt):
        for yind in range(0, ypt):
            myAxis[xind][yind].imshow(modelSamples[genSampIndex].reshape((myRBMrv.dimV, myRBMrv.dimV)), cmap=cm.Greys_r,
                                      interpolation='nearest')
            genSampIndex = genSampIndex + 1
    plt.show()


if __name__ == "__main__":
    main()