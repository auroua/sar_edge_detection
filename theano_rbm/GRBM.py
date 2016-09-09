#### --------------------------------------- ####
#          Restricted Boltzmann Machine
#        (real visible/binary hidden units)
#  Distributed by Henry Ip via MIT License 2016
#### --------------------------------------- ####

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


class RBMrv_T:
    # class var goes here, instance var goes in constructor

    def __init__(self, noOfVisibleUnits, noOfHiddenUnits, CD_n, aRate, bRate, omegaRate, sigmaRate, omega=None, b=None,
                 a=None, z=None, rprop_e=0.01, rprop_en=0.005, sparseTargetp=0.01):
        '''
        constructor
        RBMrv_T(self, noOfVisibleUnits, noOfHiddenUnits, CD_n, aRate, bRate, omegaRate, sigmaRate, omega=None, b=None, a=None, z=None, rprop_e = 0.01, rprop_en =0.005, sparseTargetp=0.01):

        noOfVisibleUnits (int):         must be perfect square
        noOfHiddenUnits (int):          must be perfect square
        CD_n (int):                     no. of iterations in MCMC simulation during training, check if model means are used if CD_n = 1
        aRate (float32):                update rate of parameter \underline{a} during training
        bRate (float32):                update rate of parameter \underline{b} during training
        omegaRate (float32):            update rate of parameter \boldsymbol{\omega} during training
        sigmaRate (float32):            update rate of parameter \underline{z} during training
        omega (numpy array of float32): \omega parameter matrix with noOfVisible unit rows x noOfHiddenUnits columns
        b (numpy array of float32):     b parameter vector, size = noOfHiddenUnits
        a (numpy array of float32):     b parameter vector, size = noOfVisibleUnits
        z (numpy array of float32):     z parameter vector, size = noOfVisibleUnits
        rprop_e (float32):
        rprop_en (float32):
        sparseTargetp (float32):        target mean hidden unit activation for training. between (0,1)

        '''

        self.epsilon = 0.0000001

        theano.config.exception_verbosity = 'high'
        # rprop parameters and variables, rprop not used
        self.T_rprop_e = theano.shared(value=np.float32(rprop_e), name='T_rprop_e', borrow=True, allow_downcast=True)
        self.T_rprop_en = theano.shared(value=np.float32(rprop_en), name='T_rprop_en', borrow=True, allow_downcast=True)
        self.T_posUpdate = theano.shared(value=np.float32(0.5 * (1.0 + rprop_e)), name='T_posUpdate', borrow=True,
                                         allow_downcast=True)
        self.T_negUpdate = theano.shared(value=np.float32(0.5 * (1.0 - rprop_en)), name='T_negUpdate', borrow=True,
                                         allow_downcast=True)

        # network geometry and training parameters
        self.miniBatchSize = 0  # will be set in self.trainMB(...)
        self.parameterLoaded = False
        self.parameterSaved = False
        self.sparseTargetp = sparseTargetp
        self.CD_n = CD_n
        self.nv = noOfVisibleUnits
        self.nh = noOfHiddenUnits
        self.dimV = int(math.sqrt(self.nv))
        self.dimH = int(math.sqrt(self.nh))
        self.aRate = np.float32(aRate)
        self.bRate = np.float32(bRate)
        self.omegaRate = np.float32(omegaRate)
        self.sigmaRate = np.float32(sigmaRate)
        # initialise v and h
        self.v = np.float32(np.random.uniform(0, 1.0, self.nv))
        self.h = np.float32(np.random.binomial(1.0, 0.5, self.nh))
        self.logLikelihood = []
        self.likelihood4plot = []

        self.T_aRate = theano.shared(value=np.float32(aRate), name='T_aRate', borrow=True, allow_downcast=True)
        self.T_bRate = theano.shared(value=np.float32(bRate), name='T_bRate', borrow=True, allow_downcast=True)
        self.T_omgRate = theano.shared(value=np.float32(omegaRate), name='T_omgRate', borrow=True, allow_downcast=True)
        self.T_sigRate = theano.shared(value=np.float32(sigmaRate), name='T_sigRate', borrow=True, allow_downcast=True)

        self.loadedRates = [aRate, bRate, omegaRate,
                            sigmaRate]  # for load/saveparameters(), can load to see previous rates but differes from constructor declared rates

        self.T_rng = RandomStreams()  # use_cuda parameter set if on GPU
        # succesive calls on this T_rng will keep returning new values, so for MCMC even with
        # same start v vector value called twice consecutively you'll have different outputs
        # this is normal as the same T_rng gets called, without reset, giving different outputs everytime.

        self.T_CD_n = theano.shared(value=CD_n, name='T_CD_n', borrow=True, allow_downcast=True)

        if omega is None:  # careful! use "1.0" instead of "1" below else it all rounds to zeros!!!
            omega = np.float32(
                np.random.uniform((-1.0) * (1.0 / (np.sqrt(self.nh + self.nv))), (1.0 / (np.sqrt(self.nh + self.nv))),
                                  self.nv * self.nh).reshape((self.nv, self.nh)))
        self.omega = omega
        self.T_omega = theano.shared(value=omega, name='T_omega', borrow=True, allow_downcast=True)
        # rprop previous gradient
        self.Tomg_grad_prev = theano.shared(value=np.float32(np.abs(omega * omegaRate) + omegaRate),
                                            name='Tomg_grad_prev', borrow=True, allow_downcast=True)
        # RMSprop accumulated gradient RMS
        self.Tomg_rmsH = theano.shared(value=omega, name='Tomg_rmsH', borrow=True, allow_downcast=True)

        if b is None:
            b = np.float32(np.random.uniform((-1.0) * (1.0 / (self.nv)), (1.0 / (self.nv)), self.nh))
        self.b = b
        self.T_b = theano.shared(value=b, name='T_b', borrow=True, allow_downcast=True)
        # rprop previous gradient
        self.Tb_grad_prev = theano.shared(value=np.float32(np.abs(bRate * b) + bRate), name='Tb_grad_prev', borrow=True,
                                          allow_downcast=True)
        # RMSprop accumulated gradient RMS
        self.Tb_rmsH = theano.shared(value=b, name='Tb_rmsH', borrow=True, allow_downcast=True)

        if a is None:
            a = np.float32(np.random.uniform((-1.0) * (1.0 / (self.nh)), (1.0 / (self.nh)), self.nv))
        self.a = a
        self.T_a = theano.shared(value=a, name='T_a', borrow=True, allow_downcast=True)
        # rprop previous gradient
        self.Ta_grad_prev = theano.shared(value=np.float32(np.abs(aRate * a) + aRate), name='Ta_grad_prev', borrow=True,
                                          allow_downcast=True)
        # RMSprop accumulated gradient RMS
        self.Ta_rms = theano.shared(value=a, name='Ta_rms', borrow=True, allow_downcast=True)

        # for sigma parameter we train z instead with e^z = \sigma^2
        if z is None:
            z = np.float32(np.random.normal(0.0, (1.0 / (self.nh * self.nh)),
                                            self.nv))  # np.asarray([0.0]*self.nv, dtype=theano.config.floatX)
        self.z = z
        self.T_z = theano.shared(value=z, name='T_z', borrow=True, allow_downcast=True)
        self.T_sigmaSqr = T.exp(self.T_z)
        # rprop previous gradient
        self.Tz_grad_prev = theano.shared(value=np.float32(np.float32(np.abs(z * sigmaRate) + sigmaRate)),
                                          name='Tz_grad_prev', borrow=True, allow_downcast=True)
        # RMSprop accumulated gradient RMS
        self.Tz_rmsH = theano.shared(value=z, name='Tz_rmsH', borrow=True, allow_downcast=True)

        self.T_logZk = theano.shared(value=np.float32(0.0), name='T_logZk', borrow=True, allow_downcast=True)

        # will print in ipython notebook:
        print("RBMrv constructed for " + str(len(self.v)) + " visible units and " + str(len(self.h)) + " hidden units.")
        # print(", with Energy function:")
        # display(Math(r'E(\vec{v},\vec{h}) = \sum_i \frac{(v_i-a_i)^2}{2\sigma_i^2} - \sum_i \sum_j \omega_{ij}h_j\frac{v_i}{\sigma_i^2} - \sum_j b_j h_j'))

    def genSamples(self, noOfsamples, separation):
        """
        Generated samples from loaded parameters: genSamples(self, noOfsamples, separation)

        Args:
        separation (int):             number of MCMC separation of samples
        noOFsamples (int):            total number of samples returned

        Return:
        geneartedSamples (np array): if images, use "generatedSamples[#sample].reshape((noOfvisibleUnits,noOfvisibleUnits))" for ploting

        """
        generatedSamples = []
        initSample = T.vector("initSample", dtype=theano.config.floatX)
        [scan_resV, scan_resH, H_meanStub, V_meanStub], scan_updates = theano.scan(self.vtovMBall,
                                                                                   outputs_info=[initSample, None, None,
                                                                                                 None],
                                                                                   n_steps=separation * (
                                                                                   noOfsamples + 1))
        genSampleFn = theano.function(inputs=[initSample], outputs=[scan_resV, scan_resH], allow_input_downcast=True,
                                      updates=scan_updates)

        [currentV, currentH] = genSampleFn(np.asarray([0.0] * self.nv, dtype=theano.config.floatX))
        generatedSamples = currentV[separation:separation * (noOfsamples + 1):separation]

        return generatedSamples

    def checkNaN(self):
        """
        prints NaN tests
        works on parameters a, b, z, omega of current object
        """
        print("NaN test on omega: " + str(np.isnan(np.sum(np.sum(np.asarray(self.T_omega.eval()))))))
        print("NaN test on a: " + str(np.isnan(np.dot(np.asarray(self.T_a.eval()), np.asarray(self.T_a.eval())))))
        print("NaN test on b: " + str(np.isnan(np.dot(np.asarray(self.T_b.eval()), np.asarray(self.T_b.eval())))))
        print("NaN test on z: " + str(np.isnan(np.dot(np.asarray(self.T_z.eval()), np.asarray(self.T_z.eval())))))
        print(
        "max z = " + str(np.max(np.asarray(self.T_z.eval()))) + ", min z =" + str(np.min(np.asarray(self.T_z.eval()))))

    def printParameters(self):
        """
        prints parameters a, b, z \sigma^2, omega
        """
        print("a = " + str(self.T_a.get_value()))
        print("b = " + str(self.T_b.get_value()))
        print("z = " + str(self.T_z.get_value()))
        print("sigma^2 = " + str([math.exp(zi) for zi in self.T_z.get_value()]))
        print("omega = " + str(self.T_omega.get_value()))

    def plotAllRF(self, noOfRFs=25):
        """
        plots \omega_{ij} elements. With i=0,1,... noOfRFs as a square image

        args:
        noOfRFs (int): have to be perfect square and up to number of hidden units
        """

        inputIndex = noOfRFs + 1
        fig, myAxis = plt.subplots(int(np.sqrt(noOfRFs)), int(np.sqrt(noOfRFs)))

        xpt, ypt = myAxis.shape
        fig.tight_layout()
        for xind in range(0, xpt):
            for yind in range(0, ypt):
                myAxis[xind][yind].imshow(self.T_omega.eval()[:, inputIndex].reshape((self.dimV, self.dimV)),
                                          cmap=cm.Greys_r, interpolation='nearest')
                inputIndex = inputIndex + 1
        plt.show()
        # print("weights are between (" + str(np.min(np.min(self.T_omega.eval()))) + "," + str(np.max(np.max(self.T_omega.eval()))) + ")")

    def plotSD(self):
        """
        plot \sigma standard deviation parameter as sqaure image
        """
        SDparameter = np.exp((np.asarray(self.T_z.eval())))
        fig = plt.figure()
        im = plt.imshow(SDparameter.reshape((self.dimV, self.dimV)), cmap=cm.Greys_r, interpolation='nearest')
        fig.colorbar(im)

    def plot_a(self):
        """
        plot a parameter as an image
        """
        SDparameter = np.asarray(self.T_a.eval())
        fig = plt.figure()
        im = plt.imshow(SDparameter.reshape((self.dimV, self.dimV)), cmap=cm.Greys_r, interpolation='nearest')
        fig.colorbar(im)

    def plot_b(self):
        """
        plot b parameter as an image
        """
        SDparameter = np.asarray(self.T_b.eval())
        fig = plt.figure()
        im = plt.imshow(SDparameter.reshape((self.dimH, self.dimH)), cmap=cm.Greys_r, interpolation='nearest')
        fig.colorbar(im)

    def saveParameters(self, fileName):
        """
        saves all essential parameters so simulation can resume after calling loadParameters()
        file saved in npz format

        ars:
        fileName (string): in single quotes '...' and excluding extensions.
        """
        np.savez(fileName, T_omega=self.T_omega.eval(), Tomg_rmsH=self.Tomg_rmsH.eval(),
                 T_a=self.T_a.eval(), Ta_rms=self.Ta_rms.eval(),
                 T_b=self.T_b.eval(), Tb_rmsH=self.Tb_rmsH.eval(),
                 T_z=self.T_z.eval(), Tz_rmsH=self.Tz_rmsH.eval(),
                 Ta_grad_prev=self.Ta_grad_prev.eval(),
                 Tb_grad_prev=self.Ta_grad_prev.eval(),
                 Tz_grad_prev=self.Tz_grad_prev.eval(),
                 Tomg_grad_prev=self.Tomg_grad_prev.eval(),
                 logLikelihood=self.logLikelihood, likelihood4plot=self.likelihood4plot,
                 T_logZk=self.T_logZk.eval(),
                 loadedRates=self.loadedRates, miniBatchSize=self.miniBatchSize,
                 aRate=self.aRate, bRate=self.bRate, omegaRate=self.omegaRate, sigmaRate=self.sigmaRate,
                 CD_n=self.CD_n, sparseTargetp=self.sparseTargetp)
        # print("parameters saved in: " + str(fileName) + ".npz")
        self.parameterSaved = True

    def loadParameters(self, fileName):
        """
        loads npz file to restore all simulation parameters
        make sure the parameters you're loading fits the current object (e.g. same #visible/#hidden units)

        ars:
        fileName (string): in single quotes '...' and excluding extensions.
        """
        loadedFile = np.load(fileName + '.npz')
        self.miniBatchSize = loadedFile['miniBatchSize']
        self.aRate = np.float32(loadedFile['aRate'])  # without explicit cast it turns into float64?!
        self.bRate = np.float32(loadedFile['bRate'])
        self.omegaRate = np.float32(loadedFile['omegaRate'])
        self.sigmaRate = np.float32(loadedFile['sigmaRate'])
        self.CD_n = loadedFile['CD_n']
        self.sparseTargetp = loadedFile['sparseTargetp']
        self.T_omega.set_value(loadedFile['T_omega'])
        self.Tomg_rmsH.set_value(loadedFile['Tomg_rmsH'])
        self.T_a.set_value(loadedFile['T_a'])
        self.Ta_rms.set_value(np.float32(loadedFile['Ta_rms']))
        self.T_b.set_value(loadedFile['T_b'])
        self.Tb_rmsH.set_value(loadedFile['Tb_rmsH'])
        self.T_z.set_value(loadedFile['T_z'])
        self.Tz_rmsH.set_value(loadedFile['Tz_rmsH'])
        self.Ta_grad_prev.set_value(loadedFile['Ta_grad_prev'])
        self.Tb_grad_prev.set_value(loadedFile['Tb_grad_prev'])
        self.Tz_grad_prev.set_value(loadedFile['Tz_grad_prev'])
        self.Tomg_grad_prev.set_value(loadedFile['Tomg_grad_prev'])
        self.logLikelihood = loadedFile['logLikelihood']
        self.likelihood4plot = loadedFile['likelihood4plot']
        self.likelihood4plot = self.likelihood4plot.tolist()
        self.T_logZk.set_value(loadedFile['T_logZk'])
        self.loadedRates = loadedFile['loadedRates']
        # print("after loading, omega = " + str(self.T_omega.eval()))
        self.parameterLoaded = True

    def energyFnMB(self, VM, HM):
        """
        evaluates the energy functions of the RBM given row vector(s) of v and h


        args:
        VM (T.matrix): rows of visible layer values
        HM (T.matrix): rows of hidden layer values

        return:
        a row Theano vector, elements being E(v_row, h_row)
        """
        T_bh = T.dot(HM, self.T_b)
        T_omghv = T.transpose(T.sum(T.mul(T.dot(T.mul(T.fill(VM, T.exp(-self.T_z)), VM), self.T_omega), HM), axis=1,
                                    acc_dtype=theano.config.floatX))
        T_Vsqr = T.mul(VM - T.fill(VM, self.T_a), VM - T.fill(VM, self.T_a))
        T_VsqrOmg = T.transpose(T.sum(T.mul(T.fill(T_Vsqr, np.float32(0.5) * T.exp(-self.T_z)), T_Vsqr), axis=1,
                                      acc_dtype=theano.config.floatX))
        return -T_VsqrOmg + T_omghv + T_bh

    def vtohMB(self, VsampM):
        """
        computes hidden unit outputs given visible unit outputs ("half" a MCMC iteration)
        computes in parallel given input rows of visible units

        args:
        VsampM (T.matrix): rows of visible unit outputs

        returns:
        a T.matrix, rows of hidden unit outputs

        """
        Vomg = T.matrix(name="Vomg", dtype=theano.config.floatX)
        vtohMBres = T.matrix(name="vtohMBres", dtype=theano.config.floatX)
        T_HP = T.matrix(name="T_HP", dtype=theano.config.floatX)

        Vomg = T.dot(T.mul(T.fill(VsampM, T.exp(-self.T_z)), VsampM), self.T_omega)
        T_Hp = T.nnet.ultra_fast_sigmoid(T.fill(Vomg, self.T_b) + Vomg)
        vtohMBres = self.T_rng.binomial(size=T_Hp.shape, p=T_Hp, dtype=theano.config.floatX)
        return vtohMBres

    def vtovMBall(self, VsampM):
        """
        computes visible unit outputs given visible unit inputs (single MCMC iteration)
        multiple paralle MCMC iterations using rows of the input matrix

        args:
        VsampM (T.matrix): rows of this matrix are visible unit inputs

        return:
        ahtovMBres (T.matrix): rows of this matrix are visible unit outputs after a single MCMC iteration
        """
        # v to h part
        aVomg = T.matrix(name="Vomg", dtype=theano.config.floatX)
        avtohMBres = T.matrix(name="vtohMBres", dtype=theano.config.floatX)
        aT_HP = T.matrix(name="T_HP", dtype=theano.config.floatX)

        aVomg = T.dot(T.mul(T.fill(VsampM, T.exp(-self.T_z)), VsampM), self.T_omega)
        aT_Hp = T.nnet.ultra_fast_sigmoid(T.fill(aVomg, self.T_b) + aVomg)
        avtohMBres = self.T_rng.binomial(size=aT_Hp.shape, p=aT_Hp, dtype=theano.config.floatX)

        # h to v part:
        aT_omgH = T.matrix(name="T_omgH", dtype=theano.config.floatX)
        aT_means = T.matrix(name="T_means", dtype=theano.config.floatX)
        ahtovMBres = T.matrix(name="htovMBres", dtype=theano.config.floatX)

        aT_omgH = T.transpose(T.dot(self.T_omega, T.transpose(avtohMBres)))
        aT_means = T.fill(aT_omgH, self.T_a) + aT_omgH
        ahtovMBres = self.T_rng.normal(size=aT_means.shape, avg=aT_means, std=T.fill(aT_means, T.sqrt(T.exp(self.T_z))),
                                       dtype=theano.config.floatX)
        return [ahtovMBres, avtohMBres, aT_Hp, aT_means]

    def htovMB(self, HsampM):
        """
        computes visible unit outputs given hidden unit inputs ("half" a MCMC iteration)
        computes in parallel given input rows of hidden units

        args:
        HsampM (T.matrix): rows of hidden unit inputs

        returns:
        a T.matrix, rows of visible unit outputs

        """

        T_omgH = T.matrix(name="T_omgH", dtype=theano.config.floatX)
        T_means = T.matrix(name="T_means", dtype=theano.config.floatX)
        htovMBres = T.matrix(name="htovMBres", dtype=theano.config.floatX)

        T_omgH = T.transpose(T.dot(self.T_omega, T.transpose(HsampM)))
        T_means = T.fill(T_omgH, self.T_a) + T_omgH
        htovMBres = self.T_rng.normal(size=T_means.shape, avg=T_means, std=T.fill(T_means, T.sqrt(T.exp(self.T_z))),
                                      dtype=theano.config.floatX)
        return htovMBres

    def trainMB(self, V_egMin, noOfEpoch, noOfMiniBatchEx):
        """
        trains the current RBM object, returns nothing with parameter updates being internal

        args:
        V_egMin (theano.shared 2D array): call eval() to supply as argument. rows of this are input examples. V_egMin[N:M] extracts M-N examples, each of size noOfVisible units
        noOfEpoch (int): total number of Epoch to simulate, each Epoch goes through V_egMin
        noOfMiniBatchEx (int): number of examples to be grouped into minibatches

        """
        self.miniBatchSize = noOfMiniBatchEx
        print("size of input example is: " + str(V_egMin.shape))
        V_egM = T.matrix(name="T_egM", dtype=theano.config.floatX)
        [V_CDmAcc, H_CDmAcc, H_CDmean, V_CDmean], scan_updates = theano.scan(self.vtovMBall,
                                                                             outputs_info=[V_egM, None, None, None],
                                                                             n_steps=self.CD_n)
        V_CDm = V_CDmAcc[-1]  # these are matrixes
        H_CDm = H_CDmAcc[-1]  # these are matrixes

        H_egM = self.vtohMB(V_egM)
        energyVector_eg = self.energyFnMB(V_egM, H_egM)
        energyVector_cd = self.energyFnMB(V_CDm, H_CDm)
        costFn = T.mean(energyVector_eg, dtype=theano.config.floatX, acc_dtype=theano.config.floatX) - T.mean(
            energyVector_cd, dtype=theano.config.floatX, acc_dtype=theano.config.floatX)

        Ta_grad, Tb_grad, Tz_grad, Tomg_grad = T.grad(cost=costFn,
                                                      wrt=[self.T_a, self.T_b, self.T_z, self.T_omega],
                                                      consider_constant=[V_egM, H_egM, V_CDm, H_CDm])

        # regular gradient
        gradFromMB = theano.function(inputs=[V_egM], outputs=[Ta_grad, Tb_grad, Tz_grad, Tomg_grad],
                                     allow_input_downcast=True,
                                     updates=scan_updates + [(self.T_a, self.T_a + self.aRate * Ta_grad),
                                                             (self.T_b, self.T_b + self.bRate * Tb_grad),
                                                             (self.T_z, self.T_z + self.sigmaRate * Tz_grad),
                                                             (self.T_omega, self.T_omega + self.omegaRate * Tomg_grad)],
                                     mode='FAST_RUN')  # NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

        # rprop: Code not used
        Ta_rpropMag = T.mul(T.abs_(self.Ta_grad_prev),
                            T.mul(self.T_posUpdate, T.abs_(T.sgn(self.Ta_grad_prev) + T.sgn(Ta_grad))) +
                            T.mul(self.T_negUpdate,
                                  T.abs_(T.abs_(T.sgn(self.Ta_grad_prev) + T.sgn(Ta_grad)) - np.float32(2.0))))
        Ta_rprop = T.mul(T.sgn(Ta_grad), Ta_rpropMag.clip(np.float32(self.epsilon), 50))
        Tb_rpropMag = T.mul(T.abs_(self.Tb_grad_prev),
                            T.mul(self.T_posUpdate, T.abs_(T.sgn(self.Tb_grad_prev) + T.sgn(Tb_grad))) +
                            T.mul(self.T_negUpdate,
                                  T.abs_(T.abs_(T.sgn(self.Tb_grad_prev) + T.sgn(Tb_grad)) - np.float32(2.0))))
        Tb_rprop = T.mul(T.sgn(Tb_grad), Tb_rpropMag.clip(np.float32(self.epsilon), 50))
        Tz_rpropMag = T.mul(T.abs_(self.Tz_grad_prev),
                            T.mul(self.T_posUpdate, T.abs_(T.sgn(self.Tz_grad_prev) + T.sgn(Tz_grad))) +
                            T.mul(self.T_negUpdate,
                                  T.abs_(T.abs_(T.sgn(self.Tz_grad_prev) + T.sgn(Tz_grad)) - np.float32(2.0))))
        Tz_rprop = T.mul(T.sgn(Tz_grad), Tz_rpropMag.clip(np.float32(self.epsilon), 50))
        Tomg_rpropMag = T.mul(T.abs_(self.Tomg_grad_prev),
                              T.mul(self.T_posUpdate, T.abs_(T.sgn(self.Tomg_grad_prev) + T.sgn(Tomg_grad))) +
                              T.mul(self.T_negUpdate,
                                    T.abs_(T.abs_(T.sgn(self.Tomg_grad_prev) + T.sgn(Tomg_grad)) - np.float32(2.0))))
        Tomg_rprop = T.mul(T.sgn(Tomg_grad), Tomg_rpropMag.clip(np.float32(self.epsilon), 50))
        gradFromMBrprop = theano.function(inputs=[V_egM], outputs=[Ta_rprop, Tb_rprop, Tz_rprop, Tomg_rprop],
                                          allow_input_downcast=True,
                                          updates=scan_updates + [(self.T_a, self.T_a + Ta_rprop),
                                                                  (self.T_b, self.T_b + Tb_rprop),
                                                                  (self.T_z, self.T_z + Tz_rprop),
                                                                  (self.T_omega, self.T_omega + Tomg_rprop),
                                                                  (self.Ta_grad_prev, Ta_rprop),
                                                                  (self.Tb_grad_prev, Tb_rprop),
                                                                  (self.Tz_grad_prev, Tz_rprop),
                                                                  (self.Tomg_grad_prev, Tomg_rprop)],
                                          mode='FAST_RUN')  # NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

        # RMSprop only:
        [a_grad, b_grad, z_grad, omg_grad] = gradFromMB(V_egMin[0:noOfMiniBatchEx])  # initial RMS correction
        if (not (self.parameterLoaded) and not (self.parameterSaved)):
            self.Ta_rms.set_value(np.float32(np.abs(
                a_grad)))  # =  theano.shared(value = np.float32(np.abs(a_grad)), name = 'Ta_rms', borrow=True, allow_downcast=True)
        Tb_rms = theano.shared(value=np.float32(np.abs(b_grad)), name='Tb_rms', borrow=True, allow_downcast=True)
        Tz_rms = theano.shared(value=np.float32(np.abs(z_grad)), name='Tz_rms', borrow=True, allow_downcast=True)
        Tomg_rms = theano.shared(value=np.float32(np.abs(omg_grad)), name='Tomg_rms', borrow=True, allow_downcast=True)
        gradFromMBRMSprop = theano.function(inputs=[V_egM], outputs=[Ta_grad, Tb_grad, Tz_grad, Tomg_grad],
                                            allow_input_downcast=True,
                                            updates=scan_updates + [(self.Ta_rms, T.sqrt(
                                                T.mul(np.float32(0.9), T.mul(self.Ta_rms, self.Ta_rms)) + T.mul(
                                                    np.float32(0.1), T.mul(Ta_grad, Ta_grad)))),
                                                                    (Tb_rms, T.sqrt(T.mul(np.float32(0.9), T.mul(Tb_rms,
                                                                                                                 Tb_rms)) + T.mul(
                                                                        np.float32(0.1), T.mul(Tb_grad, Tb_grad)))),
                                                                    (Tz_rms, T.sqrt(T.mul(np.float32(0.9), T.mul(Tz_rms,
                                                                                                                 Tz_rms)) + T.mul(
                                                                        np.float32(0.1), T.mul(Tz_grad, Tz_grad)))),
                                                                    (Tomg_rms, T.sqrt(T.mul(np.float32(0.9),
                                                                                            T.mul(Tomg_rms,
                                                                                                  Tomg_rms)) + T.mul(
                                                                        np.float32(0.1), T.mul(Tomg_grad, Tomg_grad)))),
                                                                    (self.T_a, self.T_a + self.aRate * T.mul(Ta_grad,
                                                                                                             T.maximum(
                                                                                                                 np.float32(
                                                                                                                     self.epsilon),
                                                                                                                 self.Ta_rms) ** -1)),
                                                                    (self.T_b, self.T_b + self.bRate * T.mul(Tb_grad,
                                                                                                             T.maximum(
                                                                                                                 np.float32(
                                                                                                                     self.epsilon),
                                                                                                                 Tb_rms) ** -1)),
                                                                    (self.T_z,
                                                                     self.T_z + self.sigmaRate * T.mul(Tz_grad,
                                                                                                       T.maximum(
                                                                                                           np.float32(
                                                                                                               self.epsilon),
                                                                                                           Tz_rms) ** -1)),
                                                                    (self.T_omega,
                                                                     self.T_omega + self.omegaRate * T.mul(Tomg_grad,
                                                                                                           T.maximum(
                                                                                                               np.float32(
                                                                                                                   self.epsilon),
                                                                                                               Tomg_rms) ** -1))],
                                            mode='FAST_RUN')  # NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

        # sparse hidden units optimization + RMSprop:
        # first calculate probability of hidden units firing given visible examples:
        aVomg = T.dot(T.mul(T.fill(V_egM, T.exp(-self.T_z)), V_egM), self.T_omega)
        aT_Hp = T.nnet.sigmoid(T.fill(aVomg, self.T_b) + aVomg)  # T.nnet.ultra_fast_sigmoid() did not work for us
        aT_HpMean = T.mean(aT_Hp)  # mean activation over minibatch and all Hk
        # cross entropy between mean hidden unit activation and target mean activation probability "self.sparseTargetp"
        sparseHcost = T.mul(np.float32(-self.sparseTargetp), T.log(aT_HpMean)) - T.mul(
            (np.float32(1.0) - self.sparseTargetp), T.log(np.float32(1.0) - aT_HpMean))

        Tb_gradH, Tz_gradH, Tomg_gradH = T.grad(cost=sparseHcost,
                                                wrt=[self.T_b, self.T_z, self.T_omega],
                                                consider_constant=[V_egM])
        sparseGradFn = theano.function(inputs=[V_egM], outputs=[Tb_gradH, Tz_gradH, Tomg_gradH],
                                       allow_input_downcast=True, mode='FAST_RUN')

        [b_gradH, z_gradH, omg_gradH] = sparseGradFn(V_egMin[0:noOfMiniBatchEx])  # initial RMS correction

        if (not (self.parameterLoaded) and not (self.parameterSaved)):
            self.Tb_rmsH.set_value(np.float32(np.abs(b_grad - b_gradH)))
            self.Tz_rmsH.set_value(np.float32(np.abs(z_grad - z_gradH)))
            self.Tomg_rmsH.set_value(np.float32(np.abs(omg_grad - omg_gradH)))
        gradSparseH = theano.function(inputs=[V_egM],
                                      outputs=[Ta_grad, Tb_grad, Tz_grad, Tomg_grad, Tb_gradH, Tz_gradH, Tomg_gradH],
                                      allow_input_downcast=True,
                                      updates=scan_updates + [(self.Ta_rms, T.sqrt(
                                          T.mul(np.float32(0.9), T.mul(self.Ta_rms, self.Ta_rms)) + T.mul(
                                              np.float32(0.1), T.mul(Ta_grad, Ta_grad)))),
                                                              (self.Tb_rmsH, T.sqrt(T.mul(np.float32(0.9),
                                                                                          T.mul(self.Tb_rmsH,
                                                                                                self.Tb_rmsH)) + T.mul(
                                                                  np.float32(0.1),
                                                                  T.mul(Tb_grad - Tb_gradH, Tb_grad - Tb_gradH)))),
                                                              (self.Tz_rmsH, T.sqrt(T.mul(np.float32(0.9),
                                                                                          T.mul(self.Tz_rmsH,
                                                                                                self.Tz_rmsH)) + T.mul(
                                                                  np.float32(0.1),
                                                                  T.mul(Tz_grad - Tz_gradH, Tz_grad - Tz_gradH)))),
                                                              (self.Tomg_rmsH, T.sqrt(T.mul(np.float32(0.9),
                                                                                            T.mul(self.Tomg_rmsH,
                                                                                                  self.Tomg_rmsH)) + T.mul(
                                                                  np.float32(0.1), T.mul(Tomg_grad - Tomg_gradH,
                                                                                         Tomg_grad - Tomg_gradH)))),
                                                              (self.T_a, self.T_a + self.aRate * T.mul(Ta_grad,
                                                                                                       T.maximum(
                                                                                                           np.float32(
                                                                                                               self.epsilon),
                                                                                                           self.Ta_rms) ** -1)),
                                                              (self.T_b,
                                                               self.T_b + self.bRate * T.mul(Tb_grad - Tb_gradH,
                                                                                             T.maximum(np.float32(
                                                                                                 self.epsilon),
                                                                                                       self.Tb_rmsH) ** -1)),
                                                              (self.T_z,
                                                               self.T_z + self.sigmaRate * T.mul(Tz_grad - Tz_gradH,
                                                                                                 T.maximum(np.float32(
                                                                                                     self.epsilon),
                                                                                                           self.Tz_rmsH) ** -1)),
                                                              (self.T_omega, self.T_omega + self.omegaRate * T.mul(
                                                                  Tomg_grad - Tomg_gradH,
                                                                  T.maximum(np.float32(self.epsilon),
                                                                            self.Tomg_rmsH) ** -1))],
                                      mode='FAST_RUN')  # NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

        # reconstruction errors:
        [V_egM_recon, H_egM_reconStub, H_meanStubC, V_meanStubC] = self.vtovMBall(V_egM)
        V_error = V_egM - V_egM_recon
        V_errorSqr = T.mul(V_error, V_error)
        reconError = theano.function(inputs=[V_egM], outputs=[
            T.mean(T.sum(V_errorSqr, axis=1, acc_dtype=theano.config.floatX), acc_dtype=theano.config.floatX)],
                                     allow_input_downcast=True,
                                     mode='FAST_RUN')

        print("***************************************************************************************************")
        print(
        "training network with " + str(self.nv) + " real visible units and " + str(self.nh) + " binary hidden units")
        print("reconstruction error before training = " + str(np.array(reconError(V_egMin))[0]))
        noOfMiniBatches = np.int(len(V_egMin) / noOfMiniBatchEx)
        print("number of mini-batches = " + str(noOfMiniBatches) + ", with " + str(
            noOfMiniBatchEx) + " examples per mini-batch")
        print("number of Epochs = " + str(noOfEpoch))
        print("***************************************************************************************************")

        # input images already randomised with consecutive images belonging to different class, use directly as minibatch.
        for j in xrange(noOfEpoch):
            pretime = time.time()
            for i in xrange(noOfMiniBatches):
                [a_upDate, b_upDate, z_upDate, omg_upDate, b_upDateH, z_upDateH, omg_upDateH] = gradSparseH(
                    V_egMin[i * noOfMiniBatchEx:(i + 1) * noOfMiniBatchEx])

            myErr = reconError(V_egMin)
            self.likelihood4plot = self.likelihood4plot + [np.float32(myErr)]
            print("epoch " + str(j) + ": reconstruction error = " + str(myErr[0]) + ", time taken = " + str(
                time.time() - pretime))

        print("\n***************************************************************************************************")
        print("reconstruction error after training for " + str(noOfEpoch) + " epochs = " + str(
            np.array(reconError(V_egMin))[0]))
        self.checkNaN()
        print("***************************************************************************************************")

        plt.figure
        plt.plot(np.arange(0.0, len(self.likelihood4plot), 1), self.likelihood4plot)
        plt.show()