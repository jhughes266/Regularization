import numpy as np
import math
from ActivationFunctions.Sigmoid import Sigmoid
from ActivationFunctions.Tanh import Tanh
class DenseLayer():
    def __init__(self, xShape, yShape, activationFunctionType, batchSize, lr):
        #storing input and output shapes for later use within class
        self._xShape = xShape
        self._yShape = yShape
        #learning rate, batch size and whether scaling and clipping is active
        self._batchSize = batchSize
        self._lr = lr
        #calculating xavier standard deviation
        self._sigma = math.sqrt((2 / (xShape + yShape)))
        #weight and bias init
        self._w = np.random.normal(0, self._sigma, size=(yShape, xShape))
        self._b = np.zeros((yShape, 1))
        #weight and bias gradient storage vars
        self._dw = np.zeros((yShape, xShape))
        self._db = np.zeros((yShape, 1))
        #activation function
        self._actFunc = self._initActivationFunction(activationFunctionType)
        #storage var
        self._xStore = 0

    def _initActivationFunction(self, activationFunctionType):

        if activationFunctionType == "Sigmoid":
            return Sigmoid()
        elif activationFunctionType == "Tanh":
            return Tanh()
        else:
            raise ValueError("Invalid Activation function type entered")

    def Forward(self, x):
        #y AND z DO NOT NEED TO BE STORED AS THE ACTIVATION FUNCTION CLASS HANDELS THE STORAGE
        #weighted sum
        z = self._w @ x + self._b
        y = self._actFunc.Forward(z)
        #storeing
        self._xStore = x
        return y

    def Backward(self, dy):
        #activation function handels dy multiplication
        dz = self._actFunc.Backward(dy)
        self._dw += (dz @ self._xStore.T)
        self._db += dz
        dx = self._w.T @ dz
        return dx

    def GradientClipping(self, clippingType, clipThreshold=1):
        if clippingType == "Clip":
            self._dw = np.where(self._dw > clipThreshold, clipThreshold, self._dw)
            self._dw = np.where(self._dw < (-1 * clipThreshold), -1 * clipThreshold, self._dw)

            self._db = np.where(self._db > clipThreshold, clipThreshold, self._db)
            self._db = np.where(self._db < (-1 * clipThreshold), -1 * clipThreshold, self._db)
        elif clippingType == "Norm":

            self._gradientNorm = np.linalg.norm(np.concatenate((self._dw.reshape((-1, 1)), self._db.reshape((-1, 1))), axis=0))
            if self._gradientNorm > clipThreshold:
                scaleFactor = (clipThreshold / self._gradientNorm)
                self._dw = self._dw * scaleFactor
                self._db = self._db * scaleFactor
        elif clippingType == "None":
            pass
        else:
            raise ValueError("Invalid Clipping type")

    def GradientStep(self, L1=False, L2=False,lam=0):
        L1WGrad = 0
        L2WGrad = 0
        if L1:
            L1WGrad = lam * np.sign(self._w) * self._lr
        elif L2:
            L2WGrad = 2 * lam * self._w * self._lr

        gradientScalar = (1 / self._batchSize) * self._lr
        self._w -= ((self._dw * gradientScalar) + L1WGrad + L2WGrad)
        self._b -= (self._db * gradientScalar)

    def Reset(self):
        self._dw = np.zeros((self._yShape, self._xShape))
        self._db = np.zeros((self._yShape, 1))


